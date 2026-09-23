"""AFA training in the paper's setting: multiple realistic finetunes on JourneyDB.

Adapted from train_overnight.py with what this run needs:
  * N experts (the paper's set) instead of two
  * gradient accumulation -> effective batch 8 while keeping per-step memory small
  * gradient checkpointing (a 15 GB T4 cannot hold 5 expert U-Nets' activations)
  * step-based logging of the routing state: mean expert-0 attention weight and
    |attn-0.5| -- the metrics that actually move long before the loss does
  * a fixed eval batch (same images/noise/timestep) whose loss is comparable
    across steps, unlike the noisy per-step training loss
  * step-based checkpoints + --resume, and a time-based stop (--stop_hour)

  python3 train_paper_setting.py ... --dry_run_steps 3     # measure only
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import faulthandler
import signal

import PIL.Image as Image
import PIL.ImageFile as ImageFile
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from datasets import load_dataset
from diffusers.optimization import get_scheduler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import Model  # noqa: E402
from models.modules.aggregator import Aggregator  # noqa: E402

# --cudnn on/off decides the real setting; off is only the workaround for the
# T4 container's broken cuDNN (which this machine does not have).
torch.backends.cudnn.enabled = True
faulthandler.register(signal.SIGUSR1, all_threads=True)  # kill -USR1 PID dumps the stack

ASSETS = os.environ.get('AFA_ASSETS', '/data/afa-assets')

# JourneyDB contains a small number of truncated JPEGs; without this a single bad file
# kills a DataLoader worker and aborts the run (observed at step ~270 of run 1).
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_image(path, resolution=512):
    try:
        return Image.open(path).convert('RGB')
    except Exception as exc:  # unreadable file: keep the batch shape, drop the content
        print(f'  WARNING: unreadable image {path}: {type(exc).__name__}: {exc}', flush=True)
        return Image.new('RGB', (resolution, resolution))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_files', type=str, required=True, help='comma-separated expert paths')
    p.add_argument('--st_model_file', type=str, default=f'{ASSETS}/models/stable-diffusion-v1-5')
    p.add_argument('--dataset_json_file', type=str, default=f'{ASSETS}/data/journeydb_data.json')
    p.add_argument('--model_save_path', type=str, required=True)
    p.add_argument('--resolution', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--grad_accum_steps', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--gradient_checkpointing', action='store_true')
    p.add_argument('--aggregator_hidden_size', type=int, default=128)
    p.add_argument('--aggregator_num_layers', type=int, default=1)
    p.add_argument('--aggregator_num_attn_heads', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--lr_warmup_steps', type=int, default=100)
    p.add_argument('--max_optimizer_steps', type=int, default=1000000)
    p.add_argument('--save_every_steps', type=int, default=250)
    p.add_argument('--log_every_steps', type=int, default=10)
    p.add_argument('--eval_every_steps', type=int, default=50)
    p.add_argument('--stop_hour', type=int, default=None)
    p.add_argument('--stop_minute', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--cudnn', choices=['on', 'off'], default='off',
                   help='off = workaround for the broken cuDNN in this container; on = normal GPUs')
    p.add_argument('--allow_tf32', action='store_true')
    p.add_argument('--dry_run_steps', type=int, default=0, help='measure memory/time for N steps then exit')
    p.add_argument('--log_file', type=str, default=None)
    return p.parse_args()


def setup_logging(log_file):
    if not log_file:
        return
    os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
    f = open(log_file, 'a')

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s); f.write(s); sys.__stdout__.flush(); f.flush()

        def flush(self):
            sys.__stdout__.flush(); f.flush()
    sys.stdout = Tee()
    sys.stderr = Tee()


def stop_target(hour, minute):
    now = datetime.now()
    t = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return t + timedelta(days=1) if t <= now else t


def get_dataset(path, tokenizer, resolution, drop_text_rate=0.1):
    tf = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    ds = load_dataset('json', data_files=path, split='train').select_columns(['image_file', 'text'])

    def transform(batch):
        imgs = [load_image(f, resolution) for f in batch['image_file']]
        text = [''] * len(batch['text']) if torch.rand(1).item() < drop_text_rate else batch['text']
        return {'pixel_values': torch.stack([tf(im) for im in imgs], dim=0),
                'input_ids': tokenizer(text, padding='max_length', truncation=True,
                                       return_tensors='pt')['input_ids']}
    ds.set_transform(transform)
    return ds


def forward_with_maps(model, pixel_values, input_ids):
    """Model.train_forward, but returning the attention maps as well."""
    with torch.autocast(device_type=pixel_values.device.type, enabled=False):
        latents = model.vae.encode(pixel_values.to(dtype=model.vae.dtype)).latent_dist.sample()
        latents = latents * model.vae.config.scaling_factor
    latents = latents.to(dtype=model.unets[0].dtype)
    noise = torch.randn_like(latents)
    ts = torch.randint(0, model.train_scheduler.config.num_train_timesteps,
                       (latents.shape[0],), device=latents.device).long()
    noisy = model.train_scheduler.add_noise(latents, noise, ts)
    ehs = torch.stack([te(input_ids)[0] for te in model.text_encoders], dim=0)
    out = model.unets[0](noisy, ts, ehs, additional_unets=model.unets[1:],
                         aggregators=model.aggregators)
    return F.mse_loss(out.sample.float(), noise.float(), reduction='mean'), out.attn_maps


def routing_stats(attn_maps):
    w0 = float(torch.stack([m.float()[:, 0].mean() for m in attn_maps]).mean())
    dev = float(torch.stack([(m.float() - 0.5).abs().mean() for m in attn_maps]).mean())
    return w0, dev


def save_ckpt(model, path, step, epoch, history):
    os.makedirs(path, exist_ok=True)
    for i, agg in enumerate(model.aggregators):
        agg.save_pretrained(os.path.join(path, f'aggregator_{i}'))
    json.dump(dict(step=step, epoch=epoch, history=history, time=datetime.now().isoformat()),
              open(os.path.join(path, 'train_meta.json'), 'w'), indent=1)


def load_ckpt(model, path):
    meta_path = os.path.join(path, 'train_meta.json')
    if not os.path.exists(meta_path):
        return 0, 0, []
    meta = json.load(open(meta_path))
    for i, agg in enumerate(model.aggregators):
        saved = Aggregator.load_pretrained(os.path.join(path, f'aggregator_{i}'))
        # Resuming with a different architecture fails inside load_state_dict with a bare
        # size error; say which flags actually match the checkpoint instead.
        for key, flag in (('num_layers', '--aggregator_num_layers'),
                          ('hidden_size', '--aggregator_hidden_size'),
                          ('num_attn_heads', '--aggregator_num_attn_heads'),
                          ('num_experts', '--model_files')):
            if saved.config.get(key) != agg.config.get(key):
                raise SystemExit(
                    f'checkpoint aggregator_{i} has {key}={saved.config.get(key)} but this run is '
                    f'configured with {key}={agg.config.get(key)}; restart with the checkpoint value '
                    f'({flag} {saved.config.get(key)})')
        agg.load_state_dict(saved.state_dict())
    print(f'resumed from step {meta["step"]} (epoch {meta["epoch"]})')
    return meta['step'], meta['epoch'], meta.get('history', [])


def main():
    args = parse_args()
    setup_logging(args.log_file)
    torch.backends.cudnn.enabled = (args.cudnn == 'on')
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    device = torch.device('cuda')
    print(f'{"=" * 70}\nstart {datetime.now():%Y-%m-%d %H:%M:%S} | '
          f'experts={len(args.model_files.split(","))} | batch {args.batch_size} x {args.grad_accum_steps} '
          f'= eff {args.batch_size * args.grad_accum_steps} | res {args.resolution} | lr {args.lr} | '
          f'grad-ckpt {args.gradient_checkpointing} | cudnn {args.cudnn} | tf32 {args.allow_tf32}')

    tgt = stop_target(args.stop_hour, args.stop_minute) if args.stop_hour is not None else None
    print(f'stop target: {tgt:%Y-%m-%d %H:%M} ' if tgt else 'no time limit', flush=True)

    print('loading experts (see per-expert lines below)...', flush=True)
    model = Model.load_init(model_paths=args.model_files.split(','), st_model_path=args.st_model_file,
                            hidden_size=args.aggregator_hidden_size, num_layers=args.aggregator_num_layers,
                            num_attn_heads=args.aggregator_num_attn_heads)
    model.vae.requires_grad_(False); model.text_encoders.requires_grad_(False)
    model.unets.requires_grad_(False); model.aggregators.requires_grad_(True)
    model.vae.to(device, torch.float32)
    model.text_encoders.to(device, torch.float16)
    model.unets.to(device, torch.float16)
    model.aggregators.to(device, torch.float32)
    # xformers is slower than PyTorch SDPA here and is not installed; disable it.
    for u in model.unets:
        try:
            u.disable_xformers_memory_efficient_attention()
        except Exception:
            pass
    for agg in model.aggregators:
        try:
            agg.transformer.disable_xformers_memory_efficient_attention()
        except Exception:
            pass
    if args.gradient_checkpointing:
        for u in model.unets:
            u.enable_gradient_checkpointing()
        # Transformer2DModel in diffusers 0.21 does not support checkpointing;
        # the aggregator's own activations are small, so the UNet blocks are enough.
        for agg in model.aggregators:
            try:
                agg.transformer.enable_gradient_checkpointing()
            except Exception as e:
                print(f'  (aggregator checkpointing unavailable: {type(e).__name__})', flush=True)
    print('load_init done; moving components to cuda...', flush=True)
    n_agg = sum(p.numel() for p in model.aggregators.parameters())
    print(f'experts loaded; trainable aggregator params {n_agg / 1e6:.2f}M; '
          f'experts: {[os.path.basename(p) for p in args.model_files.split(",")]}', flush=True)

    # ---- fixed eval batch: same images/noise/timestep every time ----
    recs = [json.loads(l) for l in open(args.dataset_json_file)][-2:]
    tf = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    torch.manual_seed(0)
    ev_px = torch.stack([tf(load_image(r['image_file'], args.resolution)) for r in recs]).to(device, torch.float32)
    with torch.autocast(device_type='cuda', enabled=False):
        ev_lat = (model.vae.encode(ev_px.to(model.vae.dtype)).latent_dist.sample()
                  * model.vae.config.scaling_factor).to(torch.float16)
    torch.manual_seed(12345)
    ev_noise = torch.randn_like(ev_lat)
    ev_ts = torch.full((ev_lat.shape[0],), 500, dtype=torch.long, device=device)
    ev_noisy = model.train_scheduler.add_noise(ev_lat, ev_noise, ev_ts)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        ev_ehs = torch.stack([te(model.tokenizer([r['text'] for r in recs], padding='max_length',
                                                truncation=True,
                                                return_tensors='pt')['input_ids'].to(device))[0]
                              for te in model.text_encoders], dim=0).half()

    def run_eval():
        model.eval()
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            out = model.unets[0](ev_noisy, ev_ts, ev_ehs, additional_unets=model.unets[1:],
                                 aggregators=model.aggregators)
            loss = F.mse_loss(out.sample.float(), ev_noise.float(), reduction='mean')
        w0, dev = routing_stats(out.attn_maps)
        model.train()
        return loss.item(), w0, dev

    # ---- data / optimiser ----
    ds = get_dataset(args.dataset_json_file, model.tokenizer, args.resolution)

    start_step, epoch, history = (0, 0, [])
    if args.resume:
        start_step, epoch, history = load_ckpt(model, args.model_save_path)

    try:
        torch.manual_seed(7 + start_step)
        loader = iter(data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, drop_last=True))
        optimizer = optim.AdamW(model.aggregators.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = get_scheduler('constant', optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps)
        scaler = torch.amp.GradScaler('cuda')
        for _ in range(start_step):
            scheduler.step()
        model.train()
        l0, w0_0, dev0 = run_eval()
        print(f'step {start_step:6d} | eval_loss={l0:.5f} | expert0_weight={w0_0:.3f} | |attn-0.5|={dev0:.3f}'
              f'{"  (initialisation)" if start_step == 0 else ""}', flush=True)

        step = start_step
        win = []
        t_log = time.time()
        while step < start_step + args.max_optimizer_steps:
            if tgt is not None and datetime.now() >= tgt - timedelta(minutes=2):
                print(f'[{datetime.now():%H:%M}] stop time reached, saving and exiting', flush=True)
                break
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            for micro in range(args.grad_accum_steps):
                try:
                    batch = next(loader)
                except StopIteration:
                    epoch += 1
                    loader = iter(data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=args.num_workers, drop_last=True))
                    batch = next(loader)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss, maps = forward_with_maps(model, batch['pixel_values'].to(device, torch.float32),
                                                   batch['input_ids'].to(device))
                scaler.scale(loss / args.grad_accum_steps).backward()
                micro_losses.append(loss.item())
                if micro == args.grad_accum_steps - 1:
                    w0, dev = routing_stats(maps)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            win.append(dict(loss=sum(micro_losses) / len(micro_losses), w0=w0, dev=dev))

            if step % args.log_every_steps == 0:
                e_loss, e_w0, e_dev = run_eval()
                history.append(dict(step=step, epoch=epoch,
                                    train_loss=sum(w['loss'] for w in win) / len(win),
                                    train_w0=sum(w['w0'] for w in win) / len(win),
                                    train_dev=sum(w['dev'] for w in win) / len(win),
                                    eval_loss=e_loss, eval_w0=e_w0, eval_dev=e_dev))
                print(f'step {step:6d} | loss={history[-1]["train_loss"]:.5f} w0={history[-1]["train_w0"]:.3f} '
                      f'dev={history[-1]["train_dev"]:.3f} | EVAL loss={e_loss:.5f} w0={e_w0:.3f} dev={e_dev:.3f} | '
                      f'{(time.time() - t_log) / args.log_every_steps:.2f}s/step | epoch {epoch} | '
                      f'{datetime.now():%H:%M:%S}', flush=True)
                t_log = time.time()

                if args.dry_run_steps and step >= start_step + args.dry_run_steps:
                    print(f'DRY RUN: peak_alloc={torch.cuda.max_memory_allocated() / 2**30:.2f} GB '
                          f'peak_reserved={torch.cuda.max_memory_reserved() / 2**30:.2f} GB', flush=True)
                    print(f'DRY RUN grads nonzero: ' + ', '.join(
                        f'{n}={float(p.grad.abs().max()) if p.grad is not None else "None":.2e}'
                        for n, p in list(model.aggregators[0].named_parameters())[:3] +
                        list(model.aggregators[-1].named_parameters())[:3]), flush=True)
                    return

            if step % args.save_every_steps == 0:
                save_ckpt(model, args.model_save_path, step, epoch, history)
                print(f'  checkpoint saved at step {step} -> {args.model_save_path}', flush=True)

        save_ckpt(model, args.model_save_path, step, epoch, history)
        print(f'done {datetime.now():%Y-%m-%d %H:%M:%S} at step {step} (epoch {epoch})', flush=True)
    finally:
        if os.path.exists(args.log_file or ''):
            pass


if __name__ == '__main__':
    main()
