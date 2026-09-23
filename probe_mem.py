"""Memory and speed benchmark after installing xformers.

Runs real training steps (forward + backward + AdamW) at several batch sizes
with the exact dtypes train_overnight.py uses, and reports peak GPU memory and
seconds/step. Also reports which attention processor is active, so we can tell
whether xformers is really being used.

  python3 probe_mem.py --cudnn off --batches 2,4,8

Baseline (before xformers, cudnn disabled): 11.0 GB peak, 1.91 s/step, 22.3 min/epoch @ batch 2.
"""
import argparse
import json
import os
import sys
import time

import PIL.Image as Image
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from datasets import load_dataset
from diffusers.optimization import get_scheduler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import Model  # noqa: E402

ASSETS = os.environ.get('AFA_ASSETS', '/data/afa-assets')
DATASET = f'{ASSETS}/data/journeydb_data.json'
RES = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--filter', default='')
    args = ap.parse_args()

    CONFIGS = [
        ('cudnnOFF_xformersON_b2',  False, True,  2, False),
        ('cudnnOFF_xformersOFF_b2', False, False, 2, False),
        ('cudnnOFF_xformersON_b4_ckpt', False, True, 4, True),
        ('cudnnON_xformersON_b2',   True,  True,  2, False),   # last: may hit the cuDNN bug
        ('cudnnOFF_xformersOFF_b8_ckpt', False, False, 8, True),
    ]
    if args.filter:
        CONFIGS = [c for c in CONFIGS if args.filter in c[0]]

    print(f'start {time.strftime("%H:%M:%S")} configs={[c[0] for c in CONFIGS]}', flush=True)
    device = torch.device('cuda')

    model = Model.load_init(
        model_paths=[f'{ASSETS}/models/realistic_vision_v5.1/Realistic_Vision_V5.1.safetensors',
                     f'{ASSETS}/models/sd15_expert_2.safetensors'],
        st_model_path=f'{ASSETS}/models/stable-diffusion-v1-5',
        hidden_size=128, num_layers=1, num_attn_heads=8)
    model.vae.requires_grad_(False); model.text_encoders.requires_grad_(False)
    model.unets.requires_grad_(False); model.aggregators.requires_grad_(True)
    model.vae.to(device, torch.float32)
    model.text_encoders.to(device, torch.float16)
    model.unets.to(device, torch.float16)
    model.aggregators.to(device, torch.float32)
    model.train()

    def proc(m):
        if hasattr(m, 'transformer_blocks'):
            return type(m.transformer_blocks[0].attn1.processor).__name__
        try:
            return type(m.attn1.processor).__name__
        except Exception:
            return '?'

    print(f'attention processors: unet0.attn={proc(model.unets[0].down_blocks[0].attentions[0])} '
          f'unet1.attn={proc(model.unets[1].down_blocks[0].attentions[0])} '
          f'aggregator0={proc(model.aggregators[0].transformer.transformer_blocks[0])}', flush=True)

    tf = transforms.Compose([
        transforms.Resize(RES, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(RES),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    ds = load_dataset('json', data_files=DATASET, split='train').select_columns(['image_file', 'text'])

    def t(batch):
        ims = [Image.open(f).convert('RGB') for f in batch['image_file']]
        return {'pixel_values': torch.stack([tf(im) for im in ims]),
                'input_ids': model.tokenizer(batch['text'], padding='max_length', truncation=True,
                                             return_tensors='pt')['input_ids']}
    ds.set_transform(t)

    for name, cudnn, xf, batch_size, ckpt in CONFIGS:
        torch.backends.cudnn.enabled = cudnn
        for u in model.unets:
            (u.enable_xformers_memory_efficient_attention() if xf
             else u.disable_xformers_memory_efficient_attention())
            (u.enable_gradient_checkpointing() if ckpt else u.disable_gradient_checkpointing())
        for agg in model.aggregators:
            (agg.transformer.enable_xformers_memory_efficient_attention() if xf
             else agg.transformer.disable_xformers_memory_efficient_attention())
        print(f'\n=== {name} (batch {batch_size}, grad-ckpt {ckpt}) | '
              f'unet0={proc(model.unets[0].down_blocks[0].attentions[0])} '
              f'aggregator0={proc(model.aggregators[0].transformer)}', flush=True)
        opt = optim.AdamW(model.aggregators.parameters(), lr=1e-4, weight_decay=0.01)
        sched = get_scheduler('constant', optimizer=opt, num_warmup_steps=10)
        scaler = torch.amp.GradScaler('cuda')
        loader = iter(data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True))
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for step in range(args.steps + 1):
                b = next(loader)
                px = b['pixel_values'].to(device, torch.float32)
                ids = b['input_ids'].to(device)
                if step == 0:
                    torch.cuda.synchronize(); t0 = time.time()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss = model.train_forward(px, ids)
                scaler.scale(loss).backward()
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                if step > 0:
                    torch.cuda.synchronize(); times.append(time.time() - t0); t0 = time.time()
            peak_alloc = torch.cuda.max_memory_allocated() / 2**30
            peak_res = torch.cuda.max_memory_reserved() / 2**30
            s_per_step = sum(times) / len(times)
            steps_per_epoch = 20947 // batch_size
            print(f'  peak_alloc={peak_alloc:.2f} GB peak_reserved={peak_res:.2f} GB '
                  f'| {s_per_step:.2f} s/step | {steps_per_epoch} steps/epoch -> '
                  f'{steps_per_epoch * s_per_step / 60:.1f} min/epoch | loss={loss.item():.5f}', flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f'  OOM ({str(e)[:70]})', flush=True)
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f'  ERROR {type(e).__name__}: {str(e)[:110]}', flush=True)
            torch.cuda.empty_cache()
            continue
    print(f'done {time.strftime("%H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
