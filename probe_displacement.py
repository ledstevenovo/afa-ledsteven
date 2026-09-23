"""Does a non-zero conv_out init actually unlock learning in the decision layers?

RMS printed to 4 decimals cannot resolve <1% movement, so this measures the
relative displacement ||theta(t) - theta(0)|| / ||theta(0)|| per parameter group
after 20 optimizer steps (effective batch 8), for out_init_std in {0.0, 0.02}.

Reference: AdamW moves a parameter by ~lr per step when the gradient direction is
consistent, so 20 steps at lr 1e-4 is ~2e-3 absolute -- about 9% of conv_in's
0.0228 RMS. Much less than that means the group is gradient-starved.
"""
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

torch.backends.cudnn.enabled = False
ASSETS = os.environ.get('AFA_ASSETS', '/data/afa-assets')
DATASET_JSON = f'{ASSETS}/data/coco_captions_data.json'
RES, BATCH, ACCUM, STEPS = 512, 2, 4, 20
GROUPS = {
    'conv_in': 'conv_in.weight',
    'res_block.conv1': 'res_block.conv1.weight',
    'transformer.proj_in': 'transformer.proj_in.weight',
    'transformer.block0.ff': 'transformer.transformer_blocks.0.ff.net.0.proj.weight',
    'conv_out': 'conv_out.weight',
}
IDX = [0, 12, 24]


def train_forward(model, px, ids):
    with torch.autocast(device_type=px.device.type, enabled=False):
        lat = model.vae.encode(px.to(dtype=model.vae.dtype)).latent_dist.sample() * model.vae.config.scaling_factor
    lat = lat.to(model.unets[0].dtype)
    noise = torch.randn_like(lat)
    ts = torch.randint(0, model.train_scheduler.config.num_train_timesteps, (lat.shape[0],), device=lat.device).long()
    noisy = model.train_scheduler.add_noise(lat, noise, ts)
    ehs = torch.stack([te(ids)[0] for te in model.text_encoders], dim=0)
    out = model.unets[0](noisy, ts, ehs, additional_unets=model.unets[1:], aggregators=model.aggregators)
    return F.mse_loss(out.sample.float(), noise.float(), reduction='mean')


def load_batch(ds, model):
    def t(batch):
        ims = [Image.open(f).convert('RGB') for f in batch['image_file']]
        tf = transforms.Compose([
            transforms.Resize(RES, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(RES),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        return {'pixel_values': torch.stack([tf(im) for im in ims]),
                'input_ids': model.tokenizer(batch['text'], padding='max_length', truncation=True,
                                             return_tensors='pt')['input_ids']}
    ds.set_transform(t)
    return ds


def main():
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
    init = [{k: v.detach().clone() for k, v in agg.state_dict().items()} for agg in model.aggregators]
    print(f'start {time.strftime("%H:%M:%S")}; {STEPS} steps, eff batch {ACCUM * BATCH}, lr 1e-4', flush=True)

    ds = load_batch(load_dataset('json', data_files=DATASET_JSON, split='train')
                    .select_columns(['image_file', 'text']), model)

    for std in [0.0, 0.02]:
        for agg, st in zip(model.aggregators, init):
            agg.load_state_dict(st)
        torch.manual_seed(7)
        for agg in model.aggregators:
            if std > 0:
                torch.nn.init.normal_(agg.conv_out.weight, mean=0.0, std=std)
            else:
                agg.conv_out.weight.data.zero_()

        opt = optim.AdamW(model.aggregators.parameters(), lr=1e-4, weight_decay=0.01)
        sched = get_scheduler('constant', optimizer=opt, num_warmup_steps=5)
        scaler = torch.amp.GradScaler('cuda')
        torch.manual_seed(11)
        loader = iter(data.DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4, drop_last=True))
        t0 = time.time()
        for step in range(STEPS):
            opt.zero_grad(set_to_none=True)
            for micro in range(ACCUM):
                try:
                    b = next(loader)
                except StopIteration:
                    loader = iter(data.DataLoader(ds, batch_size=BATCH, shuffle=True,
                                                  num_workers=4, drop_last=True))
                    b = next(loader)
                torch.manual_seed(1000 + step * 10 + micro)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss = train_forward(model, b['pixel_values'].to(device, torch.float32),
                                         b['input_ids'].to(device))
                scaler.scale(loss / ACCUM).backward()
            opt.step()
            sched.step()

        print(f'\nout_init_std={std} after {STEPS} steps ({time.time() - t0:.0f}s) -- '
              f'relative displacement ||d||/||init||:', flush=True)
        for gname, key in GROUPS.items():
            vals = []
            for i in IDX:
                cur = dict(model.aggregators[i].named_parameters())[key].detach().float()
                old = init[i][key].float()
                vals.append((((cur - old).pow(2).sum() / old.pow(2).sum()).sqrt()).item())
            print(f'  {gname:<22} ' + '  '.join(f'a{i}={v:.5f}' for i, v in zip(IDX, vals)), flush=True)

    print('\nreference magnitudes: conv_in RMS 0.0228, transformer.proj_in 0.051, conv_out 0 '
          '(std=0) / 0.02 (std=0.02); 20 steps at lr 1e-4 with coherent Adam steps = ~0.002 absolute',
          flush=True)


if __name__ == '__main__':
    main()
