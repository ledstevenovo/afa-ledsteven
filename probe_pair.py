"""Headroom measurement for an arbitrary expert pair / dataset.

For identical latents, noise and timestep at 512px, batch 2, compares
  mean      - average of the two experts' features at every aggregation point
              (= what the aggregator does at initialisation)
  expertA   - use expert A's features at every aggregation point
  expertB   - likewise for expert B
and reports ceilings that bound what an AFA aggregator could ever reach:
  oracle-per-sample      - best expert for each whole image
  oracle-per-element     - per latent element min of the two squared errors
  optimal-mix-per-element- per-element optimal convex weight between the two
All numbers are averaged over the same (sample, timestep) rows, so they are
directly comparable. Read-only, writes nothing.

Usage:
  python3 probe_pair.py --expert_a A.safetensors --expert_b B.safetensors \
      --names "A,B" --dataset_json .../mixed_coco_anime.json --num_pairs 8 --tag t
"""
import argparse
import json
import os
import sys
import time

import PIL.Image as Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

sys.path.insert(0, '/root/bayes-tmp/afa')
from models import Model  # noqa: E402
from models.modules.aggregator import Aggregator  # noqa: E402

torch.backends.cudnn.enabled = False
ASSETS = '/root/bayes-tmp/afa-assets'
RES, BATCH, TIMESTEPS = 512, 2, [300, 500, 700, 900]

_tf = transforms.Compose([
    transforms.Resize(RES, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(RES),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

_orig = Aggregator.forward


def pick_expert_k(self, features, temb, encoder_hidden_states):
    k = int(self._pick_k)
    attn = torch.zeros(features.shape[0], features.shape[1], *features.shape[-2:], device=features.device)
    return features[:, k], attn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--expert_a', required=True)
    ap.add_argument('--expert_b', required=True)
    ap.add_argument('--names', default='A,B')
    ap.add_argument('--dataset_json', default=f'{ASSETS}/data/mixed_coco_anime.json')
    ap.add_argument('--num_pairs', type=int, default=8)
    ap.add_argument('--st_model', default=f'{ASSETS}/models/stable-diffusion-v1-5')
    ap.add_argument('--tag', default='pair')
    args = ap.parse_args()
    name_a, name_b = args.names.split(',')

    device = torch.device('cuda')
    t0 = time.time()
    model = Model.load_init(model_paths=[args.expert_a, args.expert_b], st_model_path=args.st_model,
                            hidden_size=128, num_layers=1, num_attn_heads=8)
    model.vae.requires_grad_(False); model.text_encoders.requires_grad_(False)
    model.unets.requires_grad_(False); model.aggregators.requires_grad_(False)
    model.vae.to(device, torch.float32)
    model.text_encoders.to(device, torch.float16)
    model.unets.to(device, torch.float16)
    model.aggregators.to(device, torch.float32)
    model.eval()
    print(f'[{args.tag}] loaded in {time.time() - t0:.0f}s: {name_a} + {name_b} | '
          f'unet in/cross = {model.unets[0].config.in_channels}/{model.unets[0].config.cross_attention_dim} , '
          f'{model.unets[1].config.in_channels}/{model.unets[1].config.cross_attention_dim}', flush=True)

    records = []
    with open(args.dataset_json) as f:
        for line in f:
            records.append(json.loads(line))
    photo = [r for r in records if 'anime_images' not in r['image_file']]
    anime = [r for r in records if 'anime_images' in r['image_file']]
    print(f'[{args.tag}] dataset {len(records)} records ({len(photo)} photo, {len(anime)} anime)', flush=True)

    n_ph = args.num_pairs // 2 if anime else args.num_pairs
    n_an = (args.num_pairs - n_ph) if anime else 0
    eval_recs = photo[600:600 + n_ph * BATCH] + (anime[:n_an * BATCH] if n_an else [])
    groups = ['photo'] * (n_ph * BATCH) + ['anime'] * (n_an * BATCH)
    n_pairs = len(eval_recs) // BATCH
    print(f'[{args.tag}] evaluating {n_pairs} pairs ({n_ph} photo, {n_an} anime)', flush=True)

    variants = ['mean', 'expertA', 'expertB']

    def set_variant(v):
        Aggregator.forward = _orig
        if v == 'mean':
            for agg in model.aggregators:
                agg.conv_out.weight.data.zero_()
        else:
            for agg in model.aggregators:
                agg._pick_k = 0 if v == 'expertA' else 1
            Aggregator.forward = pick_expert_k

    rows = []
    with torch.no_grad():
        for p in range(n_pairs):
            r = eval_recs[p * BATCH:(p + 1) * BATCH]
            pv = torch.stack([_tf(Image.open(x['image_file']).convert('RGB')) for x in r]).to(device, torch.float32)
            latents = (model.vae.encode(pv.to(model.vae.dtype)).latent_dist.sample()
                       * model.vae.config.scaling_factor).to(torch.float16)
            ehs = torch.stack([te(model.tokenizer([x['text'] for x in r], padding='max_length',
                                                  truncation=True,
                                                  return_tensors='pt')['input_ids'].to(device))[0]
                               for te in model.text_encoders], dim=0).half()
            for t in TIMESTEPS:
                noise = torch.randn(latents.shape, generator=torch.Generator(device='cpu').manual_seed(t),
                                    device='cpu').to(device).half()
                ts = torch.full((latents.shape[0],), t, dtype=torch.long, device=device)
                noisy = model.train_scheduler.add_noise(latents, noise, ts)
                per, preds = {}, {}
                for v in variants:
                    set_variant(v)
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        pred = model.unets[0](noisy, ts, ehs, additional_unets=model.unets[1:],
                                              aggregators=model.aggregators).sample
                    preds[v] = pred.float()
                    per[v] = ((pred.float() - noise.float()) ** 2).mean(dim=[1, 2, 3]).cpu()

                a, b, n = preds['expertA'], preds['expertB'], noise.float()
                pix_oracle = torch.minimum((a - n) ** 2, (b - n) ** 2).mean().item()
                d, rr = (b - a), (n - a)
                w = ((d * rr).sum(dim=[1, 2, 3], keepdim=True)
                     / (d * d).sum(dim=[1, 2, 3], keepdim=True).clamp_min(1e-12)).clamp(0, 1)
                pix_mix = ((w * a + (1 - w) * b - n) ** 2).mean().item()

                for i in range(len(r)):
                    rows.append(dict(group=groups[p * BATCH + i], t=t,
                                     mean=per['mean'][i].item(),
                                     expertA=per['expertA'][i].item(),
                                     expertB=per['expertB'][i].item(),
                                     pix_oracle=pix_oracle, pix_mix=pix_mix))
        print(f'[{args.tag}] {len(rows)} rows measured in {time.time() - t0:.0f}s', flush=True)

        def report(subset, label):
            if not subset:
                return
            m = {v: sum(r[v] for r in subset) / len(subset) for v in variants}
            oracle = sum(min(r['expertA'], r['expertB']) for r in subset) / len(subset)
            pix_o = sum(r['pix_oracle'] for r in subset) / len(subset)
            pix_m = sum(r['pix_mix'] for r in subset) / len(subset)
            best = min(m['expertA'], m['expertB'])
            print(f'  {label:>6} n={len(subset):3d} | {name_a}={m["expertA"]:.5f} {name_b}={m["expertB"]:.5f} '
                  f'mean={m["mean"]:.5f} | best-single={best:.5f} | oracle-per-sample={oracle:.5f} '
                  f'({(best - oracle) / best * 100:+.2f}% vs best-single) | '
                  f'oracle-per-element={pix_o:.5f} ({(m["mean"] - pix_o) / m["mean"] * 100:+.2f}% vs mean, '
                  f'{(best - pix_o) / best * 100:+.2f}% vs best-single) | '
                  f'mix-per-element={pix_m:.5f} ({(m["mean"] - pix_m) / m["mean"] * 100:+.2f}% vs mean) | '
                  f'A-win={sum(1 for r in subset if r["expertA"] < r["expertB"])}/{len(subset)}', flush=True)

        print(f'\n[{args.tag}] === headroom (positive % = ceiling is better than the baseline) ===', flush=True)
        report(rows, 'all')
        report([r for r in rows if r['group'] == 'photo'], 'photo')
        report([r for r in rows if r['group'] == 'anime'], 'anime')


if __name__ == '__main__':
    main()
