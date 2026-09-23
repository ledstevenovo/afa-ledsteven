"""Capture one expert's noise predictions on a fixed sample set.

Every expert sees the *same* latents (from the shared st_model VAE), the same
noise realizations and the same timesteps, so predictions are directly paired:
per-element squared errors and winner masks can be compared across experts.

  python3 probe_multi_expert.py --expert PATH --name NAME

Saves <out>/<name>.pt = {meta, combos: [{img, t, seed, noise, pred}]} in fp16.
Run analyze_multi_expert.py afterwards to combine all experts.
"""
import argparse
import json
import os
import sys
import time

import PIL.Image as Image
import torch
import torchvision.transforms as transforms

sys.path.insert(0, '/root/bayes-tmp/afa')
from models import Model  # noqa: E402
from models.modules.aggregator import Aggregator  # noqa: E402

torch.backends.cudnn.enabled = False
ASSETS = '/root/bayes-tmp/afa-assets'
DATASET = f'{ASSETS}/data/journeydb_data.json'
OUT = f'{ASSETS}/output/expert_preds'
RES, N_IMGS, SEEDS, TIMESTEPS, BATCH = 512, 8, [100, 101, 102, 103], [500, 800], 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--expert', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--dataset_json', default=DATASET)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, f'{args.name}.pt')
    if os.path.exists(dest):
        print(f'{dest} exists, skip', flush=True)
        return

    device = torch.device('cuda')
    t0 = time.time()
    model = Model.load_init(model_paths=[args.expert], st_model_path=f'{ASSETS}/models/stable-diffusion-v1-5',
                            hidden_size=128, num_layers=1, num_attn_heads=8)
    model.vae.requires_grad_(False); model.text_encoders.requires_grad_(False)
    model.unets.requires_grad_(False); model.aggregators.requires_grad_(False)
    model.vae.to(device, torch.float32)
    model.text_encoders.to(device, torch.float16)
    model.unets.to(device, torch.float16)
    model.aggregators.to(device, torch.float16)
    model.eval()

    # With a single expert the aggregator is mathematically the identity on that
    # expert's features, so skip its (expensive) transformer entirely -- and turn
    # xformers off, which is measurably slower than PyTorch SDPA on this T4.
    def _identity(self, features, temb, encoder_hidden_states):
        return features[:, 0], torch.ones(features.shape[0], 1, *features.shape[-2:],
                                          device=features.device)
    Aggregator.forward = _identity
    for u in model.unets:
        try:
            u.disable_xformers_memory_efficient_attention()
        except Exception:
            pass
    print(f'[{args.name}] loaded in {time.time() - t0:.0f}s | unet in/cross = '
          f'{model.unets[0].config.in_channels}/{model.unets[0].config.cross_attention_dim}', flush=True)

    recs = [json.loads(l) for l in open(args.dataset_json)][:N_IMGS]
    tf = transforms.Compose([
        transforms.Resize(RES, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(RES),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    combos = []
    with torch.no_grad():
        pv = torch.stack([tf(Image.open(r['image_file']).convert('RGB')) for r in recs]).to(device, torch.float32)
        lat = (model.vae.encode(pv.to(model.vae.dtype)).latent_dist.sample()
               * model.vae.config.scaling_factor).to(torch.float16)
        ehs = torch.stack([te(model.tokenizer([r['text'] for r in recs], padding='max_length',
                                              truncation=True,
                                              return_tensors='pt')['input_ids'].to(device))[0]
                           for te in model.text_encoders], dim=0).half()
        for t in TIMESTEPS:
            ts_all = torch.full((len(recs),), t, dtype=torch.long, device=device)
            for seed in SEEDS:
                noise = torch.randn(lat.shape, generator=torch.Generator(device='cpu').manual_seed(seed),
                                    device='cpu').to(device).half()
                noisy = model.train_scheduler.add_noise(lat, noise, ts_all)
                preds = []
                for i in range(0, len(recs), BATCH):
                    sl = slice(i, i + BATCH)
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        p = model.unets[0](noisy[sl], ts_all[sl], ehs[:, sl],
                                           additional_unets=[], aggregators=model.aggregators).sample
                    preds.append(p.float().cpu().half())
                pred = torch.cat(preds, dim=0)
                for i in range(len(recs)):
                    combos.append(dict(img=i, t=t, seed=seed,
                                       noise=noise[i].float().cpu().half(),
                                       pred=pred[i]))
    torch.save(dict(name=args.name, expert=args.expert, timesteps=TIMESTEPS, seeds=SEEDS,
                    n_imgs=N_IMGS, combos=combos), dest)
    print(f'[{args.name}] {len(combos)} combos -> {dest} '
          f'({os.path.getsize(dest) / 2**20:.1f} MB) in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
