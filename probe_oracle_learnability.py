"""Is AFA's per-pixel ceiling learnable, or is it noise?

The trained aggregator routes per pixel and per layer. Its ceiling is the
per-element oracle: for each latent element, whichever expert's prediction is
closer to the true noise. That oracle uses the noise, so it is only reachable if
the winner pattern is *predictable from the inputs*.

Test: for the same image and timestep, draw several different noise realizations
and compare the per-element winner masks. If the masks agree between draws much
above chance (0.5), the structure is image-dependent and learnable; if they agree
at ~0.5, the oracle is pure noise and unreachable.

Read-only, ~1 minute of GPU after model load.
"""
import json
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
RES, TIMESTEP, N_IMGS, N_NOISE = 512, 500, 4, 3
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--expert_a', default=f'{ASSETS}/models/realistic_vision_v5.1/Realistic_Vision_V5.1.safetensors')
_ap.add_argument('--expert_b', default=f'{ASSETS}/models/Counterfeit-V3.0_fp16.safetensors')
_ap.add_argument('--names', default='RV,Counterfeit')
_ap.add_argument('--dataset_json', default=f'{ASSETS}/data/journeydb_data.json')
_a = _ap.parse_args()
_n = _a.names.split(',')
EXPERTS = [(_n[0], _a.expert_a), (_n[1], _a.expert_b)]
DATASET_JSON = _a.dataset_json
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
    device = torch.device('cuda')
    model = Model.load_init(model_paths=[p for _, p in EXPERTS], st_model_path=f'{ASSETS}/models/stable-diffusion-v1-5',
                            hidden_size=128, num_layers=1, num_attn_heads=8)
    model.vae.requires_grad_(False); model.text_encoders.requires_grad_(False)
    model.unets.requires_grad_(False); model.aggregators.requires_grad_(False)
    model.vae.to(device, torch.float32); model.text_encoders.to(device, torch.float16)
    model.unets.to(device, torch.float16); model.aggregators.to(device, torch.float16)
    model.eval()
    print(f'loaded {EXPERTS[0][0]} + {EXPERTS[1][0]}', flush=True)

    recs = [json.loads(l) for l in open(DATASET_JSON)][:N_IMGS]

    def predict(noisy, ts, ehs, k):
        for agg in model.aggregators:
            agg._pick_k = k
        Aggregator.forward = pick_expert_k
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            return model.unets[0](noisy, ts, ehs, additional_unets=model.unets[1:],
                                  aggregators=model.aggregators).sample.float()

    t0 = time.time()
    with torch.no_grad():
        pv = torch.stack([_tf(Image.open(r['image_file']).convert('RGB')) for r in recs]).to(device, torch.float32)
        lat = (model.vae.encode(pv.to(model.vae.dtype)).latent_dist.sample()
               * model.vae.config.scaling_factor).to(torch.float16)
        ehs = torch.stack([te(model.tokenizer([r['text'] for r in recs], padding='max_length', truncation=True,
                                              return_tensors='pt')['input_ids'].to(device))[0]
                           for te in model.text_encoders], dim=0).half()
        ts = torch.full((len(recs),), TIMESTEP, dtype=torch.long, device=device)

        masks, diffs, oracles = [], [], []
        for n in range(N_NOISE):
            noise = torch.randn(lat.shape, generator=torch.Generator(device='cpu').manual_seed(100 + n),
                                device='cpu').to(device).half()
            noisy = model.train_scheduler.add_noise(lat, noise, ts)
            preds = [predict(noisy, ts, ehs, k) for k in (0, 1)]
            e = [((p - noise.float()) ** 2) for p in preds]           # per element
            masks.append(e[0] < e[1])                                  # winner per element
            diffs.append(e[0] - e[1])
            oracles.append(torch.minimum(e[0], e[1]).mean().item())
            if n == 0:
                base = [e[0].mean().item(), e[1].mean().item(), torch.stack([e[0], e[1]]).mean().item()]
        print(f'{(time.time() - t0):.0f}s  {EXPERTS[0][0]}={base[0]:.5f} {EXPERTS[1][0]}={base[1]:.5f} '
              f'mean={base[2]:.5f} per-element-oracle={sum(oracles) / len(oracles):.5f}', flush=True)

        print('\nper-element winner-mask agreement between different noise draws '
              '(0.5 = pure noise, 1.0 = fully image-determined):', flush=True)
        per_img, corrs = [], []
        for i in range(len(recs)):
            ags, cs = [], []
            for a in range(N_NOISE):
                for b in range(a + 1, N_NOISE):
                    ags.append((masks[a][i] == masks[b][i]).float().mean().item())
                    x = diffs[a][i].flatten().double()
                    y = diffs[b][i].flatten().double()
                    x = x - x.mean(); y = y - y.mean()
                    cs.append(((x @ y) / (x.norm() * y.norm() + 1e-12)).item())
            per_img.append(sum(ags) / len(ags)); corrs.append(sum(cs) / len(cs))
            print(f'  image {i}: agreement={per_img[-1]:.3f}  corr(error-diff)={corrs[-1]:+.3f}  '
                  f'({EXPERTS[0][0]} wins {(masks[0][i].float().mean().item() * 100):.0f}% of elements)', flush=True)
        print(f'\n  MEAN agreement = {sum(per_img) / len(per_img):.3f}   '
              f'MEAN corr = {sum(corrs) / len(corrs):+.3f}', flush=True)


if __name__ == '__main__':
    main()
