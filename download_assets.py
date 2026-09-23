"""Download the AFA reproduction assets (experts + CLIP + JourneyDB) from the HF mirror.

Layout written under $AFA_ASSETS (default /data/afa-assets):

  models/stable-diffusion-v1-5/                 diffusers dir (VAE + text tower + tokenizer)
  models/realistic_vision_v5.1/Realistic_Vision_V5.1.safetensors
  models/_new_experts/epicrealism.safetensors
  models/_new_experts/majicmix_v6.safetensors
  models/clip-vit-large-patch14/                CLIPScore / eval only
  data/journeydb_raw/000/*.jpg + *.json
  data/journeydb_data.json                      image_file + text records

The paper's Group I (epiCRealism + majicMIX realistic + Realistic Vision) is downloaded by
default; --group2 adds the Group II experts that paper cites, --all also grabs SD1.5 base.

huggingface.co is unreachable from this box; hf-mirror.com serves the same repos.

  python3 download_assets.py --shards 5 --workers 6
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

from huggingface_hub import hf_hub_download  # noqa: E402

ASSETS = os.environ.get('AFA_ASSETS', '/data/afa-assets')
SD15_REPO = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
CLIP_REPO = 'openai/clip-vit-large-patch14'

# Group I of the paper (Sec. 4.1): epiCRealism, majicMIX realistic, Realistic Vision.
GROUP1 = [
    ('SG161222/Realistic_Vision_V5.1_noVAE', 'Realistic_Vision_V5.1.safetensors',
     'realistic_vision_v5.1/Realistic_Vision_V5.1.safetensors'),
    ('Kalashnikov/epiCRealism', 'epicrealism_naturalSinRC1VAE.safetensors',
     '_new_experts/epicrealism.safetensors'),
    ('digiplay/majicMIX_realistic_v6', 'majicmixRealistic_v6.safetensors',
     '_new_experts/majicmix_v6.safetensors'),
]
# Group II of the paper: AbsoluteReality, CyberRealistic, RealCartoon-Realistic.
GROUP2 = [
    ('Lykon/AbsoluteReality', 'AbsoluteReality_1.8.1_pruned.safetensors',
     '_new_experts/absolute_reality.safetensors'),
    ('cyberdelia/CyberRealistic', 'CyberRealistic_V3.3_FP16.safetensors',
     '_new_experts/cyberrealistic.safetensors'),
    ('Maik2801/realcartoonRealistic_v8', 'realcartoonRealistic_v8.safetensors',
     '_new_experts/realcartoon_realistic.safetensors'),
]

# Only the weights diffusers actually loads (from_pretrained picks the non-fp16
# safetensors). Pattern-based download ("unet/*") also pulls the .bin, .fp16.bin and
# .non_ema.* variants -- ~11 GB of weights nothing here will ever read.
CLIP_FILES = ['config.json', 'preprocessor_config.json', 'model.safetensors',
              'merges.txt', 'vocab.json', 'special_tokens_map.json', 'tokenizer_config.json']

SD15_FILES = [
    'model_index.json',
    'scheduler/scheduler_config.json',
    'feature_extractor/preprocessor_config.json',
    'tokenizer/merges.txt', 'tokenizer/special_tokens_map.json',
    'tokenizer/tokenizer_config.json', 'tokenizer/vocab.json',
    'text_encoder/config.json', 'text_encoder/model.safetensors',
    'unet/config.json', 'unet/diffusion_pytorch_model.safetensors',
    'vae/config.json', 'vae/diffusion_pytorch_model.safetensors',
    # config for from_single_file(): without it diffusers fetches this from
    # raw.githubusercontent.com, which is unreachable here
    'v1-inference.yaml',
]


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def download_models(experts, workers):
    """SD1.5, CLIP and every expert share one worker pool."""
    os.makedirs(os.path.join(ASSETS, 'models'), exist_ok=True)
    jobs = [(SD15_REPO, rel, os.path.join(ASSETS, 'models/stable-diffusion-v1-5', rel),
             f'SD1.5 {rel}') for rel in SD15_FILES]
    jobs += [(CLIP_REPO, rel, os.path.join(ASSETS, 'models/clip-vit-large-patch14', rel),
              f'CLIP {rel}') for rel in CLIP_FILES]
    jobs += [(repo, fn, os.path.join(ASSETS, dest), dest) for repo, fn, dest in experts]
    return fetch_group(jobs, workers)


def fetch_group(jobs, workers):
    """jobs: [(repo, remote_filename, dest_path, label)] -> {label: status}"""
    def one(repo, filename, dest_path):
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
            return 'cached'
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        tmp = hf_hub_download(repo, filename, local_dir=os.path.join(ASSETS, '_dl'))
        os.replace(tmp, dest_path)
        return f'{os.path.getsize(dest_path) / 2**20:.0f} MB'

    status = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(one, repo, fn, dest): label for repo, fn, dest, label in jobs}
        for fut in as_completed(futs):
            label = futs[fut]
            try:
                status[label] = fut.result()
                log(f'  {label}: {status[label]}')
            except Exception as exc:  # one unavailable mirror copy must not kill the rest
                status[label] = f'FAILED {type(exc).__name__}: {str(exc)[:100]}'
                log(f'  {label}: {status[label]}')
    return status


def untar(path, dest):
    with tarfile.open(path) as tf:
        try:
            tf.extractall(dest, filter='data')
        except TypeError:  # python < 3.12
            tf.extractall(dest)


def download_journeydb(shards, workers):
    raw = os.path.join(ASSETS, 'data/journeydb_raw')
    os.makedirs(raw, exist_ok=True)
    log('JourneyDB captions.zip (~0.5 GB) ...')
    cap = hf_hub_download('wusize/journeydb', 'captions.zip', repo_type='dataset',
                          local_dir=os.path.join(ASSETS, '_dl'))
    os.makedirs(raw, exist_ok=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for i in range(shards):
            if os.path.isdir(os.path.join(raw, f'{i:03d}')):
                log(f'  shard {i:03d} already extracted')
                continue
            futures[i] = pool.submit(
                hf_hub_download, 'wusize/journeydb', f'images/{i:03d}.tar',
                repo_type='dataset', local_dir=os.path.join(ASSETS, '_dl/images'))
        for i, fut in futures.items():
            tar_path = fut.result()
            log(f'  extracting images/{i:03d}.tar')
            untar(tar_path, raw)
            os.remove(tar_path)
            with zipfile.ZipFile(cap) as zf:
                member = f'captions/{i:03d}.tar'
                zf.extract(member, os.path.join(ASSETS, '_dl/captions'))
            untar(os.path.join(ASSETS, '_dl/captions', member), raw)
            os.remove(os.path.join(ASSETS, '_dl/captions', member))
            log(f'  shard {i:03d} done')

    out = os.path.join(ASSETS, 'data/journeydb_data.json')
    n = 0
    with open(out, 'w') as w:
        for img in sorted(glob.glob(os.path.join(raw, '**', '*.jpg'), recursive=True)):
            meta = img[:-4] + '.json'
            if not os.path.exists(meta):
                continue
            caption = json.load(open(meta)).get('caption', '').strip()
            if caption:
                w.write(json.dumps({'image_file': img, 'text': caption}) + '\n')
                n += 1
    log(f'{out}: {n} records')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', type=int, default=5, help='JourneyDB image shards (20.9k images each)')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--group2', action='store_true', help='also fetch the paper Group II experts')
    ap.add_argument('--no-data', action='store_true', help='models only')
    args = ap.parse_args()

    experts = list(GROUP1)
    if args.group2:
        experts += GROUP2
    os.makedirs(ASSETS, exist_ok=True)
    log(f'assets -> {ASSETS} (HF_ENDPOINT={os.environ["HF_ENDPOINT"]})')
    download_models(experts, args.workers)
    if not args.no_data:
        download_journeydb(args.shards, args.workers)
    shutil.rmtree(os.path.join(ASSETS, '_dl'), ignore_errors=True)
    log('ALL DONE')


if __name__ == '__main__':
    sys.exit(main())
