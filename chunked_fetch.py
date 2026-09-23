"""Chunked, resumable, sha256-verified downloader for the AFA reproduction assets.

hf_hub_download opens one connection per file and tops out around 2 MB/s here; this
box's egress does 5-7 MB/s when a few files are split into byte ranges (measured).
Everything lands under $AFA_ASSETS in the layout the training/eval scripts expect.

  python3 chunked_fetch.py --assets /root/autodl-tmp/afa-assets            # models + data
  python3 chunked_fetch.py --assets ... --only experts,clip                # a subset
  python3 chunked_fetch.py --assets ... --extract                          # unpack JourneyDB
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

MIRROR = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
CHUNK = 96 * 1024 * 1024

SD15_REPO = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
CLIP_REPO = 'openai/clip-vit-large-patch14'
JD_REPO = 'wusize/journeydb'

# Only the weights diffusers actually loads: pattern downloads ("unet/*") also pull the
# .bin / .fp16.bin / .non_ema.* variants -- ~11 GB nothing here reads.
SD15_FILES = [
    'model_index.json', 'scheduler/scheduler_config.json',
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
CLIP_FILES = ['config.json', 'preprocessor_config.json', 'model.safetensors',
              'merges.txt', 'vocab.json', 'special_tokens_map.json', 'tokenizer_config.json']
EXPERTS = [
    ('SG161222/Realistic_Vision_V5.1_noVAE', 'Realistic_Vision_V5.1.safetensors',
     'models/realistic_vision_v5.1/Realistic_Vision_V5.1.safetensors'),
    ('Kalashnikov/epiCRealism', 'epicrealism_naturalSinRC1VAE.safetensors',
     'models/_new_experts/epicrealism.safetensors'),
    ('digiplay/majicMIX_realistic_v6', 'majicmixRealistic_v6.safetensors',
     'models/_new_experts/majicmix_v6.safetensors'),
]


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def head(url):
    r = subprocess.run(['curl', '-sIL', '--max-time', '60', url], capture_output=True, text=True)
    size = sha = None
    for line in r.stdout.splitlines():
        m = re.match(r'content-length:\s*(\d+)', line, re.I)
        if m:
            size = int(m.group(1))
        m = re.match(r'x-linked-etag:\s*"?([0-9a-f]{64})"?', line, re.I)
        if m:
            sha = m.group(1)
    return size, sha


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 22), b''):
            h.update(blk)
    return h.hexdigest()


def chunk_worker(url, start, end, path, tries=60):
    """Fetch [start, end] into `path`, resuming by re-requesting only what is missing.

    Do NOT combine curl -C - with -r: curl then re-requests from an offset inside a
    fixed range and appends the whole body again, so the part file grows past the file
    size (observed: a 3.97 GB file with a 9.5 GB part). Each attempt here writes to a
    scratch file, is length-checked, and only then appended -- a server that ignores
    the range (200 instead of 206) can then never corrupt the result.
    """
    expect = end - start + 1
    tmp = path + '.tmp'
    for _ in range(tries):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have == expect:
            return True
        if have > expect:
            os.remove(path)
            have = 0
        subprocess.run(['curl', '-sSL', '--max-time', '300', '-r', f'{start + have}-{end}',
                        '-o', tmp, url], capture_output=True)
        got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if got == end - (start + have) + 1:
            with open(path, 'ab') as out, open(tmp, 'rb') as src:
                for blk in iter(lambda: src.read(1 << 22), b''):
                    out.write(blk)
        if os.path.exists(tmp):
            os.remove(tmp)
    return os.path.exists(path) and os.path.getsize(path) == expect


def fetch_file(url, dest, chunks, pool):
    """Download one file as `chunks` parallel byte ranges, then concatenate."""
    try:
        size, want_sha = head(url)
    except Exception as exc:
        return dest, f'FAILED head: {exc}'
    if size is None:
        return dest, 'FAILED: no content-length'
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return dest, 'cached'
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    step = max(CHUNK, -(-size // chunks))
    spans = [(s, min(s + step - 1, size - 1)) for s in range(0, size, step)]
    tmpdir = dest + '.parts'
    os.makedirs(tmpdir, exist_ok=True)
    t0 = time.time()
    futs = [pool.submit(chunk_worker, url, s, e, f'{tmpdir}/{i:03d}')
            for i, (s, e) in enumerate(spans)]
    if not all(f.result() for f in futs):
        done = sum(os.path.getsize(f'{tmpdir}/{i:03d}') for i in range(len(spans))
                   if os.path.exists(f'{tmpdir}/{i:03d}'))
        return dest, f'RETRY: {done / 2**20:.0f}/{size / 2**20:.0f} MB downloaded'
    with open(dest, 'wb') as out:
        for i in range(len(spans)):
            with open(f'{tmpdir}/{i:03d}', 'rb') as part:
                for blk in iter(lambda: part.read(1 << 22), b''):
                    out.write(blk)
            os.remove(f'{tmpdir}/{i:03d}')
    os.rmdir(tmpdir)
    if want_sha and sha256(dest) != want_sha:
        os.remove(dest)
        return dest, 'FAILED: sha256 mismatch'
    dt = max(time.time() - t0, 1e-6)
    return dest, f'{size / 2**20:.0f} MB in {dt:.0f}s ({size / 2**20 / dt:.2f} MB/s)'


def job_list(assets, shards):
    jobs = []
    for rel in SD15_FILES:
        jobs.append((f'{MIRROR}/{SD15_REPO}/resolve/main/{rel}',
                     os.path.join(assets, 'models/stable-diffusion-v1-5', rel), 'sd15'))
    for rel in CLIP_FILES:
        jobs.append((f'{MIRROR}/{CLIP_REPO}/resolve/main/{rel}',
                     os.path.join(assets, 'models/clip-vit-large-patch14', rel), 'clip'))
    for repo, fn, dest in EXPERTS:
        jobs.append((f'{MIRROR}/{repo}/resolve/main/{fn}', os.path.join(assets, dest), 'experts'))
    jobs.append((f'{MIRROR}/datasets/{JD_REPO}/resolve/main/captions.zip',
                 os.path.join(assets, '_jd/captions.zip'), 'journeydb'))
    for i in range(shards):
        jobs.append((f'{MIRROR}/datasets/{JD_REPO}/resolve/main/images/{i:03d}.tar',
                     os.path.join(assets, f'_jd/{i:03d}.tar'), 'journeydb'))
    return jobs


def extract_journeydb(assets, shards):
    raw = os.path.join(assets, 'data/journeydb_raw')
    os.makedirs(raw, exist_ok=True)
    cap = os.path.join(assets, '_jd/captions.zip')
    for i in range(shards):
        if os.path.isdir(os.path.join(raw, f'{i:03d}')):
            log(f'  shard {i:03d} already extracted')
            continue
        with zipfile.ZipFile(cap) as zf:
            zf.extract(f'captions/{i:03d}.tar', os.path.join(assets, '_jd'))
        for tar in (os.path.join(assets, f'_jd/{i:03d}.tar'),
                    os.path.join(assets, '_jd', 'captions', f'{i:03d}.tar')):
            with tarfile.open(tar) as tf:
                try:
                    tf.extractall(raw, filter='data')
                except TypeError:
                    tf.extractall(raw)
            os.remove(tar)
        log(f'  shard {i:03d} extracted')

    records = 0
    out = os.path.join(assets, 'data/journeydb_data.json')
    with open(out, 'w') as w:
        for root, _, files in os.walk(raw):
            for name in sorted(files):
                if not name.endswith('.jpg'):
                    continue
                img = os.path.join(root, name)
                meta = img[:-4] + '.json'
                if not os.path.exists(meta):
                    continue
                with open(meta) as fh:
                    caption = json.load(fh).get('caption', '').strip()
                if caption:
                    w.write(json.dumps({'image_file': img, 'text': caption}) + '\n')
                    records += 1
    log(f'  {out}: {records} records')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assets', default=os.environ.get('AFA_ASSETS', '/data/afa-assets'))
    ap.add_argument('--chunks', type=int, default=4, help='ranges per file')
    ap.add_argument('--files', type=int, default=3, help='files in flight at once')
    ap.add_argument('--shards', type=int, default=5)
    ap.add_argument('--only', default='', help='comma list of sd15,clip,experts,journeydb')
    ap.add_argument('--extract', action='store_true', help='unpack JourneyDB and build the jsonl')
    args = ap.parse_args()

    jobs = job_list(args.assets, args.shards)
    if args.only:
        keep = set(args.only.split(','))
        jobs = [j for j in jobs if j[2] in keep]
    if not args.extract:
        log(f'{len(jobs)} files -> {args.assets} | {args.files} files x {args.chunks} chunks')
        with ThreadPoolExecutor(max_workers=args.files * args.chunks) as pool:
            for i in range(0, len(jobs), args.files):
                group = jobs[i:i + args.files]
                futs = [pool.submit(fetch_file, url, dest, args.chunks, pool)
                        for url, dest, _ in group]
                for fut in futs:
                    dest, status = fut.result()
                    log(f'  {os.path.relpath(dest, args.assets)}: {status}')
    else:
        extract_journeydb(args.assets, args.shards)
    log('done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
