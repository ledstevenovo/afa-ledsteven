"""Paper COCO protocol @256: CLIP-T, CLIP-I, FID (IS optional, needs torch-fidelity).

The paper reports FID / IS / CLIP-I / CLIP-T on COCO 2017 at 256x256 with 4 generated
images per method. Sampling all 118,287 pairs is not feasible on one GPU, so this script
uses the 5,000 validation pairs it also mentions, at 1 image per caption by default
(the sample count is printed with the results so the reduced scale stays explicit).

The real val2017 images are ALSO the reference set for FID and the partner image for
CLIP-I (both used at the same 256px as the generated images).

  python3 eval_coco_protocol.py --ckpt OUT/paper_run_group1_10k --out OUT/eval_coco5k
  python3 eval_coco_protocol.py --score_only OUT/eval_coco5k      # rescore, no GPU generation
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_paper_metrics import (ASSETS, CLIP_DIR, gen_afa, gen_expert, log, name_for)  # noqa: E402

COCO_DIR = f'{ASSETS}/data/coco'
PAPER_GROUP1 = f'{ASSETS}/models/_new_experts/epicrealism.safetensors,' \
               f'{ASSETS}/models/_new_experts/majicmix_v6.safetensors,' \
               f'{ASSETS}/models/realistic_vision_v5.1/Realistic_Vision_V5.1.safetensors'


def build_pairs(coco_dir, n, res, ref_dir):
    """[(caption, real_image_path_256)] from whichever COCO layout is on disk.

    Two layouts are supported:
      * official 2017 release: <coco_dir>/val2017/*.jpg + annotations/captions_val2017.json
        (one caption per image; the paper's "5,000 validation image-caption pairs")
      * the nlphuji 5k retrieval set: <coco_dir>/images_mscoco_2014_5k_test/*.jpg +
        captions.csv, whose `raw` column is a python-literal list of captions.
    """
    import ast
    ann = os.path.join(coco_dir, 'annotations/captions_val2017.json')
    img_dir = os.path.join(coco_dir, 'val2017')
    pairs_raw = []                                    # [(caption, source_image_path)]

    if os.path.exists(ann):
        with open(ann) as fh:
            data = json.load(fh)
        first = {}
        for a in data['annotations']:
            first.setdefault(a['image_id'], a['caption'])
        for im in data['images']:
            cap = first.get(im['id'])
            src = os.path.join(img_dir, im['file_name'])
            if cap and os.path.exists(src):
                pairs_raw.append((cap, src))
    else:
        csv_path = os.path.join(coco_dir, 'captions.csv')
        img_dir = os.path.join(coco_dir, 'images_mscoco_2014_5k_test')
        if not os.path.exists(csv_path):
            raise SystemExit(f'no COCO captions found under {coco_dir}')
        with open(csv_path) as fh:
            for row in csv.DictReader(fh):
                src = os.path.join(img_dir, row['filename'])
                if not os.path.exists(src):
                    continue
                caps = ast.literal_eval(row['raw'])
                pairs_raw.append((caps[0].strip(), src))

    os.makedirs(ref_dir, exist_ok=True)
    pairs = []
    for cap, src in sorted(pairs_raw, key=lambda t: t[1]):
        dst = os.path.join(ref_dir, os.path.splitext(os.path.basename(src))[0] + '.png')
        if not os.path.exists(dst):
            im = Image.open(src).convert('RGB')
            w, h = im.size
            s = min(w, h)
            im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize(
                (res, res), Image.BICUBIC)
            im.save(dst)
        pairs.append((cap, dst))
        if n and len(pairs) >= n:
            break
    return pairs


def _clip_image_features(paths, model, proc, device, batch=64):
    """All image embeddings in batches -- the per-image loop was 10x slower for no gain."""
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            imgs = [Image.open(p).convert('RGB') for p in paths[i:i + batch]]
            inp = proc(images=imgs, return_tensors='pt').to(device)
            f = model.get_image_features(**inp)
            feats.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(feats, axis=0)


def _clip_model(device):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_DIR).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_DIR)
    return model, proc


def score_clip_i(root, variants, pairs, args, device):
    """CLIP-I: cosine between each generated image and its real COCO partner."""
    model, proc = _clip_model(device)
    ref = _clip_image_features([p for _, p in pairs], model, proc, device)
    scores = {}
    for v in variants:
        vdir = os.path.join(root, v)
        files = [os.path.join(vdir, name_for(i, 0)) for i in range(len(pairs))]
        if not all(os.path.exists(f) for f in files):
            log(f'  CLIP-I {v}: incomplete image set, skipping')
            continue
        feat = _clip_image_features(files, model, proc, device)
        for i in range(len(pairs)):
            scores[(v, i)] = float(feat[i] @ ref[i])
        log(f'  CLIP-I {v}: {len(files)} pairs')
    del model
    torch.cuda.empty_cache()
    return scores


def score_clip_t(root, variants, pairs, args, device):
    model, proc = _clip_model(device)
    with torch.no_grad():
        txt = proc(text=[c for c, _ in pairs], return_tensors='pt', padding=True,
                   truncation=True).to(device)
        t = model.get_text_features(**txt)
        t = (t / t.norm(dim=-1, keepdim=True)).cpu().numpy()
    scores = {}
    for v in variants:
        vdir = os.path.join(root, v)
        files = [os.path.join(vdir, name_for(i, 0)) for i in range(len(pairs))]
        if not all(os.path.exists(f) for f in files):
            log(f'  CLIP-T {v}: incomplete image set, skipping')
            continue
        feat = _clip_image_features(files, model, proc, device)
        for i in range(len(pairs)):
            scores[(v, i)] = float(feat[i] @ t[i]) * 100
        log(f'  CLIP-T {v}: {len(files)} pairs')
    del model
    torch.cuda.empty_cache()
    return scores


def _inception_features(paths, device, batch=64):
    """2048-dim pool3 features from torchvision's InceptionV3.

    The canonical FID weights live in a GitHub release that is unreachable from this box,
    so this uses torchvision's ImageNet InceptionV3 instead. Every variant and the
    reference set go through the SAME weights, so the comparison between rows is valid,
    but the absolute values are NOT comparable to published FID numbers.
    """
    import torchvision
    from torchvision.models import Inception_V3_Weights
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = torchvision.models.inception_v3(weights=weights, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            chunk = paths[i:i + batch]
            imgs = []
            for f in chunk:
                im = Image.open(f).convert('RGB').resize((299, 299), Image.BICUBIC)
                imgs.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float())
            x = torch.stack(imgs).to(device)
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
            x = 2 * (x / 255.0) - 1
            feats.append(model(x).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(feats, axis=0)


def score_fid(root, variants, ref_dir, device):
    """Frechet distance between each variant's 256px images and the real val2017 images."""
    from scipy.linalg import sqrtm
    ref_files = sorted(glob.glob(os.path.join(ref_dir, '*.png')))
    if not ref_files:
        log('  FID: no reference images')
        return {}
    try:
        ref = _inception_features(ref_files, device)
    except Exception as exc:
        log(f'  FID: reference features failed ({type(exc).__name__}: {exc})')
        return {}
    mu_r, cov_r = ref.mean(0), np.cov(ref, rowvar=False)
    out = {}
    for v in variants:
        vdir = os.path.join(root, v)
        files = sorted(glob.glob(os.path.join(vdir, '*.png'))) if os.path.isdir(vdir) else []
        if not files:
            continue
        try:
            feat = _inception_features(files, device)
            mu, cov = feat.mean(0), np.cov(feat, rowvar=False)
            diff = mu - mu_r
            covmean = sqrtm(cov.astype(np.float64) @ cov_r.astype(np.float64))
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            fid = float(diff @ diff + np.trace(cov + cov_r - 2 * covmean))
            out[v] = fid
            log(f'  FID {v}: {fid:.2f} (n={len(files)} vs {len(ref_files)})')
        except Exception as exc:
            log(f'  FID {v} failed ({type(exc).__name__}: {exc})')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{ASSETS}/output/eval_coco5k')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--model_files', default=PAPER_GROUP1)
    ap.add_argument('--st_model_file', default=f'{ASSETS}/models/stable-diffusion-v1-5')
    ap.add_argument('--coco_dir', default=COCO_DIR)
    ap.add_argument('--n_captions', type=int, default=0, help='0 = all val2017 (5000)')
    ap.add_argument('--images_per_prompt', type=int, default=1)
    ap.add_argument('--batch_prompts', type=int, default=8)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--guidance', type=float, default=7.5)
    ap.add_argument('--res', type=int, default=256)
    ap.add_argument('--seed_base', type=int, default=5000)
    ap.add_argument('--variants', default='')
    ap.add_argument('--metrics', default='clip_t,clip_i,fid')
    ap.add_argument('--score_only', default=None)
    ap.add_argument('--no_generate', action='store_true')
    ap.add_argument('--aggregator_hidden_size', type=int, default=None)
    ap.add_argument('--aggregator_num_layers', type=int, default=None)
    ap.add_argument('--aggregator_num_attn_heads', type=int, default=None)
    ap.add_argument('--cudnn', choices=['on', 'off'], default='on')
    args = ap.parse_args()
    torch.backends.cudnn.enabled = (args.cudnn == 'on')
    device = torch.device('cuda')
    if args.score_only:
        args.out = args.score_only

    ref_dir = os.path.join(args.out, 'reference_256')
    pairs = build_pairs(args.coco_dir, args.n_captions, args.res, ref_dir)
    args._prompts = [c for c, _ in pairs]
    prompts = args._prompts
    log(f'{len(pairs)} COCO val2017 pairs at {args.res}px | reference: {ref_dir}')

    experts = [p for p in args.model_files.split(',') if p]
    from eval_paper_metrics import expert_tag
    tags = [expert_tag(p) for p in experts]
    expert0_variant = f'{tags[0]}_in_afa'
    all_variants = [f'only_{t}' for t in tags] + ['afa', expert0_variant]
    variants = all_variants if not args.variants else [v for v in args.variants.split(',')
                                                       if v in all_variants]

    if args.ckpt:
        cfg_path = os.path.join(args.ckpt, 'aggregator_0', 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as fh:
                cfg = json.load(fh)
            for key, argname in (('hidden_size', 'aggregator_hidden_size'),
                                 ('num_layers', 'aggregator_num_layers'),
                                 ('num_attn_heads', 'aggregator_num_attn_heads')):
                if cfg.get(key) is not None:
                    setattr(args, argname, cfg[key])
            log(f'aggregator config from checkpoint: hidden={args.aggregator_hidden_size} '
                f'layers={args.aggregator_num_layers} heads={args.aggregator_num_attn_heads}')

    if not args.no_generate and not args.score_only:
        for tag, path in zip(tags, experts):
            v = f'only_{tag}'
            if v in variants:
                log(f'generating {v}')
                gen_expert(path, prompts, args, device, os.path.join(args.out, v))
        if ('afa' in variants or expert0_variant in variants) and args.ckpt:
            for v in [x for x in ('afa', expert0_variant) if x in variants]:
                log(f'generating {v}')
                gen_afa(experts, tags, args, device, args.out, v)

    results = {}
    metrics = [m for m in args.metrics.split(',') if m]
    if 'clip_t' in metrics:
        results['clip_t'] = score_clip_t(args.out, variants, pairs, args, device)
    if 'clip_i' in metrics:
        results['clip_i'] = score_clip_i(args.out, variants, pairs, args, device)
    if 'fid' in metrics:
        fid = score_fid(args.out, variants, ref_dir, device)
        results['fid'] = fid
        print(f'\n=== FID vs COCO val2017 @{args.res}px ({len(pairs)} real images) ===')
        for v, x in sorted(fid.items(), key=lambda kv: kv[1]):
            print(f'  {v:<28} {x:.2f}')

    out_json = os.path.join(args.out, 'coco_metrics.json')
    summary = {}
    if os.path.exists(out_json):          # merge, and never lose earlier metrics
        with open(out_json) as fh:
            summary = json.load(fh)
    for name, sc in results.items():
        if not sc:
            continue
        print(f'\n=== {name} (mean over {len(pairs)} pairs) ===')
        means = {}
        for v in variants:
            vals = [x for (vv, _), x in sc.items() if vv == v]
            if vals:
                means[v] = float(np.mean(vals))
        for v, m in sorted(means.items(), key=lambda kv: -kv[1]):
            print(f'  {v:<28} {m:.4f}')
        summary[name] = means
        json.dump(summary, open(out_json, 'w'), indent=1)     # write as we go
    if results.get('fid'):
        summary['fid'] = results['fid']
        json.dump(summary, open(out_json, 'w'), indent=1)
    log(f'wrote {os.path.join(args.out, "coco_metrics.json")} (sample size {len(pairs)})')


if __name__ == '__main__':
    main()
