"""Paper-protocol evaluation on DrawBench: CLIP-T, LAION AES, ImageReward.

The paper scores 4 images per DrawBench prompt (200 prompts, DDIM 50 steps, CFG 7.5,
512px) with AES / PickScore / HPSv2 / ImageReward, and CLIP-T/CLIP-I/FID/IS on COCO.
This script covers the DrawBench half (the metrics that need no reference image set)
and keeps every generated image on disk so metrics can be recomputed without
regenerating (--score_only).

Variants follow --model_files exactly as training defines them, plus 'afa' (the trained
aggregator) and '<expert0>_in_afa' (expert 0 alone inside the AFA framework, which
isolates the aggregator from VAE / text-tower differences).

  # smoke test on 8 prompts (fast, tells you the per-image cost)
  python3 eval_paper_metrics.py --ckpt OUT/paper_run_group1_10k --n_prompts 8 --out OUT/eval_drawbench
  # full protocol
  python3 eval_paper_metrics.py --ckpt OUT/paper_run_group1_10k --out OUT/eval_drawbench
  # rescore what is already on disk
  python3 eval_paper_metrics.py --score_only OUT/eval_drawbench
"""
import argparse
import csv
import gc
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import Model  # noqa: E402
from models.modules.aggregator import Aggregator  # noqa: E402
from eval_generation import PAPER_GROUP1, SD15, ASSETS, expert_tag, slug  # noqa: E402

DRAWBENCH = f'{ASSETS}/data/drawbench/drawbench.csv'
CLIP_DIR = f'{ASSETS}/models/clip-vit-large-patch14'
AES_DIR = f'{ASSETS}/models/aesthetic'
IR_DIR = f'{ASSETS}/models/imagereward'
PICKSCORE_DIR = f'{ASSETS}/models/pickscore'
HPS_CKPT = f'{ASSETS}/models/hpsv2/HPS_v2.1_compressed.pt'
VITH_BIN = f'{ASSETS}/models/hpsv2/open_clip_ViT-H-14.safetensors'


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def load_prompts(path, n):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    col = 'Prompts' if 'Prompts' in rows[0] else list(rows[0])[0]
    prompts = [r[col].strip() for r in rows if r.get(col, '').strip()]
    return prompts[:n] if n else prompts


def to_scalar(x):
    """A CUDA tensor cannot go through np.asarray(); copy it to the host first."""
    if torch.is_tensor(x):
        x = x.detach().float().cpu()
    a = np.asarray(x).reshape(-1)
    return float(a[0])


def free(obj):
    del obj
    gc.collect()
    torch.cuda.empty_cache()


def name_for(i, k):
    return f'{i:03d}_{k:02d}.png'


def seed_for(args, i, k):
    """Deterministic seed per (prompt, image) so every variant sees the same latents."""
    return args.seed_base + i * 100 + k


# --------------------------------------------------------------------------- generate
def gen_expert(path, prompts, args, device, vdir):
    """One expert in its own pipeline, batched over prompts."""
    # original_config_file: without it diffusers fetches v1-inference.yaml from
    # raw.githubusercontent.com, which is unreachable here (load fails outright).
    yaml = os.path.join(args.st_model_file, 'v1-inference.yaml')
    pipe = Model.load_sd_model(path, torch_dtype=torch.float16,
                               original_config_file=yaml if os.path.exists(yaml) else None)
    os.makedirs(vdir, exist_ok=True)
    if pipe is None:
        log(f'  {path} not found, skipping')
        return
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    t0 = time.time()
    made = 0
    for b0 in range(0, len(prompts), args.batch_prompts):
        chunk = prompts[b0:b0 + args.batch_prompts]
        todo = [i for i in range(b0, b0 + len(chunk))
                if not all(os.path.exists(os.path.join(vdir, name_for(i, k)))
                           for k in range(args.images_per_prompt))]
        if not todo:
            continue
        # one generator per (prompt, image): diffusers requires len(generators) == effective batch,
        # and the per-(i, k) seeds make the latents identical across variants.
        gens = [torch.Generator(device=device).manual_seed(seed_for(args, i, k))
                for i in todo for k in range(args.images_per_prompt)]
        with torch.no_grad():
            out = pipe([prompts[i] for i in todo], height=args.res, width=args.res,
                       num_inference_steps=args.steps, guidance_scale=args.guidance,
                       generator=gens, num_images_per_prompt=args.images_per_prompt)
        for j, i in enumerate(todo):
            for k in range(args.images_per_prompt):
                out.images[j * args.images_per_prompt + k].save(os.path.join(vdir, name_for(i, k)))
                made += 1
    dt = time.time() - t0
    log(f'  {os.path.basename(vdir)}: {made} images in {dt / 60:.1f} min '
        f'({dt / max(made, 1):.1f} s/image)')
    free(pipe)


def gen_afa(experts, tags, args, device, out, which):
    """'afa' (aggregator) or '<expert0>_in_afa' (expert 0 only)."""
    model = Model.load_init(model_paths=experts, st_model_path=args.st_model_file,
                            hidden_size=args.aggregator_hidden_size,
                            num_layers=args.aggregator_num_layers,
                            num_attn_heads=args.aggregator_num_attn_heads)
    model.vae.requires_grad_(False); model.text_encoders.requires_grad_(False)
    model.unets.requires_grad_(False); model.aggregators.requires_grad_(False)
    model.vae.to(device, torch.float16); model.text_encoders.to(device, torch.float16)
    model.unets.to(device, torch.float16); model.aggregators.to(device, torch.float16)
    model.eval()
    for i, agg in enumerate(model.aggregators):
        agg.load_state_dict(Aggregator.load_pretrained(f'{args.ckpt}/aggregator_{i}').state_dict())

    _orig = Aggregator.forward

    def pick_expert0(self, features, temb, encoder_hidden_states):
        attn = torch.zeros(features.shape[0], features.shape[1], *features.shape[-2:],
                           device=features.device)
        return features[:, 0], attn

    Aggregator.forward = _orig if which == 'afa' else pick_expert0
    prompts = args._prompts
    vdir = os.path.join(out, which)
    os.makedirs(vdir, exist_ok=True)
    t0 = time.time()
    made = 0
    for b0 in range(0, len(prompts), args.batch_prompts):
        todo = [i for i in range(b0, min(b0 + args.batch_prompts, len(prompts)))
                if not all(os.path.exists(os.path.join(vdir, name_for(i, k)))
                           for k in range(args.images_per_prompt))]
        if not todo:
            continue
        gens = [torch.Generator(device=device).manual_seed(seed_for(args, i, k))
                for i in todo for k in range(args.images_per_prompt)]
        with torch.no_grad():
            out_obj = model.test_forward(prompt=[prompts[i] for i in todo], height=args.res,
                                         width=args.res, num_inference_steps=args.steps,
                                         guidance_scale=args.guidance, generator=gens,
                                         num_images_per_prompt=args.images_per_prompt)
        for j, i in enumerate(todo):
            for k in range(args.images_per_prompt):
                out_obj.images[j * args.images_per_prompt + k].save(os.path.join(vdir, name_for(i, k)))
                made += 1
    dt = time.time() - t0
    log(f'  {which}: {made} images in {dt / 60:.1f} min ({dt / max(made, 1):.1f} s/image)')
    Aggregator.forward = _orig
    free(model)


# ----------------------------------------------------------------------------- score
def score_clip_t(root, variants, prompts, args, device):
    from transformers import CLIPModel, CLIPProcessor
    clip = CLIPModel.from_pretrained(CLIP_DIR).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_DIR)
    scores = {}
    with torch.no_grad():
        txt = proc(text=prompts, return_tensors='pt', padding=True, truncation=True).to(device)
        tfeat = clip.get_text_features(**txt)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
        for v in variants:
            for i, prompt in enumerate(prompts):
                vals = []
                for k in range(args.images_per_prompt):
                    p = os.path.join(root, v, name_for(i, k))
                    if not os.path.exists(p):
                        continue
                    im = Image.open(p).convert('RGB')
                    ifeat = clip.get_image_features(**proc(images=im, return_tensors='pt').to(device))
                    ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
                    vals.append(float((ifeat @ tfeat[i:i + 1].T).item() * 100))
                if vals:
                    scores[(v, i)] = float(np.mean(vals))
    free(clip)
    log('  CLIP-T done')
    return scores


def score_aes(root, variants, args, device):
    from transformers import AutoModel, CLIPImageProcessor
    try:
        aes = AutoModel.from_pretrained(AES_DIR, trust_remote_code=True).to(device).eval()
        proc = CLIPImageProcessor.from_pretrained(AES_DIR)
    except Exception as exc:
        log(f'  AES unavailable ({type(exc).__name__}: {exc}); skipping')
        return {}
    scores = {}
    with torch.no_grad():
        for v in variants:
            vdir = os.path.join(root, v)
            if not os.path.isdir(vdir):
                continue
            for f in sorted(os.listdir(vdir)):
                if not f.endswith('.png'):
                    continue
                im = Image.open(os.path.join(vdir, f)).convert('RGB')
                px = proc(images=im, return_tensors='pt').to(device)
                out = aes(**px)
                val = out.logits if hasattr(out, 'logits') else out[0]
                scores[(v, f)] = to_scalar(val)
    free(aes)
    log('  AES done')
    return scores


def score_ir(root, variants, prompts, args, device):
    """ImageReward, scored with the prompt/image pairing actually matched.

    ImageReward.score(a_list, b_list) returns the CROSS PRODUCT of the two lists
    (4 prompts x 4 images -> 16 values), so scoring a whole variant in one call would
    average mismatched pairs. One prompt with its own N images returns exactly N
    matched values, which is what this does.
    """
    try:
        import ImageReward as RM
        model = RM.load('ImageReward-v1.0', download_root=IR_DIR, device=str(device))
    except Exception as exc:
        log(f'  ImageReward unavailable ({type(exc).__name__}: {exc}); skipping')
        return {}
    scores = {}
    for v in variants:
        vdir = os.path.join(root, v)
        if not os.path.isdir(vdir):
            continue
        n_scored = 0
        for i, prompt in enumerate(prompts):
            files = [os.path.join(vdir, name_for(i, k)) for k in range(args.images_per_prompt)]
            files = [f for f in files if os.path.exists(f)]
            if not files:
                continue
            try:
                with torch.no_grad():
                    vals = model.score(prompt, files)
            except Exception as exc:
                log(f'  IR failed on prompt {i} ({type(exc).__name__}: {exc}); stopping this metric')
                free(model)
                return scores
            vals = vals.detach().float().cpu() if torch.is_tensor(vals) else vals
            arr = np.asarray(vals).reshape(-1)
            if arr.size != len(files):
                log(f'  IR returned {arr.size} values for {len(files)} images on prompt {i}; stopping')
                free(model)
                return scores
            scores[(v, i)] = float(arr.mean())
            n_scored += len(files)
        log(f'  IR {v}: {n_scored} images')
    free(model)
    log('  ImageReward done')
    return scores


def score_pick(root, variants, prompts, args, device):
    """PickScore v1: fine-tuned CLIP-H/14; score = logit_scale * cosine(image, text)."""
    try:
        from transformers import AutoProcessor, AutoModel
        proc = AutoProcessor.from_pretrained(PICKSCORE_DIR)
        model = AutoModel.from_pretrained(PICKSCORE_DIR).to(device).eval()
    except Exception as exc:
        log(f'  PickScore unavailable ({type(exc).__name__}: {exc}); skipping')
        return {}
    scores = {}
    with torch.no_grad():
        for v in variants:
            vdir = os.path.join(root, v)
            if not os.path.isdir(vdir):
                continue
            for i, prompt in enumerate(prompts):
                files = [os.path.join(vdir, name_for(i, k)) for k in range(args.images_per_prompt)]
                files = [f for f in files if os.path.exists(f)]
                if not files:
                    continue
                imgs = [Image.open(f).convert('RGB') for f in files]
                image_inputs = proc(images=imgs, return_tensors='pt', padding=True).to(device)
                text_inputs = proc(text=[prompt] * len(imgs), return_tensors='pt',
                                   padding=True, truncation=True).to(device)
                iemb = model.get_image_features(**image_inputs)
                iemb = iemb / iemb.norm(dim=-1, keepdim=True)
                temb = model.get_text_features(**text_inputs)
                temb = temb / temb.norm(dim=-1, keepdim=True)
                s = (model.logit_scale.exp() * (iemb * temb).sum(dim=-1)).cpu().numpy()
                scores[(v, i)] = float(np.mean(s))
    free(model)
    log('  PickScore done')
    return scores


def score_hps(root, variants, prompts, args, device):
    """HPSv2 (human preference score). hpsv2 fetches its checkpoint via hf_hub_download,
    so HF_ENDPOINT / HF_HOME must point at the mirror and the cache holding it."""
    try:
        import hpsv2
        from hpsv2 import img_score as _img_score
        # hpsv2's vendored open_clip resolves the backbone by repo id, and hf_hub_download
        # times out on that repo here (the mirror redirects to HF's Xet bridge). Point it at
        # the locally fetched weights instead, when they are present.
        if os.path.exists(VITH_BIN):
            _orig_cmat = _img_score.create_model_and_transforms

            def _patched_cmat(model_name, pretrained=None, **kw):
                if pretrained == 'laion2B-s32B-b79K':
                    pretrained = VITH_BIN
                return _orig_cmat(model_name, pretrained, **kw)
            _img_score.create_model_and_transforms = _patched_cmat
            log(f'  HPSv2: backbone from {VITH_BIN}')
        else:
            log(f'  HPSv2: {VITH_BIN} missing, falling back to the hub download')
    except Exception as exc:
        log(f'  HPSv2 unavailable ({type(exc).__name__}: {exc}); skipping')
        return {}
    cp = HPS_CKPT if os.path.exists(HPS_CKPT) else None
    log(f'  HPSv2 checkpoint: {cp or "fetching from the hub"}')
    scores = {}
    for v in variants:
        vdir = os.path.join(root, v)
        if not os.path.isdir(vdir):
            continue
        for i, prompt in enumerate(prompts):
            files = [os.path.join(vdir, name_for(i, k)) for k in range(args.images_per_prompt)]
            files = [f for f in files if os.path.exists(f)]
            if not files:
                continue
            try:
                vals = hpsv2.score(files, prompt, cp=cp, hps_version=args.hps_version)
            except Exception as exc:
                log(f'  HPSv2 failed on prompt {i} ({type(exc).__name__}: {exc}); stopping this metric')
                return scores
            vals = vals.detach().float().cpu() if torch.is_tensor(vals) else vals
            scores[(v, i)] = float(np.mean(np.asarray(vals).reshape(-1)))
    log(f'  HPSv2 ({args.hps_version}) done')
    return scores


def summarize(scores, variants, metric, per_prompt, prompts, images_per_prompt):
    """Print a variant x metric table (mean over prompts)."""
    def prompt_index(key):
        """score_* store either an int prompt index or a '<prompt>_<k>.png' filename."""
        if isinstance(key, int):
            return key
        try:
            return int(str(key).split('_')[0])
        except (ValueError, IndexError):
            return None

    rows = {}
    for v in variants:
        vals = []
        for i in range(len(prompts)):
            sub = [s for (vv, key), s in scores.items()
                   if vv == v and prompt_index(key) == i]
            if sub:
                vals.append(float(np.mean(sub)))
        rows[v] = vals
    means = {v: (float(np.mean(xs)) if xs else float('nan')) for v, xs in rows.items()}
    print(f'\n=== {metric} (mean over {len(prompts)} prompts x {images_per_prompt} images) ===')
    for v, m in sorted(means.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else -1e9)):
        print(f'  {v:<28} {m:.4f}')
    return means, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{ASSETS}/output/eval_drawbench')
    ap.add_argument('--ckpt', default=None, help='aggregator checkpoint (needed for afa variants)')
    ap.add_argument('--model_files', default=PAPER_GROUP1)
    ap.add_argument('--st_model_file', default=SD15)
    ap.add_argument('--prompts_csv', default=DRAWBENCH)
    ap.add_argument('--n_prompts', type=int, default=0, help='0 = all (200)')
    ap.add_argument('--images_per_prompt', type=int, default=4)
    ap.add_argument('--batch_prompts', type=int, default=4)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--guidance', type=float, default=7.5)
    ap.add_argument('--res', type=int, default=512)
    ap.add_argument('--seed_base', type=int, default=1000)
    ap.add_argument('--aggregator_hidden_size', type=int, default=None)
    ap.add_argument('--aggregator_num_layers', type=int, default=None)
    ap.add_argument('--aggregator_num_attn_heads', type=int, default=None)
    ap.add_argument('--variants', default='', help='comma subset; default = all')
    ap.add_argument('--metrics', default='clip,aes,ir,pick,hps')
    ap.add_argument('--hps_version', default='v2.1', choices=['v2.0', 'v2.1'])
    ap.add_argument('--score_only', default=None, help='rescore an existing output dir')
    ap.add_argument('--no_generate', action='store_true')
    ap.add_argument('--cudnn', choices=['on', 'off'], default='on')
    args = ap.parse_args()

    torch.backends.cudnn.enabled = (args.cudnn == 'on')
    device = torch.device('cuda')

    if args.score_only:
        args.out = args.score_only
    prompts = load_prompts(args.prompts_csv, args.n_prompts)
    args._prompts = prompts
    experts = [p for p in args.model_files.split(',') if p]
    tags = [expert_tag(p) for p in experts]
    expert0_variant = f'{tags[0]}_in_afa'
    all_variants = [f'only_{t}' for t in tags] + ['afa', expert0_variant]
    variants = all_variants if not args.variants else [v for v in args.variants.split(',') if v in all_variants]
    metrics = [m for m in args.metrics.split(',') if m]
    os.makedirs(args.out, exist_ok=True)
    log(f'{len(prompts)} prompts x {args.images_per_prompt} images | variants: {", ".join(variants)}')

    if args.ckpt:
        cfg_path = os.path.join(args.ckpt, 'aggregator_0', 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as fh:
                cfg = json.load(fh)
            for key, argname in (('hidden_size', 'aggregator_hidden_size'),
                                 ('num_layers', 'aggregator_num_layers'),
                                 ('num_attn_heads', 'aggregator_num_attn_heads')):
                saved = cfg.get(key)
                if saved is not None:
                    if getattr(args, argname) is not None and getattr(args, argname) != saved:
                        log(f'WARNING: --{argname}={getattr(args, argname)} vs checkpoint {key}={saved}; '
                            f'using the checkpoint value')
                    setattr(args, argname, saved)
            log(f'aggregator config from checkpoint: hidden={args.aggregator_hidden_size} '
                f'layers={args.aggregator_num_layers} heads={args.aggregator_num_attn_heads} '
                f'experts={cfg.get("num_experts")}')

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

    # ---- scoring ----
    results = {}
    detail = {}
    for metric in metrics:
        if metric == 'clip':
            results['clip_t'] = score_clip_t(args.out, variants, prompts, args, device)
        elif metric == 'aes':
            results['aes'] = score_aes(args.out, variants, args, device)
        elif metric == 'ir':
            results['ir'] = score_ir(args.out, variants, prompts, args, device)
        elif metric == 'pick':
            results['pick'] = score_pick(args.out, variants, prompts, args, device)
        elif metric == 'hps':
            results['hps'] = score_hps(args.out, variants, prompts, args, device)

    table = {}
    for name, scores in results.items():
        if not scores:
            continue
        means, _ = summarize(scores, variants, name, name == 'clip_t', prompts,
                             args.images_per_prompt)
        table[name] = means
    # keep the per-prompt detail too: the means alone cannot support a paired test
    for name, sc in results.items():
        if sc:
            detail[name] = {f'{v}|{i}': round(float(x), 4) for (v, i), x in sc.items()}
    json.dump(detail, open(os.path.join(args.out, 'paper_metrics_detail.json'), 'w'), indent=1)

    out_path = os.path.join(args.out, 'paper_metrics.json')
    merged = {}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            merged = json.load(fh)
    merged.update({k: {v: m for v, m in x.items()} for k, x in table.items()})
    json.dump(merged, open(out_path, 'w'), indent=1)

    if 'clip_t' in table and 'afa' in table['clip_t']:
        for ref in [v for v in variants if v != 'afa']:
            if ref in table['clip_t']:
                d = table['clip_t']['afa'] - table['clip_t'][ref]
                log(f'afa - {ref}: {d:+.2f} CLIP-T')
    log(f'wrote {os.path.join(args.out, "paper_metrics.json")}')


if __name__ == '__main__':
    main()
