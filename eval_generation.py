"""Generation-quality evaluation of the trained AFA aggregator.

Quantitative + qualitative comparison on a fixed prompt set with identical initial
noise per prompt. The variants follow --model_files (the same comma-separated expert
list, in the same order, that training used):

  only_<expert>   one expert alone, its own pipeline (per-framework baseline)
  afa             the trained aggregator (Model.test_forward)
  <expert0>_in_afa  expert 0's features only inside the AFA framework: isolates the
                  aggregator's effect from VAE/text-encoder differences. "expert 0"
                  is the first entry of --model_files, exactly as in training.

Metrics
  * CLIPScore (CLIP ViT-L/14 image-text cosine x100) per image, paired per prompt
  * routing statistics for the AFA variant: per-aggregator mean attention weight
    of expert 0 and mean |attn - 0.5|, averaged over the denoising trajectory
  * PNGs + per-prompt comparison grids for visual inspection

Runs phases sequentially and frees memory between them.
"""
import argparse
import gc
import json
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import Model  # noqa: E402
from models.modules.aggregator import Aggregator  # noqa: E402

ASSETS = os.environ.get('AFA_ASSETS', '/data/afa-assets')
CKPT = f'{ASSETS}/output/paper_run'
SD15 = f'{ASSETS}/models/stable-diffusion-v1-5'
# The paper's Group I (Sec. 4.1), in the order used for training below.
PAPER_GROUP1 = f'{ASSETS}/models/_new_experts/epicrealism.safetensors,' \
               f'{ASSETS}/models/_new_experts/majicmix_v6.safetensors,' \
               f'{ASSETS}/models/realistic_vision_v5.1/Realistic_Vision_V5.1.safetensors'
CLIP_PATH = f'{ASSETS}/models/clip-vit-large-patch14'

PROMPTS = [
    # photo-style
    ('photo', 'a photo of a golden retriever puppy running on a beach at sunset'),
    ('photo', 'a close-up portrait photo of an elderly man with weathered skin, natural light'),
    ('photo', 'a photo of a busy street market in the rain, neon signs, wet reflections'),
    ('photo', 'a photo of a snow-covered mountain village at blue hour'),
    ('photo', 'a photo of a bowl of ramen with steam rising, top-down view'),
    ('photo', 'a macro photo of a dragonfly resting on a dew-covered leaf'),
    # anime-style
    ('anime', 'anime girl with silver hair holding a glowing sword, night city background'),
    ('anime', 'anime style illustration of a magical forest with floating islands'),
    ('anime', 'a cute anime cat wearing a wizard hat, studio ghibli style'),
    ('anime', 'anime mecha robot standing in a ruined city, dramatic lighting'),
    ('anime', 'anime portrait of a boy with red eyes and a black cloak'),
    ('anime', 'anime style cherry blossom schoolyard on a spring afternoon'),
]

STEPS, GUIDANCE, RES = 50, 7.5, 512


def slug(t):
    return re.sub(r'[^a-z0-9]+', '_', t.lower())[:48]


def expert_tag(path):
    """Filesystem-safe short name for one expert path; used as the variant name."""
    stem = os.path.basename(path.rstrip('/'))
    if stem.endswith('.safetensors'):
        stem = stem[: -len('.safetensors')]
    return re.sub(r'[^a-z0-9]+', '_', stem.lower()).strip('_')


def free(obj):
    del obj
    gc.collect()
    torch.cuda.empty_cache()


def make_grid(paths, labels, out_path):
    ims = [Image.open(p).convert('RGB') for p in paths]
    w, h = ims[0].size
    grid = Image.new('RGB', (w * len(ims), h + 24), 'white')
    d = ImageDraw.Draw(grid)
    for i, (im, lab) in enumerate(zip(ims, labels)):
        grid.paste(im, (i * w, 24))
        d.text((i * w + 6, 6), lab, fill='black')
    grid.save(out_path)


def generator_for(seed):
    return torch.Generator(device='cuda').manual_seed(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{ASSETS}/output/eval_gen')
    ap.add_argument('--phases', default='all', help="'all', or a comma-separated subset of the variants")
    ap.add_argument('--clip', default=CLIP_PATH)
    ap.add_argument('--ckpt', default=CKPT, help='aggregator checkpoint dir to evaluate')
    ap.add_argument('--model_files', default=PAPER_GROUP1,
                    help='comma-separated expert paths, in the SAME order training used')
    ap.add_argument('--st_model_file', default=SD15, help='SD1.5 dir: shared VAE/tokenizer/scheduler')
    ap.add_argument('--aggregator_hidden_size', type=int, default=None,
                    help='default: read from {ckpt}/aggregator_0/config.json')
    ap.add_argument('--aggregator_num_layers', type=int, default=None,
                    help='default: read from {ckpt}/aggregator_0/config.json')
    ap.add_argument('--aggregator_num_attn_heads', type=int, default=None,
                    help='default: read from {ckpt}/aggregator_0/config.json')
    ap.add_argument('--cudnn', choices=['on', 'off'], default='on',
                    help='off = T4-container workaround (broken cuDNN), on = normal GPUs')
    ap.add_argument('--max_prompts', type=int, default=0, help='use only the first N prompts')
    ap.add_argument('--swap_experts', default='', help='i,j: swap conv_out channels i and j after loading the '
                    'aggregator -- counterfactual "same routing policy, other expert preferred"')
    args = ap.parse_args()
    torch.backends.cudnn.enabled = (args.cudnn == 'on')

    experts = [p for p in args.model_files.split(',') if p]
    tags = [expert_tag(p) for p in experts]
    expert0_variant = f'{tags[0]}_in_afa'
    variants = [f'only_{t}' for t in tags] + ['afa', expert0_variant]
    phases = variants if args.phases == 'all' else [p for p in args.phases.split(',') if p in variants]
    if args.max_prompts:
        global PROMPTS
        PROMPTS = PROMPTS[:args.max_prompts]
    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda')
    print(f'start {time.strftime("%Y-%m-%d %H:%M:%S")} | cudnn={args.cudnn} | {RES}px | '
          f'{STEPS} steps | guidance {GUIDANCE}', flush=True)
    for tag, path in zip(tags, experts):
        print(f'  expert {tag}: {path}', flush=True)
    print(f'variants: {", ".join(variants)}\nphases: {", ".join(phases)}', flush=True)

    scores = {}   # (variant, i) -> clipscore
    routing = {}

    # ---------- single experts, each in its own pipeline ----------
    for tag, path in zip(tags, experts):
        variant = f'only_{tag}'
        if variant not in phases:
            continue
        t0 = time.time()
        # original_config_file: single-file checkpoints carry no diffusers config, so without
        # this diffusers fetches v1-inference.yaml from raw.githubusercontent.com -- which is
        # unreachable here and hangs on connect instead of failing fast.
        yaml = os.path.join(args.st_model_file, 'v1-inference.yaml')
        pipe = Model.load_sd_model(path, torch_dtype=torch.float16,
                                   original_config_file=yaml if os.path.exists(yaml) else None)
        if pipe is None:
            print(f'[{variant}] {path} not found, skipping', flush=True)
            continue
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=False)
        vdir = os.path.join(args.out, variant)
        os.makedirs(vdir, exist_ok=True)
        print(f'[{variant}] pipeline loaded in {time.time() - t0:.0f}s', flush=True)
        for i, (kind, prompt) in enumerate(PROMPTS):
            p = os.path.join(vdir, f'{i:02d}_{kind}_{slug(prompt)}.png')
            if not os.path.exists(p):
                with torch.no_grad():
                    out = pipe(prompt, height=RES, width=RES, num_inference_steps=STEPS,
                               guidance_scale=GUIDANCE, generator=generator_for(1000 + i)).images[0]
                out.save(p)
            print(f'[{variant}] {i + 1}/{len(PROMPTS)} {p}', flush=True)
        free(pipe)

    # ---------- the AFA model ----------
    if 'afa' in phases or expert0_variant in phases:
        missing = [p for p in experts if not os.path.exists(p)]
        if missing:
            print(f'experts missing on disk: {missing}', flush=True)
            return
        # The saved aggregator config is authoritative: building the model with different
        # flags only fails later, as a state_dict size error, after the whole training run.
        agg_cfg_path = os.path.join(args.ckpt, 'aggregator_0', 'config.json')
        if os.path.exists(agg_cfg_path):
            with open(agg_cfg_path) as fh:
                agg_cfg = json.load(fh)
            for key, argname in (('hidden_size', 'aggregator_hidden_size'),
                                 ('num_layers', 'aggregator_num_layers'),
                                 ('num_attn_heads', 'aggregator_num_attn_heads')):
                given = getattr(args, argname)
                saved = agg_cfg.get(key)
                if given is not None and saved is not None and given != saved:
                    print(f'WARNING: --{argname}={given} but the checkpoint was trained with '
                          f'{key}={saved}; using the checkpoint value', flush=True)
                setattr(args, argname, saved if saved is not None else given)
            n_saved = agg_cfg.get('num_experts')
            if n_saved is not None and n_saved != len(experts):
                print(f'ERROR: the checkpoint was trained with {n_saved} experts but '
                      f'--model_files lists {len(experts)} ({", ".join(tags)})', flush=True)
                return
            print(f'aggregator architecture from {agg_cfg_path}: hidden={args.aggregator_hidden_size} '
                  f'layers={args.aggregator_num_layers} heads={args.aggregator_num_attn_heads} '
                  f'experts={n_saved}', flush=True)
        else:
            print(f'no {agg_cfg_path}: falling back to --aggregator_* flags '
                  f'(hidden={args.aggregator_hidden_size} layers={args.aggregator_num_layers} '
                  f'heads={args.aggregator_num_attn_heads})', flush=True)
        t0 = time.time()
        model = Model.load_init(model_paths=experts, st_model_path=args.st_model_file,
                                hidden_size=args.aggregator_hidden_size,
                                num_layers=args.aggregator_num_layers,
                                num_attn_heads=args.aggregator_num_attn_heads)
        # Inference path: everything fp16, matching Model.load_pretrained(..., dtype=fp16).
        # (Training keeps the VAE fp32 because autocast handles the aggregator/UNet dtypes;
        # here there is no autocast, so a mixed fp32-params/fp16-input graph would error.)
        model.vae.requires_grad_(False); model.text_encoders.requires_grad_(False)
        model.unets.requires_grad_(False); model.aggregators.requires_grad_(False)
        model.vae.to(device, torch.float16)
        model.text_encoders.to(device, torch.float16)
        model.unets.to(device, torch.float16)
        model.aggregators.to(device, torch.float16)
        model.eval()
        meta = json.load(open(f'{args.ckpt}/train_meta.json')) if os.path.exists(
            f'{args.ckpt}/train_meta.json') else json.load(open(f'{args.ckpt}/checkpoint_meta.json'))
        for i, agg in enumerate(model.aggregators):
            agg.load_state_dict(Aggregator.load_pretrained(f'{args.ckpt}/aggregator_{i}').state_dict())
        if args.swap_experts:
            a, b = [int(x) for x in args.swap_experts.split(',')]
            for agg in model.aggregators:
                w = agg.conv_out.weight.data
                w[[a, b]] = w[[b, a]]
            print(f'[afa] swapped conv_out channels {a}<->{b} (routing preference flipped)', flush=True)
        print(f'[afa] model + epoch-{meta["epoch"]} aggregators loaded in {time.time() - t0:.0f}s',
              flush=True)

        _orig = Aggregator.forward

        def pick_expert0(self, features, temb, encoder_hidden_states):
            attn = torch.zeros(features.shape[0], features.shape[1], *features.shape[-2:],
                               device=features.device)
            return features[:, 0], attn

        for variant in [v for v in ('afa', expert0_variant) if v in phases]:
            Aggregator.forward = _orig if variant == 'afa' else pick_expert0
            vdir = os.path.join(args.out, variant)
            os.makedirs(vdir, exist_ok=True)
            for i, (kind, prompt) in enumerate(PROMPTS):
                p = os.path.join(vdir, f'{i:02d}_{kind}_{slug(prompt)}.png')
                if os.path.exists(p):
                    print(f'[{variant}] {i + 1}/{len(PROMPTS)} cached', flush=True)
                    continue
                t1 = time.time()
                with torch.no_grad():
                    out = model.test_forward(prompt=prompt, height=RES, width=RES,
                                             num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                                             generator=generator_for(1000 + i))
                out.images[0].save(p)
                if variant == 'afa' and out.inference_attn_maps:
                    # inference_attn_maps: tuple over steps of (per-aggregator maps)
                    per_agg = []
                    for a in range(len(out.inference_attn_maps[0])):
                        vals, devs = [], []
                        for step_maps in out.inference_attn_maps:
                            m = step_maps[a].float()
                            vals.append(m[:, 0].mean().item())          # expert-0 weight
                            devs.append((m - 0.5).abs().mean().item())
                        per_agg.append(dict(expert0_weight=float(np.mean(vals)),
                                            attn_dev=float(np.mean(devs))))
                    routing[prompt] = per_agg
                print(f'[{variant}] {i + 1}/{len(PROMPTS)} {time.time() - t1:.0f}s {p}', flush=True)
        Aggregator.forward = _orig
        free(model)

    # ---------- grids ----------
    gdir = os.path.join(args.out, 'grids')
    os.makedirs(gdir, exist_ok=True)
    for i, (kind, prompt) in enumerate(PROMPTS):
        paths, labels = [], []
        for v in variants:
            p = os.path.join(args.out, v, f'{i:02d}_{kind}_{slug(prompt)}.png')
            if os.path.exists(p):
                paths.append(p); labels.append(v)
        if len(paths) > 1:
            make_grid(paths, labels, os.path.join(gdir, f'{i:02d}_{kind}_{slug(prompt)}.png'))
    print(f'grids written to {gdir}', flush=True)

    if routing:
        rp = os.path.join(args.out, 'routing_stats.json')
        json.dump(routing, open(rp, 'w'), indent=1)
        n_layers = len(next(iter(routing.values())))
        mean_by_layer = [np.mean([routing[p][a]['expert0_weight'] for p in routing])
                         for a in range(n_layers)]
        dev_by_layer = [np.mean([routing[p][a]['attn_dev'] for p in routing])
                        for a in range(n_layers)]
        print(f'\n[afa] routing over {len(routing)} prompts, {n_layers} aggregation points:', flush=True)
        print('  mean expert-0 weight per aggregation point: '
              + ' '.join(f'{v:.2f}' for v in mean_by_layer), flush=True)
        print('  mean |attn-0.5| per aggregation point:      '
              + ' '.join(f'{v:.3f}' for v in dev_by_layer), flush=True)
        print(f'  overall expert-0 weight={np.mean(mean_by_layer):.3f} '
              f'overall |attn-0.5|={np.mean(dev_by_layer):.3f}  (0.5 / 0.0 = totally uniform)', flush=True)
        print(f'  routing stats -> {rp}', flush=True)

    # ---------- CLIPScore ----------
    if os.path.isdir(args.clip):
        from transformers import CLIPModel, CLIPProcessor
        t0 = time.time()
        clip = CLIPModel.from_pretrained(args.clip).to(device).eval()
        proc = CLIPProcessor.from_pretrained(args.clip)
        print(f'CLIP (ViT-L/14) loaded in {time.time() - t0:.0f}s', flush=True)
        with torch.no_grad():
            txt = proc(text=[p for _, p in PROMPTS], return_tensors='pt',
                       padding=True, truncation=True).to(device)
            tfeat = clip.get_text_features(**txt)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
            for v in variants:
                for i, (kind, prompt) in enumerate(PROMPTS):
                    p = os.path.join(args.out, v, f'{i:02d}_{kind}_{slug(prompt)}.png')
                    if not os.path.exists(p):
                        continue
                    im = Image.open(p).convert('RGB')
                    ifeat = clip.get_image_features(**proc(images=im, return_tensors='pt').to(device))
                    ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
                    scores[(v, i)] = float((ifeat @ tfeat[i:i + 1].T).item() * 100)
        free(clip)

        print('\n=== CLIPScore (x100, higher = better text-image alignment) ===', flush=True)
        print(f'{"prompt":<52} ' + ' '.join(f'{v:>10}' for v in variants), flush=True)
        for i, (kind, prompt) in enumerate(PROMPTS):
            row = [scores.get((v, i)) for v in variants]
            print(f'{kind:>5} {prompt[:46]:<46} ' + ' '.join(
                f'{x:>10.2f}' if x is not None else f'{"-":>10}' for x in row), flush=True)
        print(f'\n{"MEAN":<52} ' + ' '.join(
            f'{np.mean([scores[(v, i)] for i in range(len(PROMPTS)) if (v, i) in scores]):>10.2f}'
            for v in variants), flush=True)
        if ('afa', 0) in scores and (expert0_variant, 0) in scores:
            d = [scores[('afa', i)] - scores[(expert0_variant, i)] for i in range(len(PROMPTS))
                 if ('afa', i) in scores and (expert0_variant, i) in scores]
            print(f'afa - {expert0_variant} (same VAE/text-encoder, isolates the aggregator): '
                  f'mean={np.mean(d):+.2f}  win/tie/loss='
                  f'{sum(1 for x in d if x > 0.1)}/{sum(1 for x in d if abs(x) <= 0.1)}/'
                  f'{sum(1 for x in d if x < -0.1)}', flush=True)
        json.dump({f'{v}|{i}': s for (v, i), s in scores.items()},
                  open(os.path.join(args.out, 'clipscores.json'), 'w'), indent=1)
    else:
        print(f'CLIP not found at {args.clip}: images + routing stats only', flush=True)
    print(f'done {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


if __name__ == '__main__':
    main()
