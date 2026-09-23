"""Combine per-expert prediction captures into the metrics that decide whether
AFA's routing has anything learnable in the paper's expert setting.

For every (image, timestep) the experts saw the same latents, noise draws and
timestep, so predictions are paired. Reported per expert set:

  best single        - best expert by mean MSE
  mean-of-predictions- output-space ensemble (proxy for AFA's feature-level mean init)
  oracle per sample  - best whole-image expert per combo (uses the noise)
  oracle per element - per-element min over experts (uses the noise)
  cross-seed router  - the honest learnable ceiling: for a held-out noise draw,
                       route each element to the expert that won most often on the
                       *other* draws (majority vote of the other masks). This can
                       only exploit structure that persists across noise draws,
                       i.e. structure a trained aggregator could in principle learn.
  winner agreement   - per-element winner identity across noise draws vs the
                       independent-expert baseline (sum of squared win rates)
"""
import glob
import json
import os
import sys

import torch

ASSETS = os.environ.get('AFA_ASSETS', '/data/afa-assets')
PRED_DIR = f'{ASSETS}/output/expert_preds'
ORDER = ['sd15', 'rv', 'absolute_reality', 'majicmix_v6', 'epicrealism', 'ghibli', 'counterfeit']


def load_all(pred_dir):
    data = {}
    for f in sorted(glob.glob(os.path.join(pred_dir, '*.pt'))):
        name = os.path.basename(f)[:-3]
        d = torch.load(f, map_location='cpu')
        data[name] = {(c['img'], c['t'], c['seed']): (c['pred'].float(), c['noise'].float())
                      for c in d['combos']}
        print(f'  loaded {name}: {len(data[name])} combos', flush=True)
    return data


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--experts', default='', help='comma-separated subset')
    ap.add_argument('--pred_dir', default=PRED_DIR, help='directory of probe_multi_expert.py outputs')
    args = ap.parse_args()
    os.makedirs(args.pred_dir, exist_ok=True)
    data = load_all(args.pred_dir)
    names = [n for n in ORDER if n in data] + [n for n in data if n not in ORDER]
    if args.experts:
        want = args.experts.split(',')
        names = [n for n in want if n in data]
    if len(names) < 2:
        print('need at least two experts captured'); return
    keys = sorted(set(data[names[0]].keys()) & set(data[names[1]].keys()))
    for n in names[2:]:
        keys = sorted(set(keys) & set(data[n].keys()))
    imgs = sorted(set(k[0] for k in keys))
    timesteps = sorted(set(k[1] for k in keys))
    seeds = sorted(set(k[2] for k in keys))
    print(f'\n{len(names)} experts: {names}\n{len(keys)} combos '
          f'({len(imgs)} images x {len(timesteps)} t x {len(seeds)} noise draws)', flush=True)

    per_expert = {n: [] for n in names}
    mean_pred, oracle_samp, oracle_elem = [], [], []
    per_elem_choice = []
    for k in keys:
        e = {}
        for n in names:
            p, noise = data[n][k]
            e[n] = (p - noise) ** 2
            per_expert[n].append(e[n].mean().item())
        stack = torch.stack([data[n][k][0] for n in names])          # [E,4,64,64]
        stack_e = torch.stack([e[n] for n in names])                 # [E,4,64,64]
        mean_pred.append(((stack.mean(0) - data[names[0]][k][1]) ** 2).mean().item())
        oracle_samp.append(min(e[n].mean().item() for n in names))
        oracle_elem.append(stack_e.min(0).values.mean().item())
        per_elem_choice.append((k, stack_e.argmin(0)))

    def avg(x):
        return sum(x) / len(x)

    best = min(names, key=lambda n: avg(per_expert[n]))
    print(f'\n=== MSE (mean over {len(keys)} combos, lower is better) ===')
    for n in sorted(names, key=lambda n: avg(per_expert[n])):
        print(f'  {n:<20} {avg(per_expert[n]):.5f}')
    print(f'  {"mean-of-preds":<20} {avg(mean_pred):.5f}')
    print(f'  {"oracle-per-sample":<20} {avg(oracle_samp):.5f}  '
          f'({(avg(per_expert[best]) - avg(oracle_samp)) / avg(per_expert[best]) * 100:+.2f}% vs best single)')
    print(f'  {"oracle-per-element":<20} {avg(oracle_elem):.5f}  '
          f'({(avg(per_expert[best]) - avg(oracle_elem)) / avg(per_expert[best]) * 100:+.2f}% vs best single)')

    # winner identity persistence across noise draws, and the cross-seed router
    print(f'\n=== winner identity across noise draws (the learnable part) ===')
    choice = dict(per_elem_choice)   # argmin expert index per element
    router_mse, agree_obs, agree_indep = [], [], []
    for i in imgs:
        for t in timesteps:
            ks = [k for k in keys if k[0] == i and k[1] == t]
            if len(ks) < 2:
                continue
            masks = {k: choice[k] for k in ks}
            # observed agreement (identity-wise) between draws
            for a in range(len(ks)):
                for b in range(a + 1, len(ks)):
                    agree_obs.append((masks[ks[a]] == masks[ks[b]]).float().mean().item())
            # independent-expert baseline from win rates
            all_masks = torch.cat([masks[k].flatten() for k in ks])
            freq = torch.bincount(all_masks, minlength=len(names)).float() / all_masks.numel()
            agree_indep.append((freq ** 2).sum().item())
            # cross-seed router: majority vote of the other draws, applied to a held-out draw
            for h, kh in enumerate(ks):
                votes = torch.stack([masks[k] for j, k in enumerate(ks) if j != h])
                if len(names) == 2:
                    routed = (votes.float().mean(0) > 0.5).long()
                else:
                    counts = torch.stack([(votes == e).sum(0) for e in range(len(names))])
                    routed = counts.argmax(0)
                noise = data[names[0]][kh][1]
                stack = torch.stack([data[names[e]][kh][0] for e in range(len(names))])
                sel = torch.gather(stack, 0, routed.unsqueeze(0)).squeeze(0)
                router_mse.append(((sel - noise) ** 2).mean().item())
    print(f'  observed winner agreement={avg(agree_obs):.3f}   independent-expert baseline={avg(agree_indep):.3f}   '
          f'(chance for {len(names)} experts would be ~{1/len(names):.3f})')
    print(f'  cross-seed router MSE={avg(router_mse):.5f}  '
          f'({(avg(per_expert[best]) - avg(router_mse)) / avg(per_expert[best]) * 100:+.2f}% vs best single '
          f'"{best}")')
    print(f'  -> a router that can only use image-level structure {"DOES" if avg(router_mse) < avg(per_expert[best]) * 0.995 else "does NOT"} '
          f'beat the best single expert by >0.5%.', flush=True)

    json.dump(dict(names=names, per_expert={n: avg(per_expert[n]) for n in names},
                   mean_of_preds=avg(mean_pred), oracle_per_sample=avg(oracle_samp),
                   oracle_per_element=avg(oracle_elem), cross_seed_router=avg(router_mse),
                   agreement_observed=avg(agree_obs), agreement_independent=avg(agree_indep)),
              open(f'{args.pred_dir}/analysis.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
