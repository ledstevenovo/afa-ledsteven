"""Assemble the paper's Table 1/2 layout from the evaluation outputs.

Reads the DrawBench (paper_metrics.json) and COCO (coco_metrics.json) results and prints
the table with the paper's column order and row labels:

  COCO 2017:  FID ↓ | IS | CLIP-I | CLIP-T
  DrawBench:  AES | PS | HPSv2 | IR

Rows we can produce: the base models (each expert alone), AFA, and expert-0-inside-AFA.
The four model-merging baselines the paper lists (Weighted Merging, MBW, autoMBW,
MagicFusion) are other people's methods and are absent unless they are run separately.

  python3 make_table.py                       # prints markdown + writes table.csv/.tex
  python3 make_table.py --drawbench ... --coco ... --out DIR
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_generation import ASSETS, expert_tag  # noqa: E402

# column order of the paper's tables
COCO_COLS = [('fid', 'FID ↓'), ('is', 'IS'), ('clip_i', 'CLIP-I'), ('clip_t', 'CLIP-T')]
DB_COLS = [('aes', 'AES'), ('pick', 'PS'), ('hps', 'HPSv2'), ('ir', 'IR')]


def load(path):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def fmt(v, direct_better=True, nd=2):
    if v is None:
        return '-'
    if isinstance(v, str):
        return v
    return f'{v:.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drawbench', default=f'{ASSETS}/output/eval_drawbench/paper_metrics.json')
    ap.add_argument('--coco', default=f'{ASSETS}/output/eval_coco5k/coco_metrics.json')
    ap.add_argument('--model_files', default=None,
                    help='expert list used in training, to label Base Model A/B/C')
    ap.add_argument('--out', default=f'{ASSETS}/output')
    args = ap.parse_args()

    db = load(args.drawbench)
    coco = load(args.coco)
    if not db and not coco:
        raise SystemExit('no evaluation results yet')

    experts = [p for p in (args.model_files or '').split(',') if p]
    tags = [expert_tag(p) for p in experts] if experts else []
    if not tags:  # fall back to whatever the JSONs contain
        names = set()
        for d in (db, coco):
            for m in d.values():
                names.update(m)
        tags = sorted(n[5:] for n in names if n.startswith('only_'))

    rows = []
    for i, t in enumerate(tags):
        rows.append((f'Base Model {chr(65 + i)} ({t})', f'only_{t}'))
    rows.append(('Weighted Merging', None))
    rows.append(('MBW', None))
    rows.append(('autoMBW', None))
    rows.append(('MagicFusion', None))
    rows.append(('AFA (Ours)', 'afa'))
    if tags:
        rows.append((f'{tags[0]} in AFA (expert-0 only)', f'{tags[0]}_in_afa'))

    def cell(metric, variant):
        if variant is None:
            return None
        for d in (db, coco):
            if metric in d and variant in d[metric]:
                return d[metric][variant]
        return None

    lines = []
    lines.append('| Model | ' + ' | '.join(lbl for _, lbl in COCO_COLS + DB_COLS) + ' |')
    lines.append('|---|' + '---|' * len(COCO_COLS + DB_COLS))
    csv = ['model,' + ','.join(m for m, _ in COCO_COLS + DB_COLS)]
    for label, variant in rows:
        vals = [cell(m, variant) for m, _ in COCO_COLS + DB_COLS]
        lines.append(f'| {label} | ' + ' | '.join(fmt(v) for v in vals) + ' |')
        csv.append(label + ',' + ','.join('' if v is None else f'{v:.4f}' for v in vals))
    table = '\n'.join(lines)
    print(table)

    notes = [
        '',
        'Notes:',
        '  * FID/IS/CLIP-I/CLIP-T are on COCO 2017 at 256px; AES/PS/HPSv2/IR on DrawBench at 512px.',
        '  * Our COCO protocol uses 5,000 val image-caption pairs with 1 image each (the paper',
        '    reports 118,287 + 5,000 pairs); absolute FID/CLIP values are therefore not on the',
        '    same scale as the paper and only in-table comparisons are meaningful.',
        '  * IS is not implemented (torch-fidelity ships only an sdist here).',
        '  * Merging baselines (Weighted Merging, MBW, autoMBW, MagicFusion) are separate methods',
        '    and are shown as "-" unless they are run.',
        '  * DrawBench was evaluated once with 4 images per prompt; the paper reports 20 runs.',
    ]
    print('\n'.join(notes))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'table.csv'), 'w') as fh:
        fh.write('\n'.join(csv) + '\n')
    with open(os.path.join(args.out, 'table.md'), 'w') as fh:
        fh.write(table + '\n' + '\n'.join(notes) + '\n')
    print(f'\nwrote {args.out}/table.csv and table.md')


if __name__ == '__main__':
    main()
