| Model | FID ↓ | IS | CLIP-I | CLIP-T | AES | PS | HPSv2 | IR |
|---|---|---|---|---|---|---|---|---|
| Base Model A (epicrealism) | 45.25 | - | 0.67 | 26.71 | 5.37 | - | - | 0.13 |
| Base Model B (majicmix_v6) | 31.36 | - | 0.68 | 26.43 | 5.60 | - | - | 0.19 |
| Base Model C (realistic_vision_v5_1) | 28.96 | - | 0.69 | 27.70 | 5.45 | - | - | 0.35 |
| Weighted Merging | - | - | - | - | - | - | - | - |
| MBW | - | - | - | - | - | - | - | - |
| autoMBW | - | - | - | - | - | - | - | - |
| MagicFusion | - | - | - | - | - | - | - | - |
| AFA (Ours) | 29.82 | - | 0.70 | 27.86 | 5.53 | - | - | 0.46 |
| epicrealism in AFA (expert-0 only) | - | - | - | 27.72 | 5.52 | - | - | 0.37 |

Notes:
  * FID/IS/CLIP-I/CLIP-T are on COCO 2017 at 256px; AES/PS/HPSv2/IR on DrawBench at 512px.
  * Our COCO protocol uses 5,000 val image-caption pairs with 1 image each (the paper
    reports 118,287 + 5,000 pairs); absolute FID/CLIP values are therefore not on the
    same scale as the paper and only in-table comparisons are meaningful.
  * IS is not implemented (torch-fidelity ships only an sdist here).
  * Merging baselines (Weighted Merging, MBW, autoMBW, MagicFusion) are separate methods
    and are shown as "-" unless they are run.
  * DrawBench was evaluated once with 4 images per prompt; the paper reports 20 runs.
