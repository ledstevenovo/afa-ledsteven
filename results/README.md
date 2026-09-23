# Reproduction results (RTX PRO 6000, 2026-09-23)

Raw outputs behind `../RESULTS_PRO6000.md` — read that report for interpretation,
protocol caveats and the list of what was **not** verified.

| File | Content |
|---|---|
| `final_table.txt`, `table.md`, `table.csv` | Paper-format comparison table (base models / AFA / expert-0-in-AFA) |
| `drawbench_metrics.json` | DrawBench means per variant: CLIP-T, AES, ImageReward (200 prompts x 4 images) |
| `drawbench_metrics_detail.json` | Same, per prompt (for paired significance tests) |
| `coco_metrics.json` | COCO@256 val2017 5k: CLIP-T, CLIP-I, FID |
| `clipscores_12prompts.json` | Small-sample CLIPScore (12 prompts) — shown to be underpowered |
| `inference_routing_stats.json` | Per-aggregator attention statistics of the AFA variant at inference |
| `expert_screening_analysis.json` | Offline expert-set screening (cross-seed router gain) |
| `train_meta.json`, `train_paper_setting_10k.log` | Training history (12,500 steps, 10 epochs) and the full log |

Generated images (4,000 x 512px DrawBench + 25,000 x 256px COCO), model weights and
checkpoints are NOT in version control (they live outside the repository, ~25 GB).
