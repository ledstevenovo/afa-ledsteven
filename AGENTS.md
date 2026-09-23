# Repository Guidelines

## Project Structure & Module Organization

This repository implements Adaptive Feature Aggregation (AFA).

- `train.py`: dataset loading, preprocessing, and aggregator training using Accelerate.
- `evaluate.py`: prompt-based image generation and attention-map export; not an automated test suite.
- `models/model.py`: model loading, training/inference orchestration, and checkpoint handling.
- `models/modules/`: SABW aggregator, adapted U-Net blocks, and Stable Diffusion pipeline.
- `README.md`: paper overview and citation; `LICENSE`: usage terms.

`tests/test_regressions.py` covers training and inference regressions. No datasets or checkpoints are bundled. Keep experimental data and outputs outside version control.

## Build, Test, and Development Commands

Run commands from the repository root. See `tests/README.md` for dependencies and installation commands; this code imports older Diffusers internal modules.

- `python -m compileall train.py evaluate.py models`: check Python syntax without loading model weights; this does not validate dependencies or runtime behavior.
- `python train.py --help` and `python evaluate.py --help`: inspect CLI options after installing dependencies.
- Single-image inference with an existing saved AFA model and CUDA:

```sh
python evaluate.py --model_path /path/to/afa --prompt "A mountain lake" --num_images_per_prompt 1 --images_saved_path image.png --attn_maps_saved_path attention.pt
```

Create output directories first. Single-image output retains the requested filename; multiple images use indexed names such as `image_0.png`.

Training needs local expert models and JSON records containing `image_file` and `text`. Pass comma-separated expert names through `--model_files` and a checkpoint directory through `--model_save_path`. Training uses FP16 autocast with FP32 VAE and aggregator parameters. Validate on one GPU first.

For the paper's multi-expert setting (5-7 experts, effective batch 8, step-based resume, routing monitors) use `train_paper_setting.py`, wrapped by `run_paper_setting.sh` (15 GB T4) or `run_pro6000.sh` (>= 48 GB; that wrapper enables cuDNN, which is broken in the T4 container). Two rules learned the hard way: gate long runs on the offline headroom measurement (`probe_multi_expert.py` + `analyze_multi_expert.py`; cross-seed router gain must exceed ~2%, otherwise two correlated experts give zero learnable routing and the loss is flat no matter the hyperparameters), and judge results with generation metrics (`eval_generation.py`, CLIPScore) rather than the training MSE. See `MIGRATION_PRO6000.md` and `RESULTS_T4.md`.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/variables, and `PascalCase` classes. Follow nearby quoting and type-hint conventions. Preserve tensor-shape expectations and keep pretrained components frozen when changing aggregator training. No formatter or linter is configured; avoid unrelated reformatting.

## Testing Guidelines

Run `python -m unittest discover -s tests -v`. Tests use standard-library unittest and small random models; no downloads are needed. CUDA training checks require xformers and skip without CUDA. Add focused cases under `tests/test_*.py` for behavior changes. No coverage threshold exists. Record the GPU and dependency versions; small-model checks do not establish full-resolution training viability or paper reproduction.

## Commit & Pull Request Guidelines

History contains only `Initial commit` and `init commit`; no established convention is evident. Use concise imperative subjects, such as `Fix training checkpoint argument`. Keep changes focused. PRs should explain the problem, affected modules, validation commands/results, and relevant issues. Include generated examples for visual changes and disclose unverified GPU behavior. Never commit credentials, model weights, or private datasets.
