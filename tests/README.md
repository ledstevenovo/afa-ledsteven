# Regression tests

Run from the repository root:

```sh
python -m unittest discover -s tests -v
python train.py --help
python evaluate.py --help
```

The tests use random, small Diffusers U-Nets, a VAE, CLIP text encoders, and the actual AFA modules. They do not download pretrained weights or datasets. The CUDA test runs the actual single-GPU training loop with Accelerate and xformers; it verifies an FP32 aggregator update through frozen FP16 experts and FP32 VAE encoding. It skips when CUDA is unavailable. Other tests cover layer counts, gradients, aggregator checkpoint round-trip, CLI defaults, and image filenames/content. Image-generation results and the training save call are mocked in CLI tests; these are not full-model checkpoint or image-quality tests.

## Validation environment

Use Python 3.9 and install the CUDA PyTorch stack as described in the [root README](../README.md). Dependency versions are maintained in [requirements.txt](../requirements.txt). From the repository root:

```sh
python -m pip install -r requirements.txt
```

These versions target the repository's existing Diffusers interfaces. They are not a claim that current releases are compatible. Do not replace an existing environment's packages in place.

Validated on Windows with an RTX 4060 Laptop GPU (8 GB): all six tests passed, including CUDA training. The aiohttp pin avoids an import-time Windows certificate-store error encountered with 3.13.5 on this machine. Optional Triton/Flash Attention availability warnings and existing unclosed JSON-file warnings did not prevent these tests from passing.

Full SD1.5 expert weights, real training data, T4 peak memory, long training runs, and paper metrics require separate validation. Multi-GPU execution is outside this regression suite.
