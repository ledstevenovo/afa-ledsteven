# Regression tests

Run from the repository root:

```sh
python -m unittest discover -s tests -v
python train.py --help
python evaluate.py --help
```

The tests use random, small Diffusers U-Nets, a VAE, CLIP text encoders, and the actual AFA modules. They do not download pretrained weights or datasets. The CUDA test runs the actual single-GPU training loop with Accelerate and xformers; it verifies an FP32 aggregator update through frozen FP16 experts and FP32 VAE encoding. It skips when CUDA is unavailable. Other tests cover layer counts, gradients, aggregator checkpoint round-trip, CLI defaults, and image filenames/content. Image-generation results and the training save call are mocked in CLI tests; these are not full-model checkpoint or image-quality tests.

## Validation environment

The compatibility environment uses Python 3.9, PyTorch 2.3.0 with CUDA 12.1, torchvision 0.18.0, NumPy 1.24.3, and Pillow 11.3.0. Install the following into an isolated environment with that PyTorch stack:

```sh
python -m pip install diffusers==0.20.2 transformers==4.33.3 accelerate==0.23.0 huggingface-hub==0.25.2 datasets==2.14.7 pyarrow==14.0.2 safetensors==0.4.5 xformers==0.0.26.post1 aiohttp==3.8.6
```

These versions target the repository's existing Diffusers interfaces. They are not a claim that current releases are compatible. Do not replace an existing environment's packages in place.

Validated on Windows with an RTX 4060 Laptop GPU (8 GB): all six tests passed, including CUDA training. The aiohttp pin avoids an import-time Windows certificate-store error encountered with 3.13.5 on this machine. Optional Triton/Flash Attention availability warnings and existing unclosed JSON-file warnings did not prevent these tests from passing.

Full SD1.5 expert weights, real training data, T4 peak memory, long training runs, and paper metrics require separate validation. Multi-GPU execution is outside this regression suite.
