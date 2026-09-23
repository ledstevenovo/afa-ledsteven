<div align="center">
<h1> Ensembling Diffusion Models via Adaptive Feature Aggregation</h1>

<p align="center">
Cong Wang<sup>1*</sup>, Kuan Tian<sup>2*</sup>, Yonghang Guan<sup>2</sup>, Jun Zhang<sup>2†</sup>, Zhiwei Jiang<sup>1†</sup>, Fei Shen<sup>2</sup>, Xiao Han<sup>2</sup>, Qing Gu<sup>1</sup>, Wei Yang<sup>2</sup>
<br>
<sup>1</sup> Nanjing University,
<sup>2</sup> Tencent AI Lab
<br>
<sup>*</sup> Equal contribution.
<sup>†</sup> Corresponding authors.
<br><br>
[<a href="https://arxiv.org/abs/2405.17082" target="_blank">arXiv</a>]
<br>
</div>

## Abstract

The success of the text-guided diffusion model has inspired the development and release of numerous powerful diffusion models within the open-source community.
These models are typically fine-tuned on various expert datasets, showcasing diverse denoising capabilities. 
Leveraging multiple high-quality models to produce stronger generation ability is valuable, but has not been extensively studied.
Existing methods primarily adopt parameter merging strategies to produce a new static model. 
However, they overlook the fact that the divergent denoising capabilities of the models may dynamically change across different states, such as when experiencing different prompts, initial noises, denoising steps, and spatial locations. 
In this paper, we propose a novel ensembling method, Adaptive Feature Aggregation (AFA), which dynamically adjusts the contributions of multiple models at the feature level according to various states (i.e., prompts, initial noises, denoising steps, and spatial locations), thereby keeping the advantages of multiple diffusion models, while suppressing their disadvantages.
Specifically, we design a lightweight Spatial-Aware Block-Wise (SABW) feature aggregator that adaptive aggregates the block-wise intermediate features from multiple U-Net denoisers into a unified one.
The core idea lies in dynamically producing an individual attention map for each model's features by comprehensively considering various states.
It is worth noting that only SABW is trainable with about 50 million parameters, while other models are frozen. 
Both the quantitative and qualitative experiments demonstrate the effectiveness of our proposed Adaptive Feature Aggregation method.

## Installation and checks

Use an isolated Python 3.9 environment. From the repository root, install the CUDA 12.1 PyTorch wheels first, then the pinned dependencies:

```sh
python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

Training and inference require an NVIDIA GPU with a compatible driver. The CUDA availability check should print `True`; the CUDA regression test skips if no GPU is available. `requirements.txt` pins direct dependencies and selected compatibility dependencies, not every transitive package.

The small-model tests were validated on Windows with an RTX 4060 Laptop GPU. Full-model training on a T4 and paper metrics remain unverified. See [tests/README.md](tests/README.md) for test coverage and limitations. Pretrained models and datasets must be supplied separately.

## Multi-expert training (paper setting) and large-GPU migration

`train.py` trains an aggregator over two experts with a fixed small batch. For the paper's setting
(5-7 realistic SD1.5 finetunes on JourneyDB, effective batch 8+) the repo now ships a screening gate,
a training entry point and a generation-based evaluation:

```sh
# 1. screen the expert set BEFORE spending GPU time: is anything learnable? (~2 min per expert)
python3 probe_multi_expert.py --expert /path/to/expert.safetensors --name name
python3 analyze_multi_expert.py            # go/no-go: cross-seed router gain > ~2%

# 2. train (wrappers run_paper_setting.sh / run_pro6000.sh add crash-retry with --resume)
python3 train_paper_setting.py --model_files A,B,C --dataset_json_file records.json \
    --batch_size 8 --grad_accum_steps 1 --cudnn on --resume --model_save_path out

# 3. judge with generation metrics, not with the training loss
python3 eval_generation.py --ckpt out --out eval_out
```

- `MIGRATION_PRO6000.md` - moving to a 96 GB card (RTX PRO 6000): environment (torch cu128 is
  required for Blackwell), model/data sources, memory sizing, training commands, validation ladder.
- `RESULTS_T4.md` - every measurement from the T4 stage: why the loss stays flat with two
  correlated experts, per-expert-set learnable headroom, and the CLIPScore comparison.
- `requirements-pro6000.txt` - dependency set validated for sm_120 (do **not** install xformers).

## Reference
```
@article{wang2024ensembling,
  title={Ensembling Diffusion Models via Adaptive Feature Aggregation},
  author={Wang, Cong and Tian, Kuan and Guan, Yonghang and Zhang, Jun and Jiang, Zhiwei and Shen, Fei and Han, Xiao and Gu, Qing and Yang, Wei},
  journal={arXiv preprint arXiv:2405.17082},
  year={2024}
}
```
