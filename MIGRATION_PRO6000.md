# 迁移到 RTX PRO 6000（96GB）完整方案

> 目标：把这套 AFA 实验从 T4 15GB 迁到 96GB 卡上，跑论文配置（5–7 专家、batch 8–16、512px）并做**生成质量**评测。
> 关键判断：**大卡的价值是解锁规模 + 把一晚从"0.33 epoch"变成"2–3 套完整实验"**，不改变"MSE 最优 ≠ 画质最优"这个结构性问题（见第 7 节）。

---

## 0. 磁盘布局（AutoDL/Seetacloud：系统盘 vs 数据盘）——先做这一步

云实例的系统盘通常只有 30–50GB 且扩容贵，**数据盘**（AutoDL 是 `/root/autodl-tmp`，其它平台常见 `/data`、`/hy-tmp`）才是放资产的地方。用 `df -h` 确认哪个挂载点大。

```bash
# 代码也放数据盘（用绝对路径，别用相对目录，否则落在系统盘 /root）
git clone -b pro6000-migration <repo-url> /root/autodl-tmp/afa
cd /root/autodl-tmp/afa

# 资产与缓存一律指向数据盘；写进 ~/.bashrc 免得忘
export AFA_ASSETS=/root/autodl-tmp/afa-assets
export HF_HOME=/root/autodl-tmp/hf_home        # 默认在 ~/.cache/huggingface = 系统盘，下 15GB 模型会直接撑爆
mkdir -p "$AFA_ASSETS" "$HF_HOME"
echo 'export AFA_ASSETS=/root/autodl-tmp/afa-assets' >> ~/.bashrc
echo 'export HF_HOME=/root/autodl-tmp/hf_home'       >> ~/.bashrc
```

已经误放系统盘的话：`mv ~/afa /root/autodl-tmp/afa`（跨盘 mv 等价复制+删除；代码只有 ~1MB，秒完）。
本文档后续所有路径示例里的 `/data/afa-assets` 请按实际替换为 `$AFA_ASSETS`。

## 1. 环境（最容易踩的坑，务必先做）

**Blackwell 显卡（sm_120）需要 CUDA 12.8+ 与 PyTorch ≥ 2.7 的 cu128 轮子。**
本机验证：`torch.cuda.get_arch_list()` = `['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']` —— **当前的 torch 2.5.1+cu124 在 PRO 6000 上跑不了**（无 sm_120 内核）。

```bash
# 1) 先装 cu128 版 torch（不要用仓库 requirements 里的 torch==2.3.0）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2) 其余依赖按本机验证过的版本装（与仓库 requirements 略有差异，以这份为准）
pip install diffusers==0.21.4 transformers==4.39.3 accelerate==0.26.1 \
            huggingface-hub==0.23.5 datasets==2.21.0 safetensors==0.8.0 \
            Pillow tqdm numpy omegaconf aiohttp

# 3) 不要装 xformers。PyTorch SDPA 在新卡上走 FlashAttention 更快；
#    代码里对 xformers 是 try/except + 训练脚本会主动 disable，装了反而可能踩兼容问题。

# 4) 验证
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
python -c "import torch,x; x=torch.randn(4096,4096,device='cuda',dtype=torch.float16); print((x@x).abs().mean().item())"
```

**不要**再设置 `torch.backends.cudnn.enabled=False`——那是本机容器的 workaround；新脚本用 `--cudnn on` 正常开启（约 1.5–2× 提速）。

---

## 2. 需要带走的代码与文件

```bash
# 本机打包（含未提交的补丁；排除大文件）
cd /root/bayes-tmp/afa
tar czf /tmp/afa_code.tgz \
  --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  train_paper_setting.py run_pro6000.sh check_paper_run.sh \
  eval_generation.py probe_multi_expert.py analyze_multi_expert.py \
  probe_pair.py probe_oracle_learnability.py probe_mem.py train.py evaluate.py \
  models/ README.md AGENTS.md requirements.txt
```

带走的**关键补丁**（都在 `models/` 里，未提交）：
| 文件 | 改动 | 原因 |
|---|---|---|
| `models/modules/pipeline_stable_diffusion_aggregator.py` | `randn_tensor` 改从 `diffusers.utils.torch_utils` 导入 | 新版 diffusers 移除了顶层导出 |
| `models/model.py` | xformers 调用包 try/except；`load_init(..., aggregator_out_init_std=0.0)` 透传 | xformers 可能不可用；可切换聚合器初始化方式 |
| `models/modules/aggregator.py` | 新增 `out_init_std`（默认 0.0 = 论文原始零初始化）；修正 bias 零初始化的笔误 | 参数化可调 |
| `train_paper_setting.py` | 论文配置训练器：路由监控、固定 eval batch、步级 checkpoint/`--resume`、`--cudnn/--allow_tf32`、`--stop_hour`、`--grad_accum_steps` | 本方案的主脚本 |

---

## 3. 模型清单（论文配置）

论文用的是 **base SD1.5 + 6 个 Civitai 写实微调**；HF 镜像上可匿名获取的对应文件：

| # | 论文来源 | 本地文件名 | 获取（`HF_ENDPOINT=https://hf-mirror.com`，除 SD1.5 外都是单文件） | 大小 |
|---|---|---|---|---|
| 8 | `stable-diffusion-v1-5/stable-diffusion-v1-5` | `models/stable-diffusion-v1-5/`（diffusers 目录） | `snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5", allow_patterns=["unet/*","vae/*","text_encoder/*","tokenizer/*","scheduler/*","feature_extractor/*","model_index.json"])` | 4 GB |
| 9 | Civitai 25694 = **epiCRealism** | `_new_experts/epicrealism.safetensors` | `Kalashnikov/epiCRealism` → `epicrealism_naturalSinRC1VAE.safetensors` | 2.0 GB |
| 10 | Civitai 43331 = **majicMIX realistic** | `_new_experts/majicmix_v6.safetensors` | `digiplay/majicMIX_realistic_v6` → `majicmixRealistic_v6.safetensors` | 2.2 GB |
| 11 | Civitai 4201 = **Realistic Vision** | `realistic_vision_v5.1/Realistic_Vision_V5.1.safetensors` | `SG161222/Realistic_Vision_V5.1_noVAE` | 4.3 GB |
| 12 | Civitai 81458 = **AbsoluteReality** | `_new_experts/absolute_reality.safetensors` | `Lykon/AbsoluteReality` → `AbsoluteReality_1.8.1_pruned.safetensors` | 2.0 GB |
| 13 | Civitai 15003 = **未识别** | — | 需要 Civitai token 或问论文作者 | — |
| 14 | Civitai 97744 = **RealCartoon-Realistic** | `_new_experts/realcartoon_realistic.safetensors` | HF 上未找到对应 repo（只有 `7Whitefire7/RealCartoon3D`）；建议 Civitai API 下 ckpt 后用 `from_single_file` 直接读 | ~4 GB |
| 15 | LAION 美学打分器（**评测用**） | `models/aesthetic/` | `shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE`（也可用 `camenduru/improved-aesthetic-predictor`） | 3.5 MB + CLIP |

> 注意：**所有专家的 cross-attention 必须同为 768 维、输入 4 通道**（SD1.5 家族）。SDXL / SD2.x / inpainting 模型**不能用**（前者维度不同，后者是 9 通道输入）。

一键下载（新机器上跑）：
```bash
export HF_ENDPOINT=https://hf-mirror.com
python - <<'PY'
import os
from huggingface_hub import hf_hub_download, snapshot_download
A="/data/afa-assets/models"; os.makedirs(f"{A}/_new_experts", exist_ok=True)
snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5", local_dir=f"{A}/stable-diffusion-v1-5",
                  allow_patterns=["unet/*","vae/*","text_encoder/*","tokenizer/*","scheduler/*",
                                  "feature_extractor/*","model_index.json","*.json"])
jobs=[("SG161222/Realistic_Vision_V5.1_noVAE","Realistic_Vision_V5.1.safetensors","realistic_vision_v5.1/Realistic_Vision_V5.1.safetensors"),
      ("Kalashnikov/epiCRealism","epicrealism_naturalSinRC1VAE.safetensors","_new_experts/epicrealism.safetensors"),
      ("digiplay/majicMIX_realistic_v6","majicmixRealistic_v6.safetensors","_new_experts/majicmix_v6.safetensors"),
      ("Lykon/AbsoluteReality","AbsoluteReality_1.8.1_pruned.safetensors","_new_experts/absolute_reality.safetensors")]
for repo, fn, out in jobs:
    p=hf_hub_download(repo, fn, local_dir=f"{A}/_dl"); os.replace(p, f"{A}/{out}")
    print("ok", out, os.path.getsize(f"{A}/{out}")/2**30, "GB")
snapshot_download("shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE",
                  local_dir=f"{A}/aesthetic", allow_patterns=["*.pth","*.json","*.yaml","*.py","*.txt"])
print("ALL DONE")
PY
```

---

## 4. 数据（JourneyDB）

镜像上的 `wusize/journeydb`（`repo_type=dataset`）匿名可下：`captions.zip`（488MB，含 200 个 caption tar）+ `images/NNN.tar`（每片 736MB ≈ **20.9k 张 512×512 图**，共 200 片 ≈ 4.2M 张）。

```bash
export HF_ENDPOINT=https://hf-mirror.com
python - <<'PY'
import os, subprocess
from huggingface_hub import hf_hub_download
D="/data/afa-assets/data/journeydb_raw"; os.makedirs(D, exist_ok=True)
os.chdir(D)
hf_hub_download("wusize/journeydb", "captions.zip", repo_type="dataset", local_dir=".")
for i in range(5):                      # 5 片 ≈ 10.5 万张（≈论文量级）；想更多就把 5 改大
    fn=f"images/{i:03d}.tar"
    if os.path.isdir(f"{i:03d}"): continue
    p=hf_hub_download("wusize/journeydb", fn, repo_type="dataset", local_dir="./images")
    subprocess.run(["tar","-xf",p,"-C","."],check=True); os.remove(p)
    subprocess.run(["unzip","-o","-j","captions.zip",f"captions/{i:03d}.tar","-d","."],check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run(["tar","-xf",f"{i:03d}.tar","-C","."],check=True); os.remove(f"{i:03d}.tar")
    print("shard", i, "done")
PY
# 构建训练用 JSONL（image_file + text）
python - <<'PY'
import glob, json, os
out="/data/afa-assets/data/journeydb_data.json"; n=0
with open(out,"w") as w:
    for d in sorted(glob.glob("/data/afa-assets/data/journeydb_raw/0*")):
        for f in sorted(glob.glob(os.path.join(d,"*.jpg"))):
            j=f[:-4]+".json"
            if os.path.exists(j) and json.load(open(j)).get("caption","").strip():
                w.write(json.dumps({"image_file":f,"text":json.load(open(j))["caption"]})+"\n"); n+=1
print(out, n, "records")
PY
```

**规模建议**：5 片 = 10.5 万图（与论文 `num_data 10000 × 10 epochs` 相当）；10 片 = 21 万图；20 片 ≈ 42 万图（约 15GB）。

---

## 5. 训练命令（96GB 的显存数学 + 现成脚本）

每个专家的常驻开销 ≈ 1.72GB（UNet fp16）+ 0.25GB（CLIP fp16）；激活在 batch 2、2 专家时实测 ≈4.1GB，随专家数与 batch 近似线性：

| 专家数 | 权重 | batch 8 激活 | batch 8 合计 | batch 16 合计 |
|---|---|---|---|---|
| 5 | 10.4 GB | ~41 GB | **~51 GB** ✓ | ~92 GB（紧张） |
| 7 | 14.5 GB | ~57 GB | **~72 GB** ✓ | ~115 GB ✗ |

**推荐配置**：
- **7 专家：`BATCH=8 GRAD_ACCUM=1`**（≈72GB）
- **5 专家：`BATCH=12 GRAD_ACCUM=1`**（≈72GB）
- 都不要梯度累积、不要梯度检查点（那是 15GB 卡上的拐杖）。

```bash
# 一键（脚本会自动收集磁盘上所有专家）
cd /path/to/afa
AFA_ASSETS=/data/afa-assets BATCH=8 GRAD_ACCUM=1 WORKERS=16 \
  tmux new -d -s afa_run 'bash run_pro6000.sh'

# 想看时间限制就加 STOP_HOUR=9；想换保存目录用 SAVE=...
```

`run_pro6000.sh` 内部的实际命令（便于手工调整）：
```bash
python3 -u train_paper_setting.py \
  --model_files /data/afa-assets/models/stable-diffusion-v1-5,\
/data/afa-assets/models/realistic_vision_v5.1/Realistic_Vision_V5.1.safetensors,\
/data/afa-assets/models/_new_experts/epicrealism.safetensors,\
/data/afa-assets/models/_new_experts/absolute_reality.safetensors,\
/data/afa-assets/models/_new_experts/majicmix_v6.safetensors \
  --dataset_json_file /data/afa-assets/data/journeydb_data.json \
  --resolution 512 --batch_size 8 --grad_accum_steps 1 --num_workers 16 \
  --lr 1e-4 --weight_decay 0.01 --lr_warmup_steps 100 \
  --save_every_steps 500 --log_every_steps 10 --eval_every_steps 50 \
  --cudnn on --allow_tf32 --resume \
  --model_save_path /data/afa-assets/output/paper_run_pro6000
```

吞吐预期：~1–1.5 s/步（eff 8）→ **1 epoch（10.5 万图 / 8 = 13k 步）≈ 4–5.5 小时**；10 万图 ≈ 3.7 小时。

---

## 6. 验证阶梯（先筛再训，避免白跑）

1. **专家筛选（每个专家 ~2 分钟，强烈建议先做）**
   ```bash
   for spec in "sd15:/data/afa-assets/models/stable-diffusion-v1-5" ...; do
       python3 probe_multi_expert.py --expert "${spec#*:}" --name "${spec%%:*}"; done
   python3 analyze_multi_expert.py               # 输出 跨噪声路由收益 / oracle 上限 / 一致性
   ```
   判据：**跨噪声路由收益 > 2%** 才值得训练（本机实测：5 专家 +5.75%、3 专家 +4.63%、2 个同族 ≈ 0%）。
2. **500–1000 步小跑**：看日志里 `EVAL loss` 是否下降、`w0`（expert0 平均权重）是否偏离均匀值 `1/N`。
3. **长跑**：每 500 步存盘、可 `--resume`；崩溃由 `run_pro6000.sh` 自动续。
4. **生成质量评测（决定性）**
   ```bash
   python3 eval_generation.py --ckpt /data/afa-assets/output/paper_run_pro6000 \
       --out /data/afa-assets/output/eval_pro6000          # CLIPScore + 出图 + 四联对比图
   ```
   再加 **LAION 美学分**（`models/aesthetic/`）+ 可选 FID（用留出的 1 万张 JourneyDB 图算参考统计）。
   ⚠️ **不要用训练 MSE 判断成败**（见下节）。

---

## 7. 迁移前必须知道的结构性风险（换卡不解决）

- 本机实测：**MSE 上 base SD1.5 最好（0.05720），但生成质量上 Realistic Vision 更好（CLIPScore 30.86 vs 28.37）——两者反号。**
- 后果：MSE 训练出的聚合器会把权重压向 base SD1.5（本机 3 专家 run 里 `w0` 已从 0.333 涨到 0.76），训练指标漂亮但出图可能更差（本机用旧 2 专家 checkpoint 实测：聚合器 CLIPScore 30.01 vs 只用 RV 31.10）。
- 因此 PRO 6000 上要**同时跑两条线**：
  1. **论文原始目标（MSE）** → 验证"按论文方法能不能复现他们的增益"（他们的评测是 FID/CLIP/美学分）；
  2. **质量对齐目标**（单步去噪 → VAE 解码 → CLIP/美学分反传，或对专家做加权奖励）→ 96GB 才放得下 VAE 解码器 + CLIP 图像塔（15GB 卡上这条路被封死）。
- 只有第 2 条能回答"MSE 训练能不能变成画质提升"；如果第 1 条在生成指标上也输给单模型，那就说明**该论文的增益依赖他们特定的专家组合/数据**，而不是普遍成立。

---

## 8. 时间预期

| 步骤 | T4 15GB（现状） | RTX PRO 6000 96GB |
|---|---|---|
| 环境迁移（cu128 torch + 依赖） | — | 30–60 分钟 |
| 模型下载（≈15GB）+ 数据下载（5 片 ≈1GB/10.5 万图） | 已完成 | 30–60 分钟 |
| 专家筛选 | 30 分钟 | **5 分钟** |
| 训练到饱和（10 万图） | 33 小时 | **3–4 小时** |
| 生成质量评测（12 prompt × 4 变体） | ~1 小时 | **~15 分钟** |
| 一晚 9 小时产出 | 0.33 epoch | **2–3 套完整实验**（不同专家组合 / 目标函数） |

---

## 9. 迁移后立刻要做的三件事（清单）

- [ ] `python -c "import torch; torch.randn(8,4,64,64,device='cuda')"` 通过，且 `get_arch_list()` 含 `sm_120`
- [ ] `probe_multi_expert.py` + `analyze_multi_expert.py`：确认 5–7 专家的**跨噪声路由收益 > 2%**（否则先换专家组合，别训）
- [ ] 小跑 500 步确认 `EVAL loss` 下降 → 挂长跑（`run_pro6000.sh`） → 结束后 `eval_generation.py` 出 CLIPScore 对比图
