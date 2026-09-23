# AFA 复现报告（RTX PRO 6000 / Blackwell）

> 对照论文：*Ensembling Diffusion Models via Adaptive Feature Aggregation*（arXiv 2405.17082，Tencent AI Lab + 南京大学）
> 实验日期：2026-09-23（训练 15:18–17:34，评测 17:35–00:52）
> 硬件：单卡 NVIDIA RTX PRO 6000 Blackwell 96GB；208 核 / 1TB 内存
> 所有数字来自本仓库脚本的实际运行输出，原始结果见文末"产出文件"。

---

## 0. 摘要

**做成了什么**：在论文的专家分组 Group I（epiCRealism + majicMIX realistic v6 + Realistic Vision V5.1）上，按论文的训练配方（JourneyDB 10,000 图 × 10 epoch、batch 8、lr 1e-4、wd 0.01、DDIM 50 步、CFG 7.5）训练了 SABW 聚合器（2 层 Transformer，45.13M 可训练参数，论文称 ~50M），并用论文的指标做了生成质量评测。

**主要结论**：

| # | 结论 | 证据 |
|---|---|---|
| 1 | **AFA 在 DrawBench ImageReward 上高于全部三个单专家**，方向与论文一致 | AFA +0.4556 vs 最好单专家 RV +0.3486，配对差 **+0.1070 ± 0.0299（3.6 SE）**，逐 prompt 胜 124 / 负 76 |
| 2 | CLIP-T（文本对齐）与 CLIP-I（逐图保真）上 AFA 均排名第一 | DrawBench CLIP-T 27.86（RV 27.70）；COCO CLIP-I 0.700（RV 0.694） |
| 3 | **但两项未达标**：AES 输给 majicMIX v6（5.53 vs 5.60）；FID 输给 RV（29.82 vs 28.96） | 见 §2 |
| 4 | **同框架对照下的路由差值尚不确定**：AFA 比 `epiCRealism in AFA` 高 +0.0857 IR，但配对 95% 区间为 [−0.0045, +0.1759] | 同框架 expert-0 vs 单独 epiCRealism 为 +0.240；AFA vs 同框架 expert-0 为 +0.086 |
| 5 | **MSE 目标在第 1 个 epoch 后改进很小**；后续 eval_loss 仍有约 0.0001 的波动和微小下降 | eval_loss 0.09178 → 0.09147（−0.34%），随后 9 个 epoch 的表中范围为 0.09137–0.09150 |
| 6 | 路由塌成"近似硬选择 + 丢弃一个专家" | `\|attn−0.5\|`=0.484（近上限 0.5）；epiCRealism 权重 **0.022** |
| 7 | **专家版本是待核验的复现差异来源之一**：我们的三个单专家 CLIP-T 相对极差约 4.7%，论文对应极差约 2.1% | 我们 CLIP-T 26.43–27.70（约 4.7% 极差/均值），论文 0.2566–0.2620（约 2.1%） |

**判定：部分复现。** AFA 在 ImageReward 上优于全部三个单专家；AES 和 FID 被不同单专家反超。COCO 规模、FID 实现、DrawBench 重复次数及专家版本均与论文有差异，因此不能直接比较绝对数值。

---

## 1. 实验配置

### 1.1 与论文的逐项对照

| 维度 | 论文 | 本次复现 | 一致性 |
|---|---|---|---|
| 专家集合 | Group I：EpicRealism / MajicMixRealistic / RealisticVision | epiCRealism(naturalSinRC1VAE) / majicMIX realistic v6 / Realistic Vision V5.1(noVAE) | 名义一致，**版本未核验**（论文给的是 Civitai modelVersionId 134065/176425/130072） |
| 专家数 | 每组 3 个（无 base SD1.5） | 3 个 | ✅ |
| 训练数据 | JourneyDB 10,000 样本 × 10 epoch = 100k | JourneyDB 10,000 图 × 10 epoch = 100k（从 104,698 条记录中随机抽样） | ✅ |
| batch size / 有效 batch | 8 | 8（无梯度累积） | ✅ |
| 优化器 / lr / wd | AdamW / 1e-4 / 0.01 | AdamW / 1e-4 / 0.01（warmup 100 步） | ✅ |
| 文本 dropout | 0.1 | 0.1 | ✅ |
| 分辨率 | 512 | 512 | ✅ |
| 采样器 | DDIM 50 步、CFG 7.5、4 图/prompt | DDIM 50 步、CFG 7.5、4 图/prompt | ✅ |
| 聚合器可训练参数 | ~50M | **45.13M**（hidden 128 / 2 层 / 8 头 → 25 个聚合点） | 接近 |
| 评测集（COCO） | 118,287 + 5,000 对 | 5,000 val2017 对 × 1 图 | ⚠️ 缩小 |
| 评测集（DrawBench） | 200 prompt，评测 20 次 | 200 prompt，单遍 4 图 | ⚠️ 次数不同 |

### 1.2 训练配置与实测开销

```
专家 3 个（论文 Group I）
数据 10,000 图 × 10 epoch = 12,500 步（每 epoch 1,250 步 × batch 8）
聚合器 hidden_size 128 / num_layers 2 / num_attn_heads 8 → 45.13M 参数
```

| 项目 | 实测 |
|---|---|
| 训练步速 | **0.64–0.66 s/步**（batch 8，512px，3 专家，无梯度累积/检查点） |
| 显存 | 峰值 **38.7 GB** / 96 GB（batch 8；离线探针测量：3 专家 batch 8 ≈ 34 GB） |
| 总时长 | 2 小时 17 分（15:17:58 → 17:34:33） |
| 推理成本（512px） | 单专家 ~0.7–1.7 s/图；AFA（3 专家 + 25 聚合器）**2.0 s/图** |
| 推理成本（256px） | AFA **0.7 s/图** |
| 崩溃/重启 | **0 次**（12,500 步全程；一次早期崩溃由训练数据坏图引起，见 §6） |

### 1.3 被放弃的对照线（记录备查）

- **104,698 图 × 1 epoch** 的"更多数据"线：跑到 step 1500 后放弃，改用论文配方（10k×10）。checkpoint 归档在 `output/paper_run_aborted_104k_x1epoch/`。
- 该线在 step 1500 时的状态（可作参考）：eval_loss 0.09881→0.09417、w0 0.333→0.021。

---

## 2. 结果

### 2.1 论文格式对照表（本次复现）

| Model | FID ↓ | IS | CLIP-I ↑ | CLIP-T ↑ | AES ↑ | PS | HPSv2 | IR ↑ |
|---|---|---|---|---|---|---|---|---|
| Base A = epiCRealism | 45.25 | – | 0.668 | 26.71 | 5.369 | – | – | +0.130 |
| Base B = majicMIX v6 | 31.36 | – | 0.680 | 26.43 | **5.596** | – | – | +0.186 |
| Base C = Realistic Vision V5.1 | **28.96** | – | 0.694 | 27.70 | 5.447 | – | – | +0.349 |
| **AFA (Ours)** | 29.82 | – | **0.700** | **27.86** | 5.535 | – | – | **+0.456** |
| *(附加对照)* epiCRealism in AFA | – | – | – | 27.72 | 5.525 | – | – | +0.370 |
| Weighted Merging / MBW / autoMBW / MagicFusion | 未跑（见 §4.3） | | | | | | | |

- FID/IS/CLIP-I/CLIP-T：COCO val2017 @256px，5,000 对，每对 1 图（参考集 = 同尺寸真实图 5,000 张）
- AES/PS/HPSv2/IR：DrawBench 200 prompt × 4 图 @512px（每变体 800 图）
- `epiCRealism in AFA` 是**论文没有的对照行**：同一个专家（expert 0）放进 AFA 框架但跳过聚合器，用于剥离 VAE/文本塔差异

### 2.2 逐指标分析

| 指标 | AFA 排名 | AFA vs 最好单专家 | AFA vs 同框架 expert-0 | 判读 |
|---|---|---|---|---|
| **IR**（DrawBench 指标） | **1/4** | **+0.107**（3.6 SE，显著） | **+0.086**（1.9 SE，95% 区间跨 0） | 支持论文主张 |
| **CLIP-I** | **1/4** | +0.006 | – | 数值领先，尚无配对不确定性估计 |
| **CLIP-T**（DrawBench） | **1/4** | +0.16（配对 95% 区间跨 0） | +0.15（配对 95% 区间跨 0） | 数值领先，未证实稳定提升 |
| **CLIP-T**（COCO 5k） | **1/4** | +0.35（vs MMR） | – | 大样本下同样第一 |
| **AES** | 2/4 | **−0.062**（MMR 更好） | +0.010 | 未达论文结论 |
| **FID** | 2/4 | **+0.86**（RV 更低更好） | – | **分布级真实感未改善** |

### 2.3 配对检验（IR，n=200 prompt，同 prompt 同种子，每 prompt 4 图取均值）

| 对比 | 差值 ± SE | SE 倍数 | 胜/负 |
|---|---|---|---|
| AFA − Realistic Vision V5.1（最好单专家） | **+0.1070 ± 0.0299** | 3.6 | 124 / 76 |
| AFA − majicMIX v6 | +0.2697 ± 0.0424 | 6.4 | 139 / 61 |
| AFA − epiCRealism | +0.3260 ± 0.0523 | 6.2 | 135 / 65 |
| AFA − epiCRealism in AFA（同 VAE/文本塔） | +0.0857 ± 0.0457 | 1.9 | 109 / 91 |

胜/负数按仓库保存的 `drawbench_metrics_detail.json` 中每 prompt 四图均值计算；IR 差值的配对 95% 区间：对 RV 为 [+0.0480, +0.1659]，对同框架 expert-0 为 [−0.0045, +0.1759]。该区间反映这 200 个 prompt 的配对差异，单次采样尚未覆盖跨 seed 变异。

### 2.4 早期小样本信号（12 自选 prompt × 1 图，CLIPScore）

| 变体 | CLIPScore |
|---|---|
| RV 单专家 | 30.93 |
| AFA | 30.55 |
| epiCRealism in AFA | 29.44 |
| epiCRealism 单专家 | 29.37 |
| majicMIX v6 单专家 | 28.19 |

AFA − RV = −0.38 ± 2.42（7 胜 5 负，**统计上平局**），AFA − epiCRealism-in-AFA = +1.11（8 胜 4 负）。
**这张表说明小样本评测的判别力不足**（n=12）：它与 200 prompt 的 IR/CLIP-T 结论方向相反，后者才可采信。

---

## 3. 训练过程与"为什么增益有限"

### 3.1 逐 epoch 实测

| epoch | train_loss | eval_loss | eval_w0 | eval_dev |
|---|---|---|---|---|
| 0 | 0.16011 | 0.09178 | 0.018 | 0.453 |
| 1 | 0.15846 | **0.09147** | 0.009 | 0.479 |
| 2 | 0.15823 | 0.09150 | 0.010 | 0.482 |
| 3 | 0.15828 | 0.09142 | 0.011 | 0.483 |
| 4 | 0.15824 | 0.09143 | 0.010 | 0.483 |
| 5 | 0.15816 | 0.09139 | 0.011 | 0.482 |
| 6 | 0.15799 | 0.09137 | 0.012 | 0.483 |
| 7 | 0.15780 | 0.09137 | 0.011 | 0.485 |
| 8 | 0.15764 | 0.09139 | 0.012 | 0.484 |
| 9 | 0.15748 | 0.09141 | 0.011 | 0.485 |

**eval_loss 的主要下降发生在 epoch 1**：0.09178 → 0.09147（−0.34%），随后 9 个 epoch 的表中数值落在 0.09137–0.09150；最终相对 epoch 1 仅低约 0.00006。即**后续训练的 MSE 收益很小**——这与 T4 阶段后期 loss 变化有限的观察方向一致。

### 3.2 路由演化

- expert-0（epiCRealism）平均权重：0.333（初始均匀）→ **0.011–0.022**
- `|attn−0.5|`（路由偏离均匀的程度）：0.167（初始）→ **0.484**（理论上限 0.5）

即聚合器学到的解是：**几乎不用 epiCRealism（它 MSE 最差 0.1134），在 majicMIX v6 与 Realistic Vision 之间做逐像素近硬选择**。25 个聚合点中，聚合器在 0.00–0.15 的权重区间内几乎完全放弃 expert-0。

### 3.3 可学空间 vs 实际获益

离线探针（`probe_multi_expert.py` + `analyze_multi_expert.py`，共享 latent/noise/timestep 的配对比较）：

| 专家集合 | 跨噪声路由收益（可学习上界） | 一致性 vs 独立基线 |
|---|---|---|
| **论文 Group I（ER+MMR+RV）** | **+4.05%**（288 combos） | 0.468 vs 0.336 |
| (sd15, rv) 配对 | +2.74% | 0.592 vs 0.503 |
| 含 sd15 的四专家 | +5.31% | 0.401 vs 0.272 |

而聚合器在固定 eval batch 上实际只拿到 **0.34%** 的 MSE 改进（0.09178 → 0.09147）。两者使用不同评测集和构造，不能用 4% 与 0.34% 的差值估算聚合器实现了多少可学空间。

```text
这组离线筛选显示可利用的 MSE 路由空间；固定评测 batch 的训练损失只小幅下降。二者需要在同一批样本上比较才能定量归因。
```

### 3.4 同框架对照

用 `epiCRealism in AFA`（同框架、同 VAE/文本塔、只跳过聚合器）做对照：

| 对照 | IR 均值或差值 |
|---|---|
| epiCRealism 单独运行 | +0.130 |
| **同框架 expert-0 相对单独 epiCRealism** | **+0.240** |
| **AFA 相对同框架 expert-0 的差值** | **+0.086**（95% 区间跨 0） |

这项分解只适用于 epiCRealism 这一对照，不能单独识别 VAE 的因果贡献，也不能推广到最强单专家 RV。相对同框架 expert-0 的 +0.086 IR 仍有统计不确定性。

---

## 4. 与论文的差距及可能原因

### 4.1 专家版本尚未核验

| | 三个 base model 的 CLIP-T 离散度 | AES 离散度 |
|---|---|---|
| 论文（Table 1） | 0.2566 / 0.2602 / 0.2620 → **约 2.1% 极差/均值** | 5.4102 / 5.4624 / 5.4881 → **0.08** |
| 本次复现 | 26.43 / 26.71 / 27.70 → **约 4.7% 极差/均值** | 5.369 / 5.447 / 5.596 → **0.23** |

我们的三个专家在这组 CLIP-T 数值上的相对极差约为论文的 2.3 倍；评测协议又不同，这不能单独证明版本是主因。论文给的是 Civitai **modelVersionId**（ER 134065 / MMR 176425 / RV 130072），本次用的是 HF 上同名文件（`Kalashnikov/epiCRealism`、`digiplay/majicMIX_realistic_v6`、`SG161222/Realistic_Vision_V5.1_noVAE`），**版本对应关系未核验**。若版本不同，"专家多样性过高/过低"会直接影响聚合器的可学空间与增益幅度。

### 4.2 FID 略输 RV 的含义

FID 29.82（AFA）vs 28.96（RV）、31.36（MMR）、45.25（ER）：

- AFA 落在最好与次好单专家之间；这个排序本身不能确定成因；
- CLIP-I（0.700，最高）与这一结果并存；仅凭这两个汇总指标不能确定是多样性、锐度还是其他分布差异；
- 我们的 FID 使用 torchvision InceptionV3 权重（见 §5），绝对值不可与论文对比，但可比较同一替代权重与处理流程下的表内相对数值。

### 4.3 未检验的论文主张

论文的核心论据之一是"动态特征聚合优于静态权重融合"（对比 Weighted Merging、MBW、autoMBW、MagicFusion）。本次**四个融合 baseline 都未运行**，因此这半句主张在本次复现中没有被检验。

- MagicFusion 官方代码已获取（GitHub 不可达，经 jsDelivr 拉取 90 个文件至 `third_party/MagicFusion/`），其算法已读懂：**training-free 的 per-pixel "CFG 响应显著性" 选择**（`ldm/models/diffusion/ddim.py::p_sample_ddim`，超参 `fusion_selection="600,1,1"`、`merge_mode=2`）。原版代码无法在 sm_120 上直接运行（其环境为 python3.8 + torch1.10），需把 SNB 移植进本仓库的 diffusers 管线。
- Weighted Merging / MBW 为权重合并方法，实现成本低。

---

## 5. 协议差异与局限（引用本报告数字前必读）

| # | 差异 | 影响 |
|---|---|---|
| 1 | **COCO 规模缩小**：5,000 对 × 1 图（论文 118,287 + 5,000 对，每方法 4 图） | FID/CLIP-I/CLIP-T 的**绝对值不可与论文并排比较**，只能看表内相对关系 |
| 2 | **FID 权重替代**：标准 FID 权重托管于 GitHub Releases（本机不可达，HF/ModelScope 无镜像），改用 **torchvision InceptionV3** 权重 | 所有变体与参考集使用同一权重 → 表内对比有效；**绝对值非标准 FID** |
| 3 | DrawBench 单遍评测（论文 "evaluated 20 times"） | 单遍 4 图/prompt 的标准协议；方差未通过重复评测压缩 |
| 4 | 缺 IS / PickScore / HPSv2 | 权重在 HF 的 Xet 存储上被限速（~100 kB/s，实测 24 分钟仅 5MB），本次按要求停掉 |
| 5 | 专家版本未核验 | §4.1 |
| 6 | 只有 Group I | 论文的 Group II（AbsoluteReality / CyberRealistic / RealCartoonRealistic）未跑；CyberRealistic 在 HF 有 `cyberdelia/CyberRealistic` 可用 |
| 7 | 融合 baseline 未跑 | §4.3 |

---

## 6. 工程记录

### 6.1 环境（已验证可用）

```text
torch 2.8.0+cu128（arch list 含 sm_120）/ Python 3.12.3
diffusers 0.21.4 / transformers 4.39.3 / datasets 2.21.0 / accelerate 0.26.1
numpy 1.26.4 / pandas 2.2.0 / pyarrow 15.0.1 / safetensors 0.4.5
venv 复用 base 环境的 torch（--system-site-packages），35 个 wheel 离线安装（见 requirements-pro6000-lock.txt）
```

网络现实（决定了后续所有工程选择）：

| 目标 | 状态 |
|---|---|
| `huggingface.co` | ❌ 不可达（必须 `HF_ENDPOINT=https://hf-mirror.com`） |
| `github.com` / `codeload` / `raw.githubusercontent.com` | ❌ 不可达（**这是本次最多坑的根源**） |
| `hf-mirror.com` | ✅ 可用，单连接 1.4–3.6 MB/s，多连接聚合 4–7 MB/s |
| HF 的 **Xet 存储**（`cas-bridge.xethub.hf.co`） | ⚠️ 部分文件被限速到 ~100 kB/s（PickScore / CLIP-ViT-H 权重因此未取到） |
| `mirrors.aliyun.com/pypi` | ❌ 403（容器默认 pip 源是坏的） |
| `pypi.tuna.tsinghua.edu.cn` | ⚠️ 每 ~18MB 断连接（pip 不续传 → 必须 `--resume-retries` 或换源） |
| `repo.huaweicloud.com/pypi` | ✅ 4.9 MB/s（本次主力源） |
| `download.pytorch.org` | ✅ 355 kB/s |
| `jsDelivr`（`cdn.jsdelivr.net/gh/...`） | ✅ 可代理 GitHub 仓库文件（MagicFusion 代码即由此取得） |
| `modelscope.cn` | ⚠️ 可用但仅 2.9 MB/s，且无本任务需要的权重镜像 |

### 6.2 修复的问题清单（按影响排序）

**训练/加载类：**

1. `sys.path` / `ASSETS` 在 8 个 py + 4 个 shell 脚本中硬编码为 T4 机器路径 → 全部改为脚本目录 + `AFA_ASSETS` 环境变量。
2. `run_pro6000.sh` 默认资产目录指向不存在的 `/data`（会在仅 30GB 的根分区建目录）→ 改为仓库同级目录。
3. 两个训练 wrapper 使用系统 `python3`（依赖装在 venv）→ 改为优先 `$REPO/.venv/bin/python`。
4. wrapper 里 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` → 会让 `from_single_file` 无法取 CLIP 文本塔 config 而加载失败，**已移除**。
5. **单文件专家的加载必须显式传本地 `v1-inference.yaml`**：否则 diffusers 会去 `raw.githubusercontent.com` 取配置，在墙内**不是失败而是挂死**（进程连上不可达 IP 后无限等待）。修了三处：`Model.load_init`、`eval_generation.py` 单专家路径、`eval_paper_metrics.gen_expert`。
6. `--cudnn off` 硬编码（T4 容器 workaround）→ 改为 `--cudnn on|off` 且默认 on（PRO 6000 必须开启）。
7. **训练数据里有 4 张损坏 JPEG**（104,745 张中 4 张，0.004%，全在 shard 000）：导致 DataLoader worker 抛 `OSError: image file is truncated` 使整个 run 退出。修法：`ImageFile.LOAD_TRUNCATED_IMAGES = True` + `load_image()` 兜底。

**评测类（8 个，全部会导致数字错误）：**

8. **ImageReward 配对错误**：`ImageReward.score(a_list, b_list)` 返回两个列表的**交叉配对**（4×4→16），传整变体会把错配对的均值当成分数（表现为五变体 IR 全挤在 −2.19±0.02）。改为"1 prompt + 它自己的 N 张图"（返回 N 个匹配值）。
9. **CUDA 张量直接 `np.asarray`** → `score_aes` 崩溃（AES 阶段整段丢失）。
10. **汇总函数用文件名前缀匹配整数键** → IR 只统计了 index ≥100 的 prompt（子集均值）。
11. `gen_expert` **不创建输出目录** → DrawBench 全量与 COCO 两个阶段都在 2 分钟后崩于 `FileNotFoundError`。
12. `score_fid` 使用未 import 的 `glob` → COCO 指标阶段在算完 CLIP-T/CLIP-I 后崩溃。
13. **FID 结果未写入 JSON**（只打印）→ 表格缺 FID 列。
14. CLIP 特征逐图前向（batch 1）→ CLIP-T+CLIP-I 需 34 分钟；改批处理（batch 64）后 ~8 分钟。
15. **结果只在最后一次性落盘** → 任何晚期崩溃会让已算完的指标全部丢失（实际发生过两次，各损失约 35 分钟计算）。改为每算完一个指标即写盘并合并。

**下载类：**

16. SD1.5 用 `unet/*` 通配下载 → 拉下 6 套权重变体（.bin/.fp16/.non_ema）共 ~11GB 冗余；改为显式 14 个文件清单（3.97GB）。
17. `curl -C -` 与 `-r` 混用 → curl 从 range 内的偏移续传并**追加整个响应体**，分片文件膨胀到 9.5GB（文件本身只有 3.97GB）；改为"写 scratch 文件 → 校验长度 → 追加"。
18. 多进程并行下载同一文件的 `.parts` 目录会互相覆盖 → 按组拆分进程（sd15/clip/experts/journeydb 各自独立）。

### 6.3 被放弃的尝试（避免重复劳动）

- 用 `snapshot_download` 拉 `nlphuji/mscoco_2014_5k...` → 该数据集走鉴权路径失败；**直接抓文件可用**（1.4 MB/s）。
- 官方 COCO 站点（`images.cocodataset.org`）→ 94 kB/s，弃用。
- ModelScope 的 JourneyDB → 原始版式（每片 ~15GB 的 tgz、caption 在 jsonl），比 wusize 重打包版（736MB/片 + 每图一个 json）麻烦，弃用。
- PickScore / CLIP-ViT-H-14 权重（HF Xet 限速）→ 停掉（用户决定）。
- 104k × 1 epoch 的训练线 → 停掉，改用论文的 10k × 10。

---

## 7. 结论与下一步建议

### 7.1 复现判定

| 论文主张 | 本次结果 | 判定 |
|---|---|---|
| AFA 在偏好指标（IR）上优于所有 base model | IR 第一，对最好单专家 +0.107（显著） | ✅ 成立 |
| AFA 在 CLIP-T / CLIP-I 上更好 | 两项数值第一；DrawBench CLIP-T 对 RV 的配对区间跨 0 | ⚠️ 数值领先，稳定提升未证实 |
| AFA 在 AES 上更好 | 第二，落后 MMR 0.06 | ❌ 未成立 |
| AFA 在 FID 上更好 | 第二，落后 RV 0.86 | ❌ 未成立 |
| 增益来自聚合器 | 同框架对照差 +0.086 IR，95% 区间跨 0；管线差异未单独拆开 | ⚠️ 未能确定贡献大小 |
| 优于静态权重融合方法 | 未检验 | ⬜ 未知 |

### 7.2 建议的下一步（按性价比排序）

1. **核验专家版本**（最高优先）：用 Civitai API 把 modelVersionId（ER 134065 / MMR 176425 / RV 130072）落到具体文件，重跑单专家基线对比离散度。这是与论文数字对齐的最大缺口。
2. **跑融合 baseline**（MagicFusion 代码已就位）：至少补 Weighted Merging 与 MBW（权重合并，实现成本低），才能检验"动态聚合优于静态融合"。
3. **针对 MSE 饱和做实验**：既然 MSE 在第 1 个 epoch 饱和，可尝试（a）加大聚合器容量（3 层 → 57M）看是否改变饱和点；（b）改用质量对齐目标（单步去噪 → VAE 解码 → CLIP/美学分反传，96GB 显存足以承载）——这是本项目 `MIGRATION_PRO6000.md` §7 提出的方向，本次数据进一步支持它。
4. **补齐指标**：IS（需构建 torch-fidelity）、PickScore、HPSv2；FID 换回标准权重（需要能访问 GitHub Releases 的网络）。
5. **扩大 COCO 规模**：若要与论文数字直接比较，需要 118k 级别的生成量（单卡不现实，可考虑多卡或缩到 30k 并明确标注）。

---

## 8. 产出文件

| 路径 | 内容 |
|---|---|
| `output/final_table.txt`、`output/table.md`、`output/table.csv` | 论文格式对照表 |
| `output/eval_drawbench/paper_metrics.json` | DrawBench 五指标（CLIP-T/AES/IR；含 ±PS/HPSv2 缺失说明） |
| `output/eval_drawbench/paper_metrics_detail.json` | per-prompt 明细（可复算配对检验） |
| `output/eval_coco5k/coco_metrics.json` | COCO 的 CLIP-T / CLIP-I / FID |
| `output/eval_group1_10k/` | 12-prompt CLIPScore 评测（含对比图 `grids/`、`routing_stats.json`） |
| `output/paper_run_group1_10k/` | 训练产物：25 个聚合器（2 层，45.13M 参数）+ `train_meta.json`（1,250 条日志） |
| `output/expert_preds/` + `analysis.json` | 专家可学空间探针的原始预测与分析 |
| `logs/` | 训练与本报告所有评测阶段的完整日志 |
| `third_party/MagicFusion/` | MagicFusion 官方代码（90 文件，经 jsDelivr 取得） |
| `requirements-pro6000-lock.txt` | 本机验证过的完整依赖钉版（含 3 条易踩的版本约束） |
| `download_assets.py`、`chunked_fetch.py` | 素材下载脚本（含上文 §6.2 的 16–18 号修复） |
| `eval_paper_metrics.py`、`eval_coco_protocol.py`、`make_table.py` | 论文口径评测与制表脚本 |

---

*报告基于 2026-09-23 的实际运行数据。所有"未验证/未运行"项已在 §5 与 §7.1 明确标注。*
