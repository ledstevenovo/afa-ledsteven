# T4 阶段的实测结果（迁移前的判断依据）

本文件汇总在单卡 T4 15GB 上跑完的测量，是迁移到 96GB 显卡时的配置依据。所有数字都是本机实测，脚本见本仓库
`probe_*.py` / `analyze_multi_expert.py` / `eval_generation.py` / `train_paper_setting.py`。

## 1. 为什么"训练 loss 一直平坦"——不是超参问题

用 (Realistic Vision V5.1, SD1.5) 这对专家在 1403 张 COCO 上训练 42 个 epoch：

| 区间 | avg_loss |
|---|---|
| epoch 1–10（lr 1e-4 constant） | 0.15697 ± 0.00623 |
| epoch 11–27（同） | 0.15786 ± 0.00443 |
| epoch 28–42（lr 5e-5 cosine） | 0.15675 ± 0.00297 |

线性趋势 **+1.85e-5/epoch**（42 epoch 累计 +0.0008，统计上为零）；epoch 1 的 0.14634 是全程最低。

**排除掉的原因（都是对照实验，不是推理）**：

| 怀疑点 | 实验 | 结果 |
|---|---|---|
| 学习率 | 5e-5 vs 1e-4 | 终点相同 |
| 有效 batch | 2 vs 8（梯度累积） | 250 步达到 60k 步的同一 loss；60k 步 vs 125 步终点逐位相同（eval_loss 0.09942） |
| 优化器写法 | `optimizer.step()`（GradScaler 未 unscale）vs `scaler.step()` | 轨迹逐位相同（0.09945 vs 0.09944），scale 全程未变 |
| fp16 精度 | 聚合器 fp32 前向 vs fp16 | 无差异 |
| 参数化 | `conv_out` 零初始化 vs `out_init_std=0.02` | 150 步终点相同（w0 0.13 / 0.12）；位移测量显示即使零初始化，conv_in/transformer 也在以 ~0.25×lr/步 学习 |
| 训练时长 | 250 步 vs 60k 步 | 同一终点 |

**真因**：这对专家在训练目标上的可提升空间只有 **~0.6%**（见下节），而这个量级远小于 epoch 间噪声（sd 0.005）。

## 2. 专家组合决定"可学习空间"（关键测量）

配对测量（同一 latent / noise / timestep，512px，每对 8 图 × 4 timestep）：

| 专家组合 | 整图级 oracle | 逐元素 oracle | **跨噪声路由（可学习）** | 一致性 vs 独立基线 |
|---|---|---|---|---|
| RV + base SD1.5（2 个） | +0.02% | +10.9% | ≈ 0 | 0.573 vs ~0.51 |
| RV + Counterfeit（二次元） | +0.22% | +13.1% | ≈ 0 | 0.579 vs ~0.51 |
| RV + Ghibli | +0.02% | +10.9% | ≈ 0 | 0.579 vs ~0.51 |
| **sd15 + rv + epicrealism（3 个）** | +0.01% | +19.1% | **+4.63%** | 0.487 vs 0.355 |
| **论文 5 专家**（sd15, rv, AbsoluteReality, majicMIX v6, epiCRealism） | +0.01% | +22.1% | **+5.75%** | 0.387 vs 0.242 |

指标说明：
- **逐元素 oracle** = 每个潜元素取"最接近真噪声"的专家（用了真值噪声，**不可学习**，只是绝对上界）。
- **跨噪声路由** = 用同一图像在*其他噪声实现*上得到的逐元素胜负（多数投票）去路由*留出的*那个噪声——只用图像级信息，因此是**可学习上界**。
- **一致性** = 逐元素胜负身份在不同噪声实现间的一致率，与"独立专家"基线的对比（越高于基线，说明胜负越由图像决定、越可学）。

结论：**2 个同族专家之间"谁更好"接近硬币（一致性仅略高于基线）→ 训不出东西；5 个专家时一致性显著高于基线（0.387 vs 0.242）→ 有 ~5.75% 的可学习空间。** 这解释了为什么之前无论怎么调超参都没用，也说明迁移后必须用满 5–7 个专家。

## 3. 生成质量：MSE 最优 ≠ 画质最优（必须平行验证）

对 42-epoch checkpoint 做 CLIPScore 评测（12 prompt × 4 变体，同 seed，CLIP ViT-L/14）：

| 变体 | CLIPScore |
|---|---|
| Realistic Vision 单模型 | 30.86 |
| base SD1.5 单模型 | 28.37 |
| 训练出的聚合器 | 30.01 |
| Realistic Vision 放进 AFA 模型（同 VAE/文本塔，隔离聚合器影响） | **31.10** |

**聚合器比"同一框架下只用最好的单专家"低 1.08 分（12 个 prompt 里 10 个更差）**，路由统计显示它把 88% 的权重给了 base SD1.5——因为**训练 MSE 偏好 SD1.5（0.05720 < RV 0.05887），而生成质量偏好 RV（30.86 > 28.37），两者反号**。

→ 迁移后必须同时验证两条线：① 论文原始 MSE 目标能否在生成指标上复现增益；② 质量对齐目标（单步去噪 → VAE 解码 → CLIP/美学分反传），后者需要 96GB 显存才放得下。

## 4. T4 环境实测（迁移配置依据）

| 项目 | 实测 |
|---|---|
| 训练显存 | 2 专家 batch 2 ≈ 9.4GB（torch 峰值）；3 专家 batch 2 = **13.9GB**（可用 15.4GB）；5 专家直接 **CUDA OOM**（权重 10.4GB + 搬运峰值） |
| 速度 | 2 专家 batch 2：**2.03 s/步**；3 专家 eff batch 8（2×4）：**9.37 s/步** → 1 epoch（8.4 万图）27.3 h |
| xformers | 装上后 **慢 25%**（2.55 vs 2.03 s/步）且不省显存；默认 PyTorch SDPA 更快 → 训练脚本主动 disable |
| cuDNN | **坏的**（`CUDNN_STATUS_NOT_INITIALIZED`，系统 cuDNN 9.1 vs torch 自带 9.24 冲突）；默认、`LD_LIBRARY_PATH`、`LD_PRELOAD` 三种修法都无效 → 只能 `cudnn.enabled=False`（代价 1.5–2×） |
| 梯度检查点 | 3 专家 batch 4 时 8.3GB（可行）但 4.83 s/步（2.4× 慢）；UNet 支持，**Transformer2DModel 不支持**（直接 raise，需 try/except） |
| 数据 | JourneyDB 经 `wusize/journeydb`（`repo_type="dataset"`）匿名可得：`captions.zip` 488MB + `images/NNN.tar` 736MB/片（20.9k 张 512×512） |

## 5. 由此确定的迁移配置

- **专家数**：5–7 个（3 个损失 20% 的收益，2 个几乎为零）。
- **batch**：96GB 上 7 专家 batch 8（≈72GB）或 5 专家 batch 12（≈72GB），不需要梯度累积/检查点。
- **评测**：CLIPScore（`eval_generation.py`）+ LAION 美学分（`shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE`）+ 可选 FID。**不要用训练 MSE 判断成败。**
- **先筛再训**：`probe_multi_expert.py` + `analyze_multi_expert.py` 每个专家 2 分钟，**跨噪声路由收益 >2% 才值得训**。

## 6. 未验证 / 待办

- 5 专家、512px 的完整训练**在 T4 上从未跑成**（显存不足），因此 §2 的 +5.75% 只是离线估计的**上界**，需要用 96GB 卡实际训练验证。
- 论文的 6 个 Civitai 专家里，**RealCartoon-Realistic（Civitai 97744）未获取**、**Civitai 15003 未识别**。
- JourneyDB 上 5 专家的训练数据规模：论文为 1 万图 × 10 epoch；本项目数据为 8.4 万图（可再扩到 200 片 ≈ 420 万图）。
- 质量对齐目标（VAE 解码 + CLIP 反传）尚未实现。
