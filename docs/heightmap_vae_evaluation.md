# 高度图 VAE 训练效果评估标准

对应 `scripts/height_vae/train_height_vae.py` 每 epoch 生成的 `outputs/height_vae/visualizations/epoch_XXXX.png`。

---

## 效果图结构

| 行 | 子图 | 内容 |
|----|------|------|
| 1 | 1×4 曲线 | Total Loss / MSE Loss / KL Loss / Geo Loss |
| 2 左 | 2 张热力图 | 原始高度图 (terrain) / 重建高度图 (terrain) |
| 2 右 | 1 张热力图 + 1 条剖线 | 绝对误差热力图 (hot) + 中心水平剖线（蓝=原图，橙=重建） |

---

## 一、Loss 曲线解读

### 1.1 Total Loss

| 表现 | 解读 | 应对 |
|------|------|------|
| 单调下降并趋于平缓 | 正常收敛 | 继续训练 |
| 持续震荡 | 学习率过高或 batch 太小 | 降低 `LEARNING_RATE`（如 5e-5）或增大有效 batch |
| 先快速下降后停止 | 模型容量饱和 | 增大 `block_out_channels` 或 `latent_channels` |
| 突然飙升 | 梯度爆炸 | 检查数据是否有 NaN/Inf，或降低 `GRAD_CLIP` |

### 1.2 MSE Loss

| 表现 | 解读 | 应对 |
|------|------|------|
| 与 Total Loss 平行下降 | 像素级重建在改善 | 正常 |
| 下降极慢 | 模型容量不够或学习率太低 | 增大 `block_out_channels` 或 `LEARNING_RATE` |
| 波动大于 Total Loss | MSE 对极端值敏感 | 开启 `USE_HUBER_LOSS = True` |

### 1.3 KL Loss

**这是最重要的信号**，用于诊断 posterior collapse。

| 表现 | 解读 | 应对 |
|------|------|------|
| 初期 0 → 逐渐到 1e-5~1e-4 | 最理想 | 正常，隐空间被有效利用 |
| 始终 < 1e-8 | **Posterior Collapse**：解码器忽略隐变量 | 增大 `LOSS_WEIGHT_KL`（1e-6 → 1e-5）；或延长 `KL_ANNEALING_EPOCHS` |
| 突然 > 1e-2 | KL 主导训练，MSE 被压制 | 降低 `LOSS_WEIGHT_KL` |

**Posterior Collapse 的后果**：VAE 退化为普通 autoencoder，隐空间不可控，下游扩散模型将从无意义的噪声空间中生成。

**判定阈值**：
- KL < 1e-7 持续超过 10 epoch → 几乎肯定 collapse
- 1e-7 < KL < 1e-5 → 弱正则化，可接受但不太理想
- 1e-5 < KL < 1e-3 → 健康

### 1.4 Geo Loss

| 表现 | 解读 | 应对 |
|------|------|------|
| 与 MSE 同步下降，下降斜率相似 | 几何约束和像素约束捕捉同质信息 | 可适当降低 `LOSS_WEIGHT_GEO`（0.8 → 0.5） |
| 比 MSE 下降明显更快 | 陡崖等高梯度区域快速改善 | 正常，梯度约束比像素约束更易优化 |
| 几乎不动 | Sobel 核在当前数据范围中无有效信号 | 检查数据范围是否过小（如全图接近平坦） |
| 下降但重建图陡崖仍模糊 | geo loss 权重不够 | 增大 `LOSS_WEIGHT_GEO`（0.8 → 1.2） |

---

## 二、原始 vs 重建 对比热力图

### 2.1 整体评估

| 现象 | 原因 | 应对 |
|------|------|------|
| 重建图整体发灰 | 高值被压低、低值被抬高，MSE 做均值回归 | 启用 `USE_HUBER_LOSS = True` |
| 重建图整体偏暗/偏亮 | VAE bias 未校准 | 检查数据 [0,1] 映射是否一致 |
| 平地出现波纹/颗粒伪影 | 过拟合训练噪声 | 增大 `LOSS_WEIGHT_KL` 或添加 dropout |
| 山脉轮廓漂移 | 低频信息丢失（下采样过深） | 减少下采样次数或增大 `latent_channels` |

### 2.2 陡崖评估

| 现象 | 原因 | 应对 |
|------|------|------|
| 陡崖边缘模糊、变宽 | latent 通道数不足 | 增大 `latent_channels`（4 → 6 或 8） |
| 陡崖边缘出现锯齿 | 卷积 stride 导致的棋盘效应 | 检查 `UpDecoderBlock2D` 的上采样方式 |
| 陡崖两侧出现光晕/鬼影 | 过平滑效应 | 增大 `LOSS_WEIGHT_GEO` |

---

## 三、误差热力图解读

| 热点位置 | 含义 | 根因 |
|----------|------|------|
| 陡崖/山脊线 | 高频信息丢失 | latent channels 太少或下采样过多 |
| 峰顶/谷底 | 极端值回归 | MSE 对 outlier 惩罚过重 |
| 大片均匀偏高 | 全局偏移 | VAE 输出层 bias 或数据归一化不一致 |
| 图像四边 | padding 效应 | 卷积边界失真 |
| 均匀分散、无热点 | 随机噪声，训练已达理想状态 | —

---

## 四、定量标准

数据在归一化空间 [0, 1] 中（log 变换后）：

| 重建质量 | MAE (归一化空间) | 等效物理误差 (假设全幅 6000m) | 使用判断 |
|---------|-----------------|---------------------------|---------|
| 不合格 | > 0.05 | > 300m | 陡崖完全失真 |
| 及格 | 0.02 – 0.05 | 120 – 300m | 地形轮廓可见，细节丢失 |
| 良好 | 0.01 – 0.02 | 60 – 120m | 大部分细节保留 |
| 优秀 | < 0.01 | < 60m | 陡崖边缘清晰，可进入下游训练 |

**Max Error**：
- 理想 < 0.10（全幅范围内最大 600m 偏差，通常在山顶）
- 若 Max Error > 0.2 且持续多个 epoch 不降，说明存在系统性重建瓶颈

---

## 五、高程剖线解读

中心水平剖线是最直接的定量验证：

| 现象 | 解读 | 应对 |
|------|------|------|
| 峰谷位置对齐但高度被削平 | latent bottleneck | 增大 `latent_channels` |
| 平滑区域有锯齿噪声 | 过拟合 | 增大 KL 权重 |
| 整体平移（重建线高于或低于原线） | 直流分量丢失 | 检查 conv 层 bias 初始化 |
| 剖线贴合度从 epoch 0→N 逐步改善 | 模型在学习 | 正常 |

---

## 六、逐 epoch 里程碑检查

| 阶段 | 预期现象 |
|------|---------|
| Epoch 0-5 | Total Loss 快速下降；重建图模糊，剖线完全不贴合；KL 趋近 0 |
| Epoch 5-20 | MSE 下降放缓；KL 开始上升（若 annealing 结束）；重建图出现大致轮廓 |
| Epoch 20-50 | 剖线峰谷位置对齐；误差热力图热点缩小；陡崖轮廓可见 |
| Epoch 50-100 | Total Loss 趋于平稳；MSE 和 Geo 基本收敛；误差热力图仅剩山顶热点 |

---

## 七、调参优先级

1. **KL Loss 过小** → 调 `LOSS_WEIGHT_KL`
2. **陡崖模糊** → 调 `LOSS_WEIGHT_GEO` 或 `latent_channels`
3. **整体重建偏灰/偏亮** → 开 `USE_HUBER_LOSS`
4. **MSE 不降** → 调 `LEARNING_RATE` 或 `block_out_channels`
5. **显存不够** → 降 `BATCH_SIZE`、调 `GRADIENT_ACCUMULATION_STEPS`、开 `ENABLE_GRAD_CHECKPOINTING`

**核心原则**：每次只改一个参数，训练 20 epoch 后对照效果图判断方向。
