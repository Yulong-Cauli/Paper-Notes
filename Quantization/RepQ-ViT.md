# RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers

**出处会议：** ICCV 2023  
**是否开源：** https://github.com/zkkli/RepQ-ViT  
**关键词：** 视觉Transformer (ViT)，训练后量化 (PTQ)，尺度重参数化 (Scale Reparameterization)，极低比特量化 (4-bit)

---

## 1. 概述

本文直击 Vision Transformer (ViT) 在极低比特（如 4-bit）量化时精度崩溃的痛点。

ViT 中存在两个具有“极端分布”的激活值：**LayerNorm 后的剧烈通道间波动** 和 **Softmax 后的极端幂律分布**。

传统 PTQ 方法受限于硬件部署要求，在量化阶段就强行使用简单的硬件友好量化器（如逐层量化、$\log_2$ 量化），导致严重的量化误差。

RepQ-ViT 提出了一种全新的**“量化-推理解耦范式”**：在量化阶段使用**复杂的量化器**（保精度），在推理阶段通过**尺度重参数化（Scale Reparameterization）**将其无缝转换为**简单的量化器**（保速度）。该方法无需超参数调节和昂贵的重建过程，在 4-bit 下首次将 ViT 的精度提升到了可用水平。

<div align="center"><img src="https://raw.githubusercontent.com/Yulong-Cauli/Paper-Notes/main/assets/RepQ-ViT/Figure1.jpeg" alt="Overview of RepQ-ViT" width="80%"></div>
*(注：对应论文 Figure 1，展示了量化过程与推理过程的解耦及重参数化桥梁)*

---

## 2. 方法：两大尺度重参数化魔法

RepQ-ViT 的核心在于用严谨的数学等价/近似变换，将复杂的量化参数“揉”进网络的前后层结构中。

### 2.1 针对 LayerNorm 的重参数化：逐通道 $\rightarrow$ 逐层

**痛点：** LayerNorm 后的激活值在不同通道间差异巨大，必须用**逐通道量化 (Channel-wise)** 才能保精度，但这在硬件激活运算上不被支持（硬件仅支持**逐层量化 Layer-wise**）。

**解决思路：**
假设原逐通道的比例尺为 $s$，零点为 $z$。目标是转换为统一的逐层参数 $\tilde{s}$ 和 $\tilde{z}$（如取平均值）。
定义变异因子：倍数关系 $r_1 = s/\tilde{s}$，差值关系 $r_2 = z - \tilde{z}$。

1. **构造等效的伪激活值**：
   为了让后续量化器能用统一的 $\tilde{s}, \tilde{z}$，我们需要把输入 $X'$ 偷偷替换为 $\widetilde{X}'$：
   $$
   \widetilde{X}' = \frac{X' + s \odot r_2}{r_1}
   $$
   
2. **向前吸收（修改 LayerNorm 参数）**：
   将上述变换融合进前置的 LayerNorm 的仿射参数 $\gamma, \beta$ 中，得到新的部署参数：
   $$
   \widetilde{\gamma} = \frac{\gamma}{r_1}, \quad \widetilde{\beta} = \frac{\beta + s \odot r_2}{r_1}
   $$

3. **向后补偿（修改下一层 Linear 的权重）**：
   由于输入给下一层的内容变了，为了保证最终输出不变，必须修改下一层（QKV投影层）的权重 $W^{qkv}$ 和偏置 $b^{qkv}$：
   $$
   \widetilde{W}^{qkv} = r_1 \odot W^{qkv}, \quad \widetilde{b}^{qkv} = b^{qkv} - (s \odot r_2) W^{qkv}
   $$
   *(注：由于这里用 $r_1$ 放缩了权重 $W$，改变了其分布，因此 $\widetilde{W}$ 需要在极少量校准数据上重新校准一下量化参数，这会带来极微小的精度损失，但换来了硬件的完美支持。)*

---

### 2.2 针对 Softmax 的重参数化：$\log_{\sqrt{2}}$ $\rightarrow$ $\log_2$

**痛点：** Attention 矩阵呈幂律分布，大部分值接近0，极少数大值非常关键。硬件友好的 $\log_2$ 量化器分辨率不足，会将大量关键分数粗暴截断。底数为 $\sqrt{2}$ 的量化器精度高，但**不支持硬件移位加速（Bit-shifting）**。

**解决思路：**
利用对数与指数的数学性质，将 $\log_{\sqrt{2}}$ 强行拆解为纯整数的 $\log_2$ 和位移操作。

1. **量化阶段（换底公式）**：
   $$
   A^{(\mathbb{Z})} = \text{clip}\left(\left\lfloor -\log_{\sqrt{2}} \frac{A}{s} \right\rceil, \dots\right) = \text{clip}\left(\left\lfloor -2\log_2 \frac{A}{s} \right\rceil, \dots\right)
   $$
   量化时只需��乘以常数 2 即可转换，极其简单。

2. **反量化阶段（奇偶性拆解）**：
   原本的反量化公式为 $\hat{A} = s \cdot 2^{-\frac{A^{(\mathbb{Z})}}{2}}$。
   此时指数 $-\frac{A^{(\mathbb{Z})}}{2}$ 可能不是整数，无法使用硬件移位。作者引入向下取整和奇偶指示函数 $\mathbb{1}(\cdot)$（偶数为0，奇数为1）：
   $$
   \hat{A} = s \cdot 2^{\lfloor -A^{(\mathbb{Z})}/2 \rfloor} \cdot \left[ \mathbb{1}(A^{(\mathbb{Z})}) \cdot (\sqrt{2} - 1) + 1 \right]
   $$
   将后面那一坨多出来的常数直接吸收到全新的比例尺 $\tilde{s}$ 中：
   $$
   \tilde{s} = s \cdot \left[ \mathbb{1}(A^{(\mathbb{Z})}) \cdot (\sqrt{2} - 1) + 1 \right]
   $$
   最终反量化变成 $\hat{A} = \tilde{s} \cdot 2^{\text{整数}}$，**完美支持无分支（Branchless）的纯移位硬件加速，且数学上 100% 等价，一分精度不掉！**

---

## 3. 实验结果与深度分析

### 3.1 极限 4-bit 图像分类 (ImageNet)

在 W4/A4 的极限设置下，此前的方法全面崩盘，RepQ-ViT 实现了“起死回生”。

| **方法 (W4/A4)**    | **No HP** | **No REC** | **ViT-B** | **DeiT-S** | **Swin-S** |
| :------------------ | :-------: | :--------: | :-------: | :--------: | :--------: |
| FP32 (基准)         |     -     |     -      |   84.54   |   79.85    |   83.23    |
| FQ-ViT              |     ×     |     ✓      |   0.10    |    0.10    |    0.10    |
| PTQ4ViT             |     ×     |     ×      |   30.69   |   34.08    |   76.09    |
| APQ-ViT             |     ×     |     ×      |   41.41   |   43.55    |   77.15    |
| **RepQ-ViT (Ours)** |   **✓**   |   **✓**    | **68.48** | **69.03**  | **79.45**  |

- **分析**：在无超参 (No HP) 且无昂贵重建 (No REC) 的纯 PTQ 设定下，RepQ-ViT 在 DeiT-S 上将准确率从 43.55% 暴涨至 69.03% (+25.48%)，首次将 ViT 的 4-bit 量化推向可用水平。

### 3.2 下游任务：目标检测与实例分割 (COCO)

基于 Mask R-CNN 框架验证模型泛化性。

| **模型 (W4/A4)** | **骨干网络** | **方法**     | **APbox** | **APmask** |
| :--------------- | :----------- | :----------- | :-------- | :--------- |
| Mask R-CNN       | Swin-T       | FP32         | 46.0      | 41.6       |
|                  |              | APQ-ViT      | 23.7      | 22.6       |
|                  |              | **RepQ-ViT** | **36.1**  | **36.0**   |

- **分析**：下游高维视觉任务结构复杂，对量化误差极其敏感。APQ-ViT 在 Swin-T 上出现了严重的水土不服，而 RepQ-ViT 依然稳健，APbox 领先近 12.4 个点。在 6-bit 下更是几乎无损。

### 3.3 核心消融实验 (Ablation Studies)

验证重参数化的实际收益（以 DeiT-S 为例）。

**LayerNorm 激活量化消融：**
*   纯逐层量化 (Layer-wise)：33.17%（崩塌）
*   纯逐通道量化 (Channel-wise)：70.28%（精度高，但硬件不支持）
*   **Scale Reparam (Ours)**：**69.03%**（硬件完美支持，精度仅因权重重新校准损失 1.25%）

**Softmax 激活量化消融：**
*   普通的 $\log_2$ 量化：67.71%
*   完美的 $\log_{\sqrt{2}}$ 量化：69.03%
*   **Scale Reparam (Ours)**：**69.03%**（100% 数学等价，精度无损转移到硬件位移运算）

### 3.4 部署与校准效率 (Efficiency)

| **方法**     | **校准数据量** | **耗时 (单卡 3090)** |
| :----------- | :------------- | :------------------- |
| FQ-ViT       | 1000 张        | 0.5 分钟             |
| PTQ4ViT      | 32 张          | 3.2 分钟             |
| **RepQ-ViT** | **32 张**      | **1.3 分钟**         |

- **分析**：因为没有任何梯度重建和复杂的超参搜索，RepQ-ViT 只需要 32 张校准图，在 1 分钟出头即可完成一整个 ViT 模型的量化，极其契合工业界快速落地的需求。

---

## 4. 总结与启示

1. **打破思维定式**：RepQ-ViT 证明了“硬件友好的量化器”不需要在量化第一天就强加给模型。通过**解耦思想**，我们可以先用最贴合数据分布的复杂量化器保住精度，再用纯数学的**重参数化 (Reparameterization)** 巧妙转换为底层硬件支持的格式。
2. **算力转移的艺术**：针对 LayerNorm 的逐通道误差，论文巧妙地利用网络中相邻算子的线性关系，将局部的缩放偏差“吸收”并“转移”到了前后层的权重参数里，这种方法极具通用性。
3. **彻底落地的价值**：无超参数、无重建网络、仅需 32 张图、耗时 1 分钟。这是一个在工程界直接就能写入量化编译器（如 TensorRT / OpenVINO 工具链）的极简 SOTA 方案。