# VQ-Map: 基于矢量量化的离散空间鸟瞰图地图布局估计

**出处会议：** NeurIPS 2024  
**是否开源：** https://github.com/Z1zyw/VQ-Map  
**关键词：** BEV感知、语义地图构建、离散表示(VQ-VAE)、Deformable Attention、跨模态对齐(PV-BEV)

---

## 1. 概述

传统的 BEV 地图估计通常依赖复杂的深度估计（如 LSS）来构建密集的 3D 空间特征，随后进行连续值的像素级分割。但由于遮挡和深度不准确，生成的地图往往存在严重的伪影（断流、不符合物理常识）。

**VQ-Map 的核心思想：**将“连续密集的 BEV 特征回归问题”转化为“稀疏离散的 Token 分类问题”。它利用类似 VQ-VAE 的生成模型提取 BEV 语义先验（构建一部“地图字典”），随后仅通过 2D 图像特征去查询这本字典，从而在不依赖显式深度估计的情况，生成高质量、连贯的 BEV 地图。

---

## 2. 方法

整体架构分为完全解耦的两个阶段：**阶段一（离散表示学习）** 和 **阶段二（PV-BEV 对齐与 Token 预测）**。

### 2.1 第一阶段：离散表示学习 (Discrete Representation Learning)

该阶段在真实的 BEV 语义标签（GT）上进行自编码训练，目的是构建一个离散的隐空间（Codebook）。

1. **BEV Patch Embedding ($\mathcal{E}$)**：
   将 $H \times W$ 的 BEV GT 图划分为 $P \times P$（如 $8 \times 8$）的不重叠 Patch，总计 $N$ 个。通过浅层 CNN 将每个 Patch 编码为连续向量 $\mathbf{z}_c^i \in \mathbb{R}^D$。

2. **向量量化 (Vector Quantization, $\mathcal{Q}$)**：
   定义码本 $\mathbf{V} \in \mathbb{R}^{K \times D}$（包含 $K$ 个聚类中心）。对于每个 $\mathbf{z}_c^i$，在码本中寻找 L2 距离最近的向量 $\mathbf{v}_k$ 作为量化后的特征 $\mathbf{z}_q^i$。这里的索引 $k_q$ 即为 **BEV Token**。

3. **生成解码器 (Decoder, $\mathcal{D}$)**：
   利用量化后的 $\mathbf{z}_q$ 序列重建原始 BEV 地图。

$$
\mathbf{M}_{c}' = \mathcal{D}(\mathcal{Q}(\mathcal{E}(\mathbf{M}_{c})))
$$



第一阶段的总损失为重建损失与量化损失之和： $\mathcal{L} = \mathcal{L}_{re} + \mathcal{L}_{vq}$

**① 重建损失 $\mathcal{L}_{re}$ (Class-specific weighted MSE)：**
$$
\mathcal{L}_{re} = \frac{1}{C} \sum_{c=1}^{C} \frac{\|\mathbf{M}_{c} - \mathbf{M}_{c}'\|_{2}^{2}}{1 + \|\mathbf{M}_{c}\|_{1}}
$$
*   **解释**：由于地图类别极度不平衡（例如可行驶区域占比极大，而停止线极小），这里采用了**类别自适应加权**的均方误差。分母 $1 + \|\mathbf{M}_{c}\|_{1}$ 代表该类别在 GT 中的像素总数，**像素越少的类别，惩罚权重越大**，强迫模型记住细粒度的道路拓扑。

**② 向量量化损失 $\mathcal{L}_{vq}$ (Vector Quantization Loss)：**
$$
\mathcal{L}_{vq} = \frac{1}{N} \sum_{i=1}^{N} \left( \underbrace{\left\| \mathbf{z}_q^i - \operatorname{sg}(\ell_2(\mathbf{z}_c^i)) \right\|_2^2}_{\text{拉近码本}} + \underbrace{\left\| \operatorname{sg}(\mathbf{z}_q^i) - \ell_2(\mathbf{z}_c^i) \right\|_2^2}_{\text{Commitment Loss}} + \underbrace{\sum_{j=1}^{N_{\text{aug}}} \left\| \operatorname{sg}(\mathbf{z}_q^i) - \ell_2(\tilde{\mathbf{z}}_c^{i,j}) \right\|_2^2}_{\text{一致性增强损失}} \right)
$$
*   **解释**：
    *   **$\operatorname{sg}(\cdot)$**：Stop-gradient（停止梯度）操作。
    *   **第一项**：强迫选中的码本向量 $\mathbf{z}_q$ 向连续特征 $\mathbf{z}_c$ 靠近（实际训练中作者也配合使用了 EMA 指数移动平均来平滑更新码本）。
    *   **第二项**：Commitment Loss（承诺损失），强迫特征提取器输出的 $\mathbf{z}_c$ 尽量靠近码本空间，防止输出特征到处乱跑。
    *   **第三项（本文微创新）**：引入了 $N_{\text{aug}}$ 次 Patch 级别的空间数据增强（如小尺度旋转、平移）。要求增强后的连续特征 $\tilde{\mathbf{z}}_c$ 依然量化到同一个 $\mathbf{z}_q$ 上，这极大地增强了聚类中心（Codebook）对几何形变的鲁棒性。

### 2.2 第二阶段：PV-BEV 对齐 (PV-BEV Alignment)

该阶段的核心任务是：基于相机透视图（PV）提取的特征，预测当前 BEV 空间每个 Patch 对应的 Token 索引是几号。

![Figure4](D:\Research\Notes\assets\VQ-Map\Figure4.png)

1. **Token Query 初始化**：
   初始化 $N$ 个可学习的 Embedding $\mathbf{E}_q \in \mathbb{R}^{N \times D}$，作为侦探（Query），每个 Query 负责预测一个固定的 BEV 空间网格。

2. **局部自注意力 (Local Self-Attention)**：
   在 $5 \times 5$ 的局部邻域内计算 Query 之间的自注意力，利用空间先验平滑特征。

3. **可变形交叉注意力 (Deformable Cross-Attention)**：
   利用相机内外参，将 Query 对应的 3D 锚点投影到多视角的 2D 图像特征图上。通过 Deformable Attention 稀疏地采样图像特征并更新 Query。这就是“跨模态对齐”的核心。

4. **Token 分类预测**：
   经过多层 Transformer Decoder 后，张量被 Reshape 回 2D 网格，通过 MLP 映射到长度为 $K$ 的分类 Logits。
   **监督信号**：使用 Focal Loss，将阶段一提取出的真实 Token 索引作为标签，进行**多分类任务**。

---

## 3. 推理过程深度解析 (Inference)

在推理时，模型巧妙地应用了 **Soft Token** 技术来实现“无损解码”。

1. **预测概率**：Token Decoder 预测出网格中每个位置属于 $K$ 种 Token 的概率分布矩阵 $\mathbf{P} \in \mathbb{R}^{N \times K}$。
2. **软查表 (Soft Lookup)**：不使用 `argmax` 截断（避免梯度和特征的硬性丢失），而是将概率作为权重，与冻结的码本 $\mathbf{V} \in \mathbb{R}^{K \times D}$ 进行加权求和（张量乘法）：
   $$
   \mathbf{Z}_{soft} = \mathbf{P} \times \mathbf{V}
   $$
3. **图像生成**：将连续特征 $\mathbf{Z}_{soft}$ 喂入冻结的生成解码器 $\mathcal{D}$，单次前向传播即可输出高质量、高分辨率的 BEV 地图。

---

## 4. 实验结果与深度分析

### 4.1 核心指标对比 (State-of-the-Art Comparison)

在 nuScenes 验证集上（环视，Surround-View）：

| **方法**          | **Drivable** | **Ped. Cross.** | **Walkway** | **Stopline** | **Mean (mIoU)** |
| ----------------- | ------------ | --------------- | ----------- | ------------ | --------------- |
| BEVFusion (基线)  | 81.7         | 54.8            | 58.4        | 47.4         | 56.6            |
| MapPrior (VQ-GAN) | 81.7         | 54.6            | 58.3        | 46.7         | 56.7            |
| DDP (Diffusion)   | 83.6         | 58.3            | 61.6        | 52.4         | 59.4            |
| **VQ-Map (Ours)** | **83.8**     | **60.9**        | **64.2**    | **57.7**     | **62.2**        |

**分析**：VQ-Map 在复杂、细��度的类别（如 Stopline 停止线涨幅 +10.3、Walkway 人行道涨幅 +5.8）上提升巨大。这些类别极度依赖结构先验，而 VQ 字典恰好弥补了图像遮挡带来的物理信息缺失。

### 4.2 零样本跨数据集泛化 (Argoverse Monocular)

*   **设定**：作者在 Argoverse 数据集做单目地图估计测试时，**直接使用了在 nuScenes 上训练好的第一阶段 Codebook 和 Decoder**。
*   **结果**：取得了 **73.4** 的 IoU（前SOTA为 68.3，提升 +5.1）。
*   **分析**：这证明了 VQ-Map 学习到的离散码本，本质上抓住了物理世界中“道路拓扑结构”的通用流形（Manifold），而非对特定数据集相机视角的过拟合。

### 4.3 消融实验：为什么用离散分类代替连续回归？

作者对比了 Token Decoder 使用不同监督信号的效果：

| **监督信号**                | **损失函数**   | **mIoU** | **评价**                                                     |
| :-------------------------- | :------------- | :------- | :----------------------------------------------------------- |
| 回归连续向量 $\mathbf{z}_c$ | MSE            | 60.3     | 空间维度直接算 L2 距离，梯度极易受图像遮挡噪声影响           |
| 回归量化向量 $\mathbf{z}_q$ | MSE            | 60.1     | 同上                                                         |
| **预测离散 Token 索引**     | **Focal Loss** | **61.8** | **将不确定的 3D 回归降维打击为有明确类别的多分类任务，抗噪性极强** |

### 4.4 计算开销与性能权衡 (Computational Overhead)

相比于同样引入生成先验的算法，VQ-Map 的前向推理极其高效：
*   **DDP (3步去噪)**：mIoU 59.4，MACs = 614.1G。
*   **VQ-Map (Tiny)**：mIoU 59.6，MACs = **86.8G**。
*   **VQ-Map (Standard)**：mIoU 62.2，MACs = 231.6G。
**分析**：无需扩散模型的自回归和反复迭代去噪。VQ-Map 是“一次前向传播 (One-pass)”，通过一次概率预测 + CNN 上采样即可出图，兼顾了生成质量与推理速度。

---

## 5. 总结笔记

1. **核心创新**：用“分类”降维打击“回归”。将连续的、受遮挡和深度误差影响严重的 3D 特征映射，转化为在固定码本上的离散 Token 分类任务。
2. **结构优势**：抛弃了 LSS 那种全图生成庞大密集的 3D Frustum 的方式，利用 Deformable Attention 在 2D 图像上进行稀疏采样，极大降低了计算量。
3. **物理意义**：学到的 Codebook 本质上是一本“道路常识字典”，所以 VQ-Map 修复伪影（如自动脑补被前车挡住的车道线、保证十字路口连贯）的能力远超纯判别式模型（如 BEVFusion）。