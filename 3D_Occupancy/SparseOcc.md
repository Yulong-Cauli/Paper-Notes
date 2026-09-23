# SparseOcc: Rethinking Sparse Latent Representation for Vision-Based Semantic Occupancy Prediction

**出处会议：** CVPR 2024 
**是否开源：** https://github.com/VISION-SJTU/SparseOcc 
**关键词：** 语义占用预测、稀疏潜变量表示、稀疏卷积、场景补全

------

## 1. 概述

传统的基于视觉的 3D 感知方法通常在稠密潜空间（Dense Latent Space）上操作，但这会引入立方级（Cubic）的时空复杂度 $O(N^3)$，限制了感知范围和空间分辨率的扩展。现有的压缩方法（如 BEV 或 TPV）虽高效但会造成严重的几何信息丢失。

**SparseOcc** 提出了一种**无损的稀疏潜变量表示（Lossless Sparse Latent Representation）**，通过纯稀疏算子实现高效的 3D 语义占用预测。其核心优势在于：

- **计算效率**：相比稠密基线，FLOPs 减少了 74.9%，内存占用减少约 31.6%~40.9%。
- **精度提升**：在 nuScenes-Occupancy 验证集上，mIoU 从 12.8% 提升至 14.1%，有效避免了空体素上的“幻觉（Hallucinations）”。

------

## 2. 方法

### 2.1 稀疏潜变量表示 (Sparse Latent Representation)

利用 Lift-Splat-Shoot (LSS) 将 2D 特征映射到 3D 空间后，由于射线投射的特性，约 80% 的体素是空的。SparseOcc 仅保留非空体素，并使用**坐标格式（COO）**存储稀疏张量：
$$
V = \{(p_i = [x_i, y_i, z_i] \in \mathbb{R}^3, f_i \in \mathbb{R}^C) | i = 1, 2, ... N\}
$$
其中 $N$ 是非空体素的数量，$p_i$ 和 $f_i$ 分别表示第 $i$ 个体素的坐标和特征。后续所有操作均在稀疏表示上进行。

### 2.2 稀疏潜层扩散器 (Sparse Latent Diffuser)

为了解决初始 3D 特征仅覆盖物体表面（可见部分）的问题，扩散器负责将非空特征传播到相邻的空区域以完成场景补全。

- **稀疏补全块 (Sparse Completion Block)**：使用 3D 稀疏卷积。为平衡补全效果与稀疏度，模型仅在必要时执行扩散。
- **上下文聚合块 (Contextual Aggregation Block)**：采用子流形稀疏卷积（Submanifold Convolution），确保输出位置仅在输入位置非空时激活，从而在加深网络提取语义的同时严格保持稀疏度。
- **内核分解 (Kernel Decomposition)**：为了利用驾驶场景的几何分布（如扁平的路面或垂直的建筑），将 $k \times k \times k$ 内核分解为三个正交的 $k \times k \times 1$、$k \times 1 \times k$ 和 $1 \times k \times k$ 内核。
  - **优势**：复杂度从 $\mathcal{O}(k^3)$ 降至 $\mathcal{O}(3k^2)$ 或 $\mathcal{O}(4k^2)$。

### 2.3 稀疏特征金字塔 (Sparse Feature Pyramid)

模型通过堆叠 $L=4$ 层扩散器并配合步长为 2 的下采样（Down-sampling）构建多尺度表示。

**稀疏体素解码器 (Sparse Voxel Decoder)**： 放弃高开销的可变形注意力（MSDeformAttn），采用轻量级的**稀疏插值与求和**机制进行多尺度融合：
$$
\hat{\mathbb{V}}_l = \sum_{j \ne l} W_j \cdot Interp(\mathbb{V}_j, \mathbb{V}_l)
$$

- **$W_j$**：第 $j$ 个尺度的学习权重。
- **$Interp$**：稀疏线性插值。利用低分辨率特征的稠密性来辅助高分辨率特征的补全。

### 2.4 稀疏 Transformer Head

------

#### 输入阶段：Queries 与 初始过滤

在框图最下方，输入是 **Queries**（$N_q$ 个向量）和从 **Sparse Feature Pyramid** 传来的多尺度特征 $\hat{\mathbb{V}}_l$。

- **算子逻辑**：在正式进入 Transformer 块之前，模型对 $\hat{\mathbb{V}}_l$ 进行了一个线性二分类（图中未细标，但在论文 3.4 节提到）。
- **公式表现**：通过分类器筛选出 $N_l$ 个预测为非空的体素特征 $f_i$。为了处理被过滤掉的空域，引入一个可学习的空值 Token $p_{\phi}$。
- **意义**：这确立了计算的边界，后续所有的矩阵运算只针对这 $N_l + 1$ 个点进行。

### 2. Masked Attention（掩码注意力层）

这是框图中最底部的计算块，也是最核心的稀疏约束层。注意到有一条标着 **Mask** 的线从顶部反馈回来。

- **数学公式**：

  $$Q_l = \text{softmax}[\mathcal{M}_{l-1} + Q_{l-1}W_q(K_l)^T]V_l + Q_{l-1}$$

  其中 $K_l$ 和 $V_l$ 分别是特征 $\hat{\mathbb{V}}_l$ 经过线性变换 $W_k, W_v$ 得到的。

- **掩码算子 $\mathcal{M}_{l-1}$**：

  $$\mathcal{M}_{l-1}(x, y, z) = \begin{cases} 0 & \text{if } \sigma(M'_{l-1}(x, y, z)) \geq 0.5 \\ -\infty & \text{otherwise} \end{cases}$$

- **流程解析**：这一层通过上一层生成的掩码 $M_{l-1}$ 来限制注意力的空间。如果位置 $(x,y,z)$ 被预测为障碍物的概率低于 0.5，权重直接设为 $-\infty$。在数学上，这迫使 Query 在计算交叉注意力时，**物理上忽略掉所有非目标区域**。

### 3. Self-Attention（自注意力层）

位于 Masked Attention 之上，通过 **Add & Norm** 连接。

- **算子逻辑**：执行标准的多头自注意力（Multi-head Self-Attention）。
- **公式表现**：$Q = \text{Attention}(Q, Q, Q)$。
- **意义**：由于每一个 Query 最终要负责预测一个特定的 3D 掩码，自注意力机制让不同的 Query 之间进行信息交换，防止多个 Query 预测同一个物体（重复预测），实现类似于非极大值抑制（NMS）的效果。

### 4. FPN 与多尺度循环

框图中部有一个 **FPN** 块。

- **流程解析**：Sparse Transformer Head 实际上是一个迭代过程。它会依次遍历特征金字塔的不同尺度（从低分辨率到高分辨率）。
- **算子逻辑**：在每一层更新 $Q$ 后，利用更新后的 $Q$ 去查询金字塔中下一尺度更精细的特征 $\hat{\mathbb{V}}_{l+1}$。这对应了图中 FPN 与 Add & Norm 的循环连接。

### 5. 最终预测：掩码解码（图中顶部的 $\otimes$ 算子）

在框图最顶端，经过多层迭代的 $Q$ 与特征进行点积，输出最终结果。

- **算子逻辑**：这是一个点积（Dot Product）解码过程。

- **数学公式**：

  $$M = Q \cdot \{p_{\phi} \cup f_i\}^T$$

  其中 $M \in \mathbb{R}^{N_q \times (N_l+1)}$。

- **结果解析**：

  1. **掩码重建**：通过点积得到 $N_q$ 个稀疏掩码。
  2. **分类**：$Q$ 同时进入一个线性分类器层（图中未画出，但属于 Head 的一部分），预测该掩码所属的语义类别（如：汽车、卡车等）。
  3. **Scatter 操作**：利用存储的坐标索引 $p_i$，将稀疏的 $M$ 填充回标准的 $H \times W \times D$ 体素网格中。

------

### 总结：公式如何体现“Sparse”？

在整个流程中，**稀疏性（Sparsity）**体现在矩阵乘法的维度上：

1. **注意力计算**：计算复杂度由全空间的 $O(N_q \cdot HWD)$ 变成了 $O(N_q \cdot N_l)$。
2. **掩码预测**：解码复杂度由 $O(N_q \cdot HWD \cdot C)$ 变成了 $O(N_q \cdot N_l \cdot C)$。

通过公式 $\mathcal{M}_{l-1}$ 的 $-\infty$ 约束，模型在数学上剔除了所有无关体素，使得整个 Transformer Head 能够以极快的速度在 3D 空间内完成迭代优化。

**你会发现这本质上是一个“由面及点”的过程，所有算子都在 $N_l$ 这个点集上运行。关于损失函数中如何利用匈牙利匹配（Hungarian Matching）来训练这套流程，需要细讲吗？**

------

## 3. 损失函数 (Objective Function)

SparseOcc 采用多任务联合监督训练：
$$
\mathcal{L} = \mathcal{L}_{mask} + \mathcal{L}_{cls} + \mathcal{L}_{depth} + \mathcal{L}_{seg}
$$

- **$\mathcal{L}_{mask}$ & $\mathcal{L}_{cls}$**：基于匈牙利匹配（Hungarian Matcher）的掩码损失和类别损失。
- **$\mathcal{L}_{depth}$**：对 LSS 生成的深度图进行监督。
- **$\mathcal{L}_{seg}$**：对稀疏化过程中的粗略二分类进行监督，确保有效过滤空体素。

------

## 4. 实验结果与分析

### 4.1 nuScenes-Occupancy 性能对比 (Table 1)

| **方法**             | **输入**   | **IoU**  | **mIoU** | **FLOPs** | **显存** | **3D 延迟** |
| -------------------- | ---------- | -------- | -------- | --------- | -------- | ----------- |
| TPVFormer            | Camera     | 15.3     | 7.8      | 1132G     | 20G      | 0.57s       |
| OpenOccupancy        | Camera     | 19.3     | 10.3     | 1716G     | 19G      | 0.84s       |
| C-CONet              | Camera     | 20.1     | 12.8     | 1810G     | 21G      | 2.18s       |
| **SparseOcc (Ours)** | **Camera** | **21.8** | **14.1** | **455G**  | **13G**  | **0.19s**   |

- **结论**：SparseOcc 在精度大幅领先的同时，计算开销仅为稠密基线的约 1/4。

### 4.2 消融实验：补全块设计 (Table 3)

| **卷积类型** | **数量** | **内核大小**          | **IoU**  | **mIoU** |
| ------------ | -------- | --------------------- | -------- | -------- |
| 无补全       | 0        | -                     | 35.5     | 12.1     |
| 普通卷积     | 1        | $3 \times 3 \times 3$ | 35.8     | 12.2     |
| **分解卷积** | **1**    | **分解 3D 核**        | **36.5** | **13.1** |

- **分析**：分解后的正交卷积内核比传统的 full kernel 具有更强的表达能力，且能更有效地利用驾驶场景的几何形状先验。

------

## 5. 总结

1. **架构革命**：彻底放弃稠密/投影表示，证明了**纯稀疏算子**可以完整支撑 3D 语义占用任务。
2. **时空解耦**：利用内核分解和稀疏金字塔，在保持大感受野的同时极大地抑制了显存和算力增长。
3. **容错性**：通过稀疏 Transformer Head 和掩码预测，模型能够更好地处理遮挡和稀疏观测，显著减少了背景噪声引发的误检。