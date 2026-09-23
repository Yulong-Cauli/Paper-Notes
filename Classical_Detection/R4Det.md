# R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection

**出处会议：** CVPR 2026

**是否开源：** 否

**关键词：** 4D雷达-相机融合、全景深度融合、可变形门控时序融合、动态精炼

------

## 1. 概述

解决 4D 雷达-相机融合中深度估计不准、缺乏自车位姿（Ego-pose）时的时序融合失效，以及稀疏雷达点云下小目标检测能力差的问题。

通过引入全景深度融合（PDF）、解耦的时空融合架构（DGTF）以及基于 2D 先验的精炼机制（IGDR），实现端到端的高性能 3D 目标检测。

$$\text{Radar/Image Inputs} \xrightarrow{\text{PDF}} \text{Aligned BEV Features} \xrightarrow{\text{DGTF}} \text{Temporal BEV} \xrightarrow{\text{IGDR}} \text{Refined BEV} \xrightarrow{\text{Detection}} \text{3D Bounding Boxes}$$

------

## 2. 方法

### 2.1 全景深度融合模块 (Panoramic Depth Fusion, PDF)

该模块解决雷达点云稀疏性与图像特征密集性的维度不匹配问题，并引入三重深度监督机制。

其pipeline依然是**邻域交叉注意力 (NCA)**：放弃全局注意力，在投影点周围的局部窗口内计算注意力，减少远距离噪声干扰。

$$
Attention(Q, K, V) = Softmax(\frac{Q K_k^T}{\sqrt{d}}) V_k
$$

创新点在于**三重深度监督**：

首先是**概率监督 ($\mathcal{L}_{prob}$)**：将稀疏真值 $d_g^{sparse}$ 转化为高斯分布 $\mathcal{G}(d_g^{sparse})$，通过 KL 散度约束预测分布 $\mathcal{P}_i$，抑制几何散射。

$$
\mathcal{L}_{prob}=\frac{1}{|\mathcal{M}_{sparse}|}\sum_{i\in\mathcal{M}_{sparse}}KL(\mathcal{G}(d_{g_i}^{sparse})||\mathcal{P}_i)
$$
然后是**基础模型引导监督 ($\mathcal{L}_{found}$)**：由稀疏雷达锚定深度损失（$\mathcal{L}_{abs}$）和密集伪标签深度损失（$\mathcal{L}_{dense}$）加权组成。核心算子为 Smooth L1 损失。公式如下：

$$
\mathcal{L}_{found} = \lambda_{abs}\mathcal{L}_{abs} + \lambda_{dense}\mathcal{L}_{dense}
$$
各项的具体计算公式与物理定义为：

仅在存在雷达点云投影的稀疏像素掩码集合 $\mathcal{M}_{sparse}$ 上，计算预测深度 $\hat{d}$ 与雷达真实绝对测量深度 $d_{g}^{sparse}$ 之间的偏差：

$$
\mathcal{L}_{abs} = \frac{1}{|\mathcal{M}_{sparse}|} \sum_{i \in \mathcal{M}_{sparse}} \text{Smooth}_{L1}(\hat{d}_i, d_{g_i}^{sparse})
$$
在全图所有像素集合 $\mathcal{M}_{all}$ 上，计算预测深度 $\hat{d}$ 与视觉基础模型（如 Metric3D）生成的密集伪真值深度 $d_{g}^{dense}$ 之间的偏差：

$$
\mathcal{L}_{dense} = \frac{1}{|\mathcal{M}_{all}|} \sum_{i \in \mathcal{M}_{all}} \text{Smooth}_{L1}(\hat{d}_i, d_{g_i}^{dense})
$$


**Smooth L1 算子定义**

$$\text{Smooth}_{L1}(x, y) = \begin{cases} 0.5(x - y)^2, & \text{if } |x - y| < \beta \\ |x - y| - 0.5\beta, & \text{otherwise} \end{cases}$$

*(注：深度回归任务中 $\beta$ 常设为 1)*

**结构化排序监督 ($\mathcal{L}_{relative}$)**：采用带有动态容差 $\tau_{ij}$ 的边缘偏置采样，约束像素间的相对几何关系，强化边缘阶跃特征。

$$\mathcal{L}_{pair}(i,j)=Softplus(-sign(d_{g_i}^{dense} - d_{g_j}^{dense})(\hat{d}_i - \hat{d}_j))$$

**总深度损失**：$\mathcal{L}_{depth}=\lambda_{1}\mathcal{L}_{prob}+\lambda_{2}\mathcal{L}_{found}+\lambda_{3}\mathcal{L}_{relative}$

### 1. 三重监督的实验支撑

论文通过 Table 5 的消融实验验证了三重监督设计的正交性和必要性。每引入一种监督机制，BEV mAP 均呈现严格的单调增长：

- **基线配置 (仅 $\mathcal{L}_{prob} + \mathcal{L}_{abs}$)**：BEV mAP 为 45.15%。此时网络仅依赖稀疏雷达点提供绝对尺度和概率形状约束。
- **引入基础模型密集监督 ($+\mathcal{L}_{dense}$)**：BEV mAP 提升至 46.08%（增幅 +0.93%）。证明密集伪标签有效填补了雷达点云的物理盲区。
- **引入结构化排序监督 ($+\mathcal{L}_{relative}$)**：BEV mAP 进一步提升至 46.86%（增幅 +0.78%）。证明在全图绝对深度约束的基础上，针对物体边缘的相对深度阶跃约束能有效改善轮廓清晰度。

### 2. $\mathcal{L}_{prob}$ 与 $\mathcal{L}_{found}$ 的非重复性推导

概率监督（$\mathcal{L}_{prob}$）与基础模型引导监督（$\mathcal{L}_{found}$）在数学优化目标和空间作用域上完全解耦，二者不构成冗余。

#### A. 数学优化目标的维度差异

- **$\mathcal{L}_{prob}$ 约束分布形状（方差）**：其直接作用于 Softmax 激活后、求期望之前的**概率密度向量**。通过最小化与高斯分布的 KL 散度，强制网络在雷达真值所在的深度区间输出极高的概率（如 0.99），压低其他区间的概率。此操作从数学上降低了深度分布的方差，确保在 LSS（Lift-Splat-Shoot）视锥转换时特征集中，消除几何散射（Geometric Scattering）。
- **$\mathcal{L}_{found}$ 约束期望数值（均值）**：其作用于概率分布求期望后得到的**连续标量深度值** $\hat{d}$。无论概率分布的方差大小，只要其期望值偏离了雷达测量值或视觉模型伪真值，Smooth L1 损失就会产生梯度，强制回归绝对尺度。

#### B. 空间作用域的互补

- **$\mathcal{L}_{prob}$ 仅具有稀疏性**：由于需要极高精度的绝对测量值来构建可靠的高斯目标分布，KL 散度计算严格限定在有雷达回波的稀疏像素集合 $\mathcal{M}_{sparse}$ 上。
- **$\mathcal{L}_{found}$ 具有全局密集性**：$\mathcal{L}_{found}$ 包含 $\lambda_{dense}\mathcal{L}_{dense}$。它利用视觉大模型在全图像素集合 $\mathcal{M}_{all}$ 上提供密集的平滑约束。在没有雷达点的区域（如吸波材质表面或天空背景），网络完全依赖 $\mathcal{L}_{dense}$ 提供空间深度结构。


### 2.2 可变形门控时序融合模块 (Deformable Gated Temporal Fusion, DGTF)

在无自车位姿先验的情况下，将时序融合解耦为空间对齐与状态更新两个独立操作。

**空间对齐分支**：使用 DCNv2 显式对齐历史与当前特征，补偿非刚性运动。
$$
(\Delta p, m) = Conv_{offset}(Concat(X_t, H_{t-1}))
$$

$$
H_{t-1}' = DCNv2(H_{t-1}, \Delta p, m)
$$

**门控更新分支**：采用 ConvGRU 机制，自适应平衡历史噪声与当前观测。更新门 $z_t$ 和候选特征 $\tilde{H}_t$ 决定最终隐藏状态。
$$
H_t = (1 - z_t) \odot X_t + z_t \odot \tilde{H}_t
$$

### 2.3 实例引导动态精炼模块 (Instance-Guided Dynamic Refinement, IGDR)

利用 2D 实例语义构建无几何投影误差的特征先验，校准 BEV 特征。

- **先验构建**：通过 Softmax 归一化实例空间分布，将实例原型广播至 BEV 空间。

  $$E_{BEV} = BMM(Softmax_{dim=N}(\frac{S_{BEV}}{\tau}), E_{proj})$$

- **动态校准与门控**：由 $E_{BEV}$ 通过 2D 卷积生成仿射变换参数（$\gamma_{BEV}, \beta_{BEV}$），作用于 BEV 特征，并使用前景门控机制过滤背景污染。

------

## 3. 实验 (TJ4DRadSet & VoD)

### 3.1 模块正交增益 (TJ4DRadSet)

系统级消融实验证明各模块带来的性能提升是正交叠加的。

| **模型配置**      | **BEV mAP**       | **3D mAP** |
| ----------------- | ----------------- | ---------- |
| **Baseline**      | 45.15             | 39.86      |
| **+ PDF**         | 46.86 (+1.71)     | 41.41      |
| **+ DGTF**        | 50.41 (+3.55)     | 44.86      |
| **+ IGDR (Ours)** | **54.07** (+3.66) | **47.29**  |

### 3.2 关键机制分析

- **即插即用性**：将 R4Det 模块附加于 BEVFusion 和 RCBEVDet 上，在 VoD 数据集上的 $\text{mAP}_{\text{EAA}}$ 分别提升了 +6.34% 和 +5.34%。
- **时序帧距敏感度**：在 DGTF 模块中，单步长（$t-1$）融合表现最佳（BEV mAP 50.41）。跨帧融合（$t-2, t-3$）会引发轨迹错位与误差累积。
- **通道对齐与优化冲突**：在 DGTF 中叠加 SE 通道注意力会使 BEV mAP 从 50.41 降至 49.12。SE 的全局空间平均池化（Squeeze）抹除了 DCNv2 建立的细粒度空间局部响应，且与 ConvGRU 自身的动态加权机制产生梯度冲突。