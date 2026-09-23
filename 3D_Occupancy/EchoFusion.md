# EchoFusion

**出处会议：** NeurIPS 2023

**是否开源：** https://github.com/tusen-ai/EchoFusion

**关键词：** 原始雷达数据融合、极坐标 BEV、极坐标对齐注意力 (PAA)、3D 目标检测

------

## 1. 概述

传统的雷达-摄像头融合方法通常依赖于经过 CFAR 检测后生成的稀疏雷达点云。这种流程会丢失大量包含弱回波信号的原始信息，且点云的方位角分辨率较低。

**EchoFusion** 提出了“跳过信号处理”的范式，直接在 **极坐标 BEV 空间** 下融合原始雷达特征（如 Range-Time 图）与图像语义特征。其核心是通过几何约束实现两种模态在特征层面的精确对齐。

------

## 2. 技术实现

EchoFusion 采用基于 Transformer 的架构，其核心组件包括极坐标 BEV 查询初始化、极坐标对齐注意力（PAA）以及极坐标解码头。

### 2.1 极坐标对齐注意力 (Polar-Aligned Attention, PAA)

PAA 模块利用 3D 空间与传感器数据维度之间的几何对应关系，分两步进行特征聚合：

#### 1. 列向图像融合 (Column-wise Image Fusion)

**公式推导**：
$$
x_I \approx u_0 + f_x \frac{-r \sin \phi + d_1}{r \cos \phi + d_3} \approx u_0 - f_x \tan \phi_{bev}
$$

当距离 $r$ 远大于相机偏移量 $d$ 时，$r$ 被约去，$x_I$ 仅取决于 $\phi$。

这意味着对于给定的方位角 $\phi_{bev}$，无论距离 $r_{bev}$ 是多少，其投影在图像上的横坐标 $x_I$ 几乎是恒定的。因此，在极坐标 BEV 空间中，沿着同一条射线（固定 $\phi$）的所有点都会投影到图像的同一列。

由于上述距离抵消特性，同一个 $\phi$ 扇区内的所有 Pillar 在图像上横向重叠。因此，模型将同一方位角 $\phi$ 下的所有 BEV Queries 聚合在一起，与图像对应列 $x_I$ 的特征进行 Cross-Attention。这种设计使得网络能够学习从图像的垂直空间信息（高度）中提取特征，以填充 BEV 空间中缺失的深度/高度线索。

#### 2. 行向雷达融合 (Range-wise Radar Fusion)

**BEV 查询 ($Q_{bev}$)**：经过列向图像融合（Column-wise Image Fusion）更新后的特征，维度表示为 $Q \in \mathbb{R}^{R \times A \times d}$，其中 $R$ 是径向距离切分数量，$A$ 是方位角切分数量，$d$ 是特征维度。

**雷达特征图 ($F_R$)**：通常采用距离-时间（Range-Time, RT）图作为输入，维度表示为 $F_R \in \mathbb{R}^{R_I \times T_I \times d}$。其中 $R_I$ 是距离 Bin 的数量，$T_I$ 是 Chirps（或时间步）的数量。

极坐标 BEV 的径向距离 $r$ 与雷达距离-时间（RT）图的距离轴天然对齐。

对于 BEV 空间中任意一个位于距离索引 $r$、方位角索引 $\phi$ 的查询向量 $q(r, \phi)$，其在雷达 RT 图中对应的物理信息仅存在于第 $r$ 行（即相同的距离 Bin）。

在同一距离 $r$ 下，所有的方位角查询 $\{q(r, \phi_1), q(r, \phi_2), \dots, q(r, \phi_A)\}$ 都对应 RT 图中的同一行特征 $\{F_R(r, t) \mid t=1, \dots, T_I\}$。

每一个距离索引为 $r$ 的 BEV Query 只与 RT 特征图中第 $r$ 行的特征（跨越所有时间步/Chirps）进行交互。
$$
\tilde{q}(r, \phi) = \text{Attention}(q(r, \phi), F_R(r, \cdot), F_R(r, \cdot))
$$
**效果**：保留了无损的距离和速度（多普勒）信息，隐式提取目标的方位角线索。

### 2.2 极坐标解码器与回归分支

检测头在极坐标系下直接预测目标属性，避免了笛卡尔坐标转换带来的插值失真。

- **极坐标交叉注意力**：对象查询（Object Queries）在极坐标网格中生成参考点 $(\hat{\rho}, \hat{\phi})$，并直接采样对应的 BEV 特征。
- **回归参数化**：
  - **位置**：预测极坐标偏移 $(\Delta \rho, \Delta \phi)$ 和高度 $z$。
  - **尺寸**：预测 $(\log l, \log w, \log h)$。
  - **朝向**：预测相对于极坐标轴的偏航角 $(\sin \theta, \cos \theta)$。

------

## 3. 实验结果

论文在 RADIal 数据集上进行了核心评估，对比了不同模态组合及其在 3D 检测任务中的表现。

### 3.1 RADIal 数据集主要性能对比

| **方法 (Methods)**    | **模态 (Modality)** | **AP (%)** | **AR (%)** | **F1 (%)** | **RE (m)** | **AE (°)** |
| --------------------- | ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| FFTRadNet             | RD                  | 96.84      | 82.18      | 88.91      | 0.11       | 0.17       |
| ADCNet                | ADC                 | 95.00      | 89.00      | 91.90      | 0.13       | 0.11       |
| **EchoFusion (Ours)** | **RT + Image**      | **96.95**  | **93.43**  | **95.16**  | 0.12       | 0.18       |

### 3.2 消融实验：雷达数据格式与模态影响

| **实验编号** | **模态组合**      | **雷达格式**    | **mAP (%)** | **NDS (%)** |
| ------------ | ----------------- | --------------- | ----------- | ----------- |
| 1            | 仅图像            | -               | 19.3        | 31.9        |
| 2            | 图像 + PCD        | 稀疏点云        | 33.1        | 47.9        |
| 3            | 图像 + RA Map     | 距离-方位图     | 38.6        | 54.7        |
| **4**        | **图像 + RT Map** | **距离-时间图** | **42.4**    | **59.3**    |

### 3.3 坐标系选择对比

| **坐标系 (Coordinate System)** | **mAP (%)** | **NDS (%)** |
| ------------------------------ | ----------- | ----------- |
| 直角坐标系 (Cartesian)         | 41.5        | 58.6        |
| **极坐标系 (Polar)**           | **42.4**    | **59.3**    |

------

## 4. 核心结论

1. **原始数据的优越性**：RT 图提供的连续能量分布远优于经过硬阈值过滤后的雷达点云（PCD）。
2. **几何对齐机制**：通过 PAA 模块在极坐标下解耦方位角（对应图像列）和距离（对应雷达行），实现了高效的多模态特征聚合。
3. **极坐标回归**：直接在极坐标空间进行边界框回归，能够更好地适应雷达传感器的测量特性。





# 附录

FMCW 雷达信号处理的本质是对一个三维数据立方体（Data Cube）进行多维离散傅里叶变换（DFT）。以下通过数学表达描述从原始采样到各类特征图的演进关系。

设定基础参数：

- $N_s$: 每个 Chirp 的采样点数（快时间维度）。
- $N_c$: 每帧的 Chirp 数量（慢时间维度）。
- $N_a$: 接收天线通道数（空间维度）。
- $S \in \mathbb{C}^{N_s \times N_c \times N_a}$: 原始 ADC 采样数据立方体。

------

### 1. ADC 原始数据 $\rightarrow$ RT 图 (Range-Time Map)

**算子：Range-FFT**

对快时间维度 $n_s$ 进行一维 FFT。对于任意天线通道 $a$ 和第 $m$ 个 Chirp：

$$X_{RT}(k, m, a) = \sum_{n_s=0}^{N_s-1} S(n_s, m, a) \cdot w_s(n_s) \cdot e^{-j \frac{2\pi}{N_{FFT,r}} k n_s}$$

其中 $k$ 为距离门索引，$w_s$ 为窗函数。

- **物理关系**：$k$ 对应距离 $R = \frac{k \cdot c \cdot f_s}{2K \cdot N_{FFT,r}}$（$K$ 为扫频斜率，$f_s$ 为采样率）。
- **RT 图定义**：通常指固定 $a$ 后的二维矩阵 $\mathbf{M}_{RT} \in \mathbb{C}^{N_{FFT,r} \times N_c}$。它保留了慢时间维度的相位演进 $\Delta \phi = \frac{4\pi v T_c}{\lambda}$。

------

### 2. RT 图 $\rightarrow$ RD 图 (Range-Doppler Map)

**算子：Doppler-FFT**

在 RT 图的基础上，沿着慢时间维度 $m$（Chirp 轴）进行第二次 FFT：

$$X_{RD}(k, l, a) = \sum_{m=0}^{N_c-1} X_{RT}(k, m, a) \cdot w_c(m) \cdot e^{-j \frac{2\pi}{N_{FFT,d}} l m}$$

- **物理关系**：$l$ 对应多普勒频率，进而对应径向速度 $v = \frac{l \cdot \lambda}{2 N_c T_c}$。
- **RD 图定义**：$\mathbf{M}_{RD} \in \mathbb{C}^{N_{FFT,r} \times N_{FFT,d}}$。此过程将时域的相位旋转相干积分为频域的能量峰值。

------

### 3. RD 图 $\rightarrow$ RA 图 (Range-Azimuth Map)

**算子：Angle-FFT / Beamforming**

对接收天线维度 $a$ 进行第三次 FFT（针对等间距线阵 ULA）：

$$X_{RAD}(k, l, p) = \sum_{a=0}^{N_a-1} X_{RD}(k, l, a) \cdot w_a(a) \cdot e^{-j \frac{2\pi}{N_{FFT,a}} p a}$$

- **物理关系**：$p$ 对应空间频率，反映入射角 $\theta = \arcsin(\frac{p \lambda}{N_{FFT,a} d})$（$d$ 为阵元间距）。

- **RA 图定义**：通常通过对 $X_{RAD}$ 在多普勒维度 $l$ 进行能量累加或取最大值（Max-pooling）得到：

  $$\mathbf{M}_{RA}(k, p) = \sum_{l=0}^{N_{FFT,d}-1} |X_{RAD}(k, l, p)|^2$$

------

### 4. RA 图 $\rightarrow$ RP 图 (Range-Polar Map)

**算子：坐标映射与特征重组**

RP 图通常指极坐标下的 BEV 表征。在深度学习框架中，它建立了从距离索引 $k$ 和角度索引 $p$ 到极坐标网格的直接映射。

若定义极坐标网格为 $(r_i, \theta_j)$，其映射关系为：

1. **距离对齐**：$r_i = k \cdot \Delta R$，即 $N_r$ 个距离划分（Range Partition）与 RT/RD 图的距离门完全一致。
2. **角度分布**：将 Angle-FFT 得到的非线性分布角 $\theta = \arcsin(\dots)$ 重采样或映射到均匀分布的 $\theta_j$。

**数学关系概括**：

$$S(n_s, m, a) \xrightarrow{\text{FFT}_{n_s}} RT(k, m, a) \xrightarrow{\text{FFT}_{m}} RD(k, l, a) \xrightarrow{\text{FFT}_{a}} RAD(k, l, p) \xrightarrow{\text{Collapse } l} RA(k, p) \approx RP(r, \theta)$$

------

### 5. 核心逻辑链总结

- **ADC** 是时域/空间域的原始离散采样复数张量。
- **RT** 是距离域-时间域特征，保留了原始速度线索（相位演进趋势）。
- **RD** 是距离域-频率域特征，解耦了距离和速度，是传统检测（CFAR）的输入。
- **RA** 是距离域-角度域特征，是水平面内目标定位的直接依据。
- **RP** 是将上述特征按照极坐标几何逻辑组织的特征图，用于与视觉等其他模态在统一的极坐标/BEV 空间内对齐。