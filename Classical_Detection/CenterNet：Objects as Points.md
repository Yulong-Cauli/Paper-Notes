# CenterNet：Objects as Points

**出处会议：** arXiv 2019  
**是否开源：** https://github.com/xingyizhou/CenterNet  
**关键词：** 中心点检测、热力图、无锚框、无NMS

---

## 方法

CenterNet 将 CornerNet 中复杂的**关键点配对**步骤简化为直接的**属性回归**。

将一个物体建模为边界框的**中心点 (Center Point)**。一旦找到中心点，其他的属性（如宽高、偏移量）直接从该点的图像特征中回归得到。

**优势**：

1. **Anchor-free**：不需要设计锚框的大小和比例，不需要 IoU 阈值匹配。
2. **无需 NMS**：每个目标只由一个中心点表示，通过热力图峰值提取即可自然分离目标，推理过程不需要非极大值抑制（NMS）。
3. **通用性**：框架易于扩展，只需增加额外的输出头即可支持 3D 检测或人体姿态估计。

------

### Architecture

网络架构采用标准的**全卷积编码器-解码器 (Encoder-Decoder)** 结构。

- **骨干网络 (Backbone)**：论文提供了三种选择以平衡速度与精度：
  - **ResNet-18 / ResNet-101**：增加了转置卷积层以进行上采样。
  - **DLA-34 (Deep Layer Aggregation)**：使用层级跳跃连接，并引入可变形卷积 (Deformable Conv) 增强特征。
  - **Hourglass-104**：堆叠沙漏网络，用于获取最佳精度。
- **下采样步长 ($R$)**：默认输出步长为 **4**。即输入 $512 \times 512$ 的图像，输出 $128 \times 128$ 的特征图。

### Prediction Heads

网络在骨干输出的特征图上，并行连接了三个独立的**卷积头 (Heads)**。这三个头共享主干特征，结构均为 $3 \times 3$ 卷积接 $1 \times 1$ 卷积。

#### Heatmaps —— 预测中心点

**输出**：尺寸为 $\frac{H}{R} \times \frac{W}{R}$ ，通道数 $C$ （类别数，如 COCO 为 80）。

**原理**：这是一个关键点估计问题。

- 预测值 $1$ 代表该位置是物体的中心。
- 预测值 $0$ 代表背景。

标签设计 (Soft Label)：

沿用 CornerNet 的策略。

- 仅在 Ground Truth 中心点位置为 1。
- 在中心点附近，根据物体尺寸计算半径，生成 **2D 高斯圆** ($e^{-\frac{x^2+y^2}{2\sigma^2}}$)，以此作为软标签，减少对正样本附近负样本的惩罚。

**损失函数 ($L_{k}$)**：修改版的 Focal Loss，用于处理正负样本极度不平衡问题。

$$
L_{k} = \frac{-1}{N} \sum_{xyc} \begin{cases} (1 - \hat{Y}_{xyc})^\alpha \log(\hat{Y}_{xyc}) & \text{if } Y_{xyc} = 1 \\ 
(1 - Y_{xyc})^\beta (\hat{Y}_{xyc})^\alpha \log(1 - \hat{Y}_{xyc}) & \text{otherwise} \end{cases}
$$


其中 $\alpha=2, \beta=4$ 是超参数， $N$ 是图像中关键点的数量。

#### Object Size —— 预测尺寸

**输出**：尺寸为 $\frac{H}{R} \times \frac{W}{R}$ ，通道数 $2$ （分别代表宽 $w$ 和高 $h$ ）。

**目的**：**替代 CornerNet 的 Embedding 配对**。既然已经锁定了物体的中心点，直接回归该物体的尺寸即可确定边界框，无需去匹配左上角和右下角。

**实现**：直接回归原始像素坐标下的物体尺寸 $s_k = (x_2^{(k)} - x_1^{(k)}, y_2^{(k)} - y_1^{(k)})$ 。为了减少计算负担，所有类别共享这个 Size Head。

**损失函数 ($L_{size}$)**：L1 Loss，仅在中心点位置计算。

$$
L_{size} = \frac{1}{N} \sum_{k=1}^{N} |\hat{S}_{p_k} - s_k|
$$

#### Local Offsets —— 精度修正

**输出**：尺寸为 $\frac{H}{R} \times \frac{W}{R}$ ，通道数 $2$ （x 轴偏移, y 轴偏移）。

目的：解决下采样（Stride=4）带来的离散化误差。

中心点坐标映射回原图时，公式为 $p = \lfloor \frac{p_{raw}}{R} \rfloor$ 。这会丢失精度，导致小物体检测 IoU 下降。

**计算公式**：预测下采样坐标与浮点坐标的差值。

$$
O_{\tilde{p}} = \frac{p}{R} - \tilde{p}
$$


**损失函数 ($L_{off}$)**：L1 Loss，仅在中心点位置计算。

$$
L_{off} = \frac{1}{N} \sum_{p} |\hat{O}_{\tilde{p}} - (\frac{p}{R} - \tilde{p})|
$$

<div align="center">
    <img src="https://raw.githubusercontent.com/Yulong-Cauli/Paper-Notes/main/assets/CenterNet/Figure_OffsetLoss_2.png" width="20%">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
      注意：这里和CornerNet的Offset Loss有一点区别
    </div>
</div>

### Inference (从点到框)

推理过程非常简单，是 CenterNet 的一大亮点：

1. **提取峰值**：在热力图上对每个类别的通道进行 $3 \times 3$ Max Pooling。如果某个点的值大于或等于其周围 8 个邻居，则保留该点作为检测到的中心点。
2. **解码**：取前 100 个峰值点，根据预测的偏移量 $\delta$ 和尺寸 $(w, h)$ 解码成边界框：
   - $x_{center} = (\hat{x} + \delta_{\hat{x}}) \times R$
   - $y_{center} = (\hat{y} + \delta_{\hat{y}}) \times R$
   - $Box = (x_{center} - w/2, y_{center} - h/2, x_{center} + w/2, y_{center} + h/2)$
3. **无需 NMS**：因为热力图峰值提取已经保证了局部唯一性。

------

### Total Loss

总的训练目标是上述三个损失的加权和：

$$
L_{det} = L_{k} + \lambda_{size} L_{size} + \lambda_{off} L_{off}
$$

**权重设置**：

- $\lambda_{size} = 0.1$
- $\lambda_{off} = 1$
- $L_k$ 为主导项。

------

## 对比分析：CenterNet vs. CornerNet

| **比较维度** | **CornerNet (2018)**                                    | **CenterNet (2019)**                                    |
| :----------: | ------------------------------------------------------- | ------------------------------------------------------- |
| **物体建模** | **一对关键点** ，左上角 + 右下角                        | **单一关键点** (中心点)                                 |
| **关键难点** | **配对 (Grouping)**：怎么知道哪个左上角对应哪个右下角？ | **无**：一个点即代表物体，不存在配对问题                |
| **解决方案** | 预测 **Embedding** 向量，通过距离聚类                   | 直接预测 **Size (宽高)**                                |
| **特征提取** | **Corner Pooling**：扫描边缘特征                        | **Center Point**：利用物体内部特征，无需特殊池化        |
|  **后处理**  | 需计算 Embedding 距离，速度较慢                         | **无 NMS**，直接提取峰值，速度极快                      |
| **检测速度** | 相对较慢 (e.g., Hourglass-104: 4.1 FPS)                 | 实时高效 (e.g., Hourglass-104: 7.8 FPS, DLA-34: 52 FPS) |



#### A. 移除了“配对 (Grouping)”步骤 —— 最大的提速点

- **CornerNet 的痛点**：图像中如果有多个同类物体，网络会输出一堆左上角和一堆右下角。CornerNet 必须引入额外的 **Embedding Head**，训练时让同一物体的角点 Embedding 相似。推理时，需要计算所有角点之间的距离来通过 Embedding 匹配成对，这是一个组合过程，严重拖慢了速度。
- **CenterNet 的改进**：由于中心点本身就是物体的唯一标识，**“配对”问题直接消失了**。网络只需要在这个中心点上“顺便”回归出物体的宽和高（Size Head）即可。这使得 CenterNet 的推理变成了纯粹的前向传播，无需复杂的组合逻辑。

#### B. 特征提取更自然 —— 移除了 Corner Pooling

- **CornerNet 的痛点**：角点（如左上角）通常在物体轮廓之外（背景区域），那里没有物体的视觉特征。因此，CornerNet 必须设计 **Corner Pooling**，强行让特征图去“向右看”和“向下看”寻找物体边缘。这增加了网络设计的复杂性。
- **CenterNet 的改进**：物体的**中心点**通常落在物体内部（即便是“C”形物体，中心点往往也能捕获丰富的语义上下文）。因此，CenterNet 不需要特殊的池化层，直接提取该位置的骨干网络特征即可进行回归。

#### C. 后处理的极简主义 —— 移除了 NMS

- **CornerNet**：虽然也使用了热力图，但在生成框之后，仍可能产生重叠的检测结果，通常需要 Soft-NMS 来进一步清理 17。
- **CenterNet**：通过在热力图上执行简单的 **$3 \times 3$ Max Pooling** 来提取峰值，这一步本身就起到了非极大值抑制的作用。只要两个物体的中心点不重合，它们就能被区分开。这使得 CenterNet 成为真正的 **End-to-End（端到端）** 检测器，无需后处理 18181818。

------

### 潜在局限性 Center Collision

虽然 CenterNet 在简洁性和速度上完胜，但在理论上有一个短板：**中心点重叠**。如果两个不同物体（同类别）的中心点，在经过下采样（Stride $R=4$ ）后，落入了特征图的同一个像素格子里，CenterNet 就只能检测出其中一个，另一个会被“吃掉”。

而，CornerNet 只要左上角或右下角不同，理论上就能区分重叠严重的物体。

**实际影响**：CenterNet 作者在 COCO 数据集上进行了统计，发现这种“中心点碰撞”极为罕见（<0.1%），远远低于基于 Anchor 的方法因 IoU 匹配失败导致的漏检率（约 20%）。因此，这个理论缺陷在实际应用中几乎可以忽略不计 20。



