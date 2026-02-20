# CornerNet

## 方法

CornerNet 摒弃了传统的“锚框（Anchor Box）”机制，而是将目标检测重新定义为**关键点检测与配对**的问题。

将一个物体检测为一对关键点——**左上角点 (Top-Left corner)** 和 **右下角点 (Bottom-Right corner)** 。

**优势**：消除了对 Anchor Box 的依赖，从而避免了设计大量锚框带来的正负样本不平衡以及复杂的超参数设计（如锚框大小、长宽比）。

---

### Architecture

使用 **Hourglass Network ** 作为骨干网络，两个堆叠的 Hourglass 模块，深度为 104 层。这种结构通过重复的下采样和上采样，能够同时捕获**全局信息**（用于分类）和**局部细节**（用于精确定位）。

**无多尺度特征**：与许多其他检测器不同，CornerNet 不使用特征金字塔（FPN）来检测不同大小的物体，而是仅使用网络最后一层的输出进行预测。

![](../../assets/CornerNet/Figure2_Overview.png)

### Prediction Modules

骨干网络的输出被送入两个独立的预测模块：**Top-Left Prediction Module** 、 **Bottom-Right Prediction Module** 。

每个预测模块内部的结构如下：

* 改进残差块：这是模块的第一部分。它将标准残差块中的第一个 $3 \times 3$ 卷积层替换为了 **Corner Pooling Module (角点池化模块)** 。
* 后处理：经过池化和残差连接后，特征图通过卷积层融合，最终分支输出 **Heatmaps、Embeddings 和 Offsets**。

![](../../assets/CornerNet/Figure7_PredictA.png)

#### Corner Pooling (角点池化)

很多时候，边界框的角点其实在物体外部，例如一个圆球的左上角背景是空的，没有局部的视觉特征可供识别。

**思路：** 要判断一个点是不是“左上角”，网络需要向**右**看（有没有物体的上边缘？）和向**下**看（有没有物体的左边缘？）。

**实现：** 以**左上角池化 (Top-Left Corner Pooling)** 为例，包含两个路径。一个路径从右向左进行 Max-pooling，另一个路径从下向上进行 Max-pooling，最后将这两个经过处理的特征图**相加**。

![](../../assets/CornerNet/Figure6_CornerPooling.png)

### Heatmaps—— 预测位置

**输出**：两组热力图（左上和右下），尺寸为 $H \times W$，通道数 $C$（类别数）。

**标签设计 (Soft Label)**：

* 仅在 Ground Truth 位置为 1（正样本）。
* 在 Ground Truth 附近，使用 **2D 高斯分布** ($e^{-\frac{x^2+y^2}{2\sigma^2}}$) 来减少对负样本的惩罚。
* **半径确定**：根据物体大小动态计算半径，确保半径内的点生成的框与真值框 IoU $\ge 0.7$ 。

**损失函数 ($L_{det}$)**：Focal Loss 的变体：$(1 - y_{cij})^\beta$ 项 降低了真值附近（$y_{cij}$ 接近 1）负样本的权重。
$$
L_{det} = \frac{-1}{N} \sum_{c=1}^{C} \sum_{i=1}^{H} \sum_{j=1}^{W} \begin{cases} (1 - p_{cij})^\alpha \log(p_{cij}) & \text{if } y_{cij} = 1 \\ (1 - y_{cij})^\beta (p_{cij})^\alpha \log(1 - p_{cij}) & \text{otherwise} \end{cases}
$$


### Embeddings —— 预测配对

**目的**：解决“哪个左上角属于哪个右下角”的问题（即分组）。

**原理**：基于 Associative Embedding。网络为每个角点预测一个 1 维向量。

**损失函数**：

* **Pull Loss ($L_{pull}$)**：拉近属于同一物体的角点嵌入向量，使其接近均值 $e_k$ 。
  $$
  L_{pull} = \frac{1}{N} \sum_{k=1}^{N} [(e_{t_k} - e_k)^2 + (e_{b_k} - e_k)^2]
  $$
  
* **Push Loss ($L_{push}$)**：推开不同物体的嵌入向量均值，设定阈值 $\Delta=1$ 。
  $$
  L_{push} = \frac{1}{N(N-1)} \sum_{k=1}^{N} \sum_{j \neq k}^{N} \max(0, \Delta - |e_k - e_j|)
  $$
  

### Offsets —— 缩放的精度微调

**目的**：解决下采样带来的坐标精度丢失问题。

**问题来源**：映射从原图到 Heatmap 时包含取整操作：$(\lfloor \frac{x}{n} \rfloor, \lfloor \frac{y}{n} \rfloor)$，这会丢失小数精度。

**计算公式**：预测取整后丢失的小数部分。
$$
o_k = (\frac{x_k}{n} - \lfloor \frac{x_k}{n} \rfloor, \frac{y_k}{n} - \lfloor \frac{y_k}{n} \rfloor)
$$
**损失函数 ($L_{off}$)**：使用 Smooth L1 Loss，仅在 Ground Truth 位置计算。
$$
L_{off} = \frac{1}{N} \sum_{k=1}^{N}  \text{SmoothL1Loss}(\mathbf{o_k}, \hat{\mathbf{o_k}})
$$
综上，总的损失函数为：
$$
L = L_{det} + \alpha L_{pull} + \beta L_{push} + \gamma L_{off}
$$
检测损失权重：1，为主导项。$\alpha$ ：0.1、 $\beta$ ：0.1、 $\gamma$ ：1
