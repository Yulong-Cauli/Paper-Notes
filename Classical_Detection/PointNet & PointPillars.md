# PointNet

## 点云无序性 & 对称函数

**问题**：输入顺序不应影响结果，点云本质上是一个集合 $\{P_1, P_2, ..., P_n\}$ ，没有固定顺序。如果把点云数据重新排列），它代表的形状没有任何改变。

然而，传统的神经网络（如 RNN）或全连接层对输入顺序非常敏感。PointNet 必须设计一种机制，让网络对输入点的 $N!$ 种排列都具有置换不变性 (Permutation Invariance) 。

**解法： $f(\{x_i\}) \approx g(h(x_i))$**

$$
f(\{x_{1},...,x_{n}\}) \approx g(h(x_{1}),...,h(x_{n}))
$$

1. **$h$ (MLP): 独立处理每个点**

   - 网络首先使用多层感知机 (MLP) 单独处理每一个点。
   - **关键点：** 这个 MLP 的权重对所有点是**共享**的 (Weight Sharing)。
   - 如果输入是 $N \times 3$ （ $N$ 个点，XYZ坐标），经过几层 MLP 后，每个点被映射到了高维特征空间（例如 $N \times 1024$ ）。这时候，点与点之间还没有发生任何交互，每个点都是独立编码的。

2. **$g$ (Symmetric Function): 对称聚合**

   - 为了把 $N$ 个点的特征汇聚成一个表示整个形状的“全局特征向量”，必须使用一个**对称函数**。

   - **对称函数定义：** 输入顺序改变，输出不变的函数（例如加法、乘法、平均值）。

     **PointNet 的选择：** **Max Pooling (最大池化)** 。

   - 它在所有点的特征维度上取最大值。比如，在 1024 维的特征空间里，每一维都选取 $N$ 个点中响应最强的那个值。

**为什么是 Max Pooling？**

Max Pooling 实际上是在捕捉点云的**关键骨架 (Skeleton)**。只有那些最显著的点会在特征图上激活最大值。

既然只取最大值，那么非关键区域的点（或者少量的噪声点）即使发生扰动，也不会影响最终的全局特征。

------

## 几何变换 & 对齐网络

**问题**：如果你把一个物体旋转 90 度，它的 $(x, y, z)$ 坐标全变了，但它还是同一个物体。

**解法：T-Net (Transformation Network)**

它的作用是预测一个变换矩阵，把输入“校正”到一个标准姿态（Canonical Space）。

架构中使用了两次 T-Net ：

1. **输入对齐 (Input Transform):**

   - 在处理原始点云之前，先预测一个 $3 \times 3$ 的仿射变换矩阵。
   - 直接用这个矩阵乘以输入坐标 $(x, y, z)$ ，实现对点云的旋转/对齐。

2. **特征对齐 (Feature Transform):**

   - 在点经过几层 MLP 映射到 64 维特征后，再次使用 T-Net 预测一个 $64 \times 64$ 的变换矩阵。

   - **难点：** 优化一个 $64 \times 64$ 的矩阵难度很大，参数多，容易过拟合或退化。

   - 正则化技巧： 作者加了一个正则化项，强制要求这个特征变换矩阵 $A$ 接近正交矩阵：

$$
L_{reg} = ||I - AA^T||_F^2
$$

   **原因：** 正交变换属于刚体变换（如旋转），它不会改变向量的模长，从而避免了在高维空间中丢失输入特征的信息。

------

## 局部与全局融合

**问题**：如果是做**分类 (Classification)**，上面的 Max Pooling 输出一个全局特征向量（Global Feature，如 1024 维）就足够了，直接喂给分类器即可 。

但是，如果是做**分割 (Segmentation)**（比如要判断每个点属于“机翼”还是“机身”），光有全局特征是不够的，因为 Max Pooling 把具体的点的信息都“池化”掉了 。

**解法：特征拼接 (Concatenation)**

PointNet 采用了一个简单粗暴但有效的策略 ：

1. 拿到 Max Pooling 后的 **全局特征向量** (Global Feature)。
2. 把它**复制** $N$ 份。
3. 将它与每个点在 Max Pooling 之前的 **局部特征** (Local Point Features) 进行**拼接 (Concatenate)**。

结果：

每个点现在的特征向量 = [我自己是谁 (局部几何) + 我属于什么物体 (全局语义)]。

这样，网络在预测每个点的标签时，既知道该点的局部几何细节，又知道整个物体的上下文信息 。



#  PointPillars

**核心思想**：将 3D 点云转换为Pillars，通过 PointNet 学习柱子特征，随后将其“散射”（Scatter）回 2D 网格生成伪图像Pseudo Image，最后利用 2D CNN 进行检测。这种设计避免了昂贵的 3D 卷积，实现了速度与精度的平衡 。

<div align="center"><img src="https://raw.githubusercontent.com/Yulong-Cauli/Paper-Notes/main/assets/PointPillars/Figure2.png"></div>

##  Pillar Feature Net (PFN)

特征提取由 PFN 完成。它的任务是将原始点云数据转换为稀疏的柱子特征向量。

**Pillar 化**：将点云在 X-Y 平面上离散化为网格，Z 轴不进行切分。

代码中通过 配置文件 实现。

```yaml
POINT_CLOUD_RANGE: [0, -39.68, -3, 69.12, 39.68, 1]   # [x_min, y_min, z_min, x_max, y_max, z_max]
VOXEL_SIZE: [0.16, 0.16, 4]  # POINT_CLOUD_RANGE 的 Z 轴总长 4米 1 - (-3) = 4
```

**特征增强 (Feature Decoration)**：原始点云通常只有 $(x, y, z, r)$ 。为了丰富局部几何信息，PointPillars 将每个点的维度扩充到 **9维** $(x, y, z, r, x_c, y_c, z_c, x_p, y_p)$ 。

- $c$ 下标：点到该 Pillar 内所有点**算术平均中心**的距离。
- $p$ 下标：点到该 Pillar **网格几何中心**的距离。

代码中通过 `PillarVFE` 类实现。

```Python
# 计算柱子内所有点的算术平均值 (x_mean, y_mean, z_mean)
points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
# 计算偏差：每个点的坐标 - 该柱子的平均坐标
f_cluster = voxel_features[:, :, :3] - points_mean

# 中心偏移特征 公式：实际坐标 - (网格索引 * 步长 + 起始偏移)
f_center = torch.zeros_like(voxel_features[:, :, :3])
f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

if self.use_absolute_xyz:
    features = [voxel_features, f_cluster, f_center]
else:
    features = [voxel_features[..., 3:], f_cluster, f_center]

if self.with_distance: # 加上点到原点的距离
    points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
    features.append(points_dist)
features = torch.cat(features, dim=-1)
```

**全局特征：**对每个 Pillar 内的点应用 `Linear -> BatchNorm -> ReLU`，最后通过 `Max Pooling` 提取该 Pillar 的全局特征。

代码中通过 `PillarVFE` 类实现。

```Python
if inputs.shape[0] > self.part: # 如果柱子数量超过 50000，就分块
    # nn.Linear performs randomly when batch size is too large
    num_parts = inputs.shape[0] // self.part
    part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                       for num_part in range(num_parts+1)]
    x = torch.cat(part_linear_out, dim=0)
else: # 数量不多直接计算
    x = self.linear(inputs)
torch.backends.cudnn.enabled = False
x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x  # 调整维度以适应 BatchNorm1d: (Batch, Channels, Length)
torch.backends.cudnn.enabled = True
x = F.relu(x)
x_max = torch.max(x, dim=1, keepdim=True)[0] # Max Pooling: 提取柱子内的全局特征
# x_max 形状: [M, 1, out_channels]
if self.last_vfe: # 如果是最后一层，直接返回聚合后的特征
    return x_max
else: # 如果不是最后一层，将“全局特征(x_max)”复制并拼接到“每个点的特征(x)”后面
    x_repeat = x_max.repeat(1, inputs.shape[1], 1)
    x_concatenated = torch.cat([x, x_repeat], dim=2)
    return x_concatenated
```

------

## Pseudo Image 生成 (Scatter)

这一步是将稀疏的柱子特征还原回 2D 空间位置，形成一张Pseudo Image，以便后续接入标准的 2D Backbone。

**Scatter 操作**：将编码后的特征 $(C, P)$ 根据其在原始网格中的坐标索引，填入到大小为 $(C, H, W)$ 的画布中。

**伪图像**：结果是一个标准的 2D 张量，可以直接使用针对图像优化的 2D 卷积网络进行处理。

代码通过 `PointPillarScatter` 类实现。

**坐标索引与填充：**

```Python
for batch_idx in range(batch_size):
    # 创建一张全 0 的空白画布
    # 形状为 [C, H*W] (这里先展平为一维，方便赋值)
    spatial_feature = torch.zeros(
        self.num_bev_features,
        self.nz * self.nx * self.ny,
        dtype=pillar_features.dtype,
        device=pillar_features.device)

    # 找出属于当前 batch 的柱子
    batch_mask = coords[:, 0] == batch_idx
    this_coords = coords[batch_mask, :]  # 获取当前帧的坐标 (z, y, x)

    # 计算每个柱子在画布上的“平铺索引”
    # 这里的公式是：index = y * Width + x
    # 因为 nz=1，且 coords[:, 1] (即z轴索引) 通常为0
    indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
    indices = indices.type(torch.long)

    # 取出对应的特征向量
    pillars = pillar_features[batch_mask, :]
    pillars = pillars.t()  # 转置，变成 [C, M_batch]

    # Scatter (填充)：把特征填入画布的对应位置
    spatial_feature[:, indices] = pillars
    batch_spatial_features.append(spatial_feature)

# 堆叠并重塑形状
# 最终形状：[Batch, C, H, W]
batch_spatial_features = torch.stack(batch_spatial_features, 0)
batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny,
                                                     self.nx)

# 存回字典，交给 Backbone (2D CNN) 处理
batch_dict['spatial_features'] = batch_spatial_features
return batch_dict
```



## Detection Head

在经过 Backbone 提取深层特征后，检测头负责输出最终的 3D 框信息。

**SSD 架构**：采用单阶段检测器（Single Shot Detector），直接在特征图上密集预测。

**分类**：使用 Focal Loss 解决正负样本不平衡。

**回归**：预测 $(x, y, z, w, l, h, \theta)$ 的残差。

**方向分类**：由于回归损失（Sine-Error）无法区分车头朝前 ($0^\circ$) 还是朝后 ($180^\circ$)，网络增加了一个 softmax 分类分支来学习离散化的方向 10。

<div align="center"><img src="https://raw.githubusercontent.com/Yulong-Cauli/Paper-Notes/main/assets/PointPillars/Figure_Loss.png"></div>

代码通过 `AnchorHeadSingle` 类实现，包含三个并行的 $1\times1$ 卷积分支。

**三个核心分支：**

```Python
# 1. 类别预测分支 (Classification Branch)
# 输入: 2D 特征图 (B, C_in, H, W)
# 输出: (B, Num_Anchors * Num_Classes, H, W)
self.conv_cls = nn.Conv2d(
    input_channels,
    self.num_anchors_per_location * self.num_class,
    kernel_size=1  # 使用 1x1 卷积，对每个像素点做全连接分类
)

# 2. 边界框回归分支 (Box Regression Branch)
# 论文定义了 7 个回归目标: (x, y, z, w, l, h, theta) [cite: 230]
# self.box_coder.code_size 通常为 7
self.conv_box = nn.Conv2d(
    input_channels,
    self.num_anchors_per_location * self.box_coder.code_size,
    kernel_size=1
)

# 3. 方向分类分支 (Direction Classifier Branch)
# 因为正弦误差损失无法区分车头朝前还是朝后 (0 vs 180度)，
# 所以增加一个 softmax 分类损失来学习车头朝向
if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
    self.conv_dir_cls = nn.Conv2d(
        input_channels,
        self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,  # 通常为 2 (正向/反向)
        kernel_size=1
    )
else:
    self.conv_dir_cls = None

self.init_weights()
```

# PillarNet

PillarNet 包含三个核心模块：

1. Encoder，稀疏 2D 卷积主干网络，提取特征。
2. Neck（v1/v2/v2-D/v3），融合特征。
3. Orientation-Decoupled IoU Loss（OD-IoU）

**总体思想：** 通过多尺度 encoder + 强融合 neck 替代 PointPillars 的弱 backbone，并通过 OD-IoU 提升 box 回归稳定性。

<div align="center"><img src="https://raw.githubusercontent.com/Yulong-Cauli/Paper-Notes/main/assets/PillarNet/Figure_Compare.png"></div>

------

## Encoder Design

**问题：** PointPillars 的 backbone 非常浅，只包含 一个小型 PointNet，用于处理 pillar 内点 和一个 2D CNN。这会导致深层语义特征弱，难以处理高分辨率（小 pillar size）。

**目标：** 解决深层语义特征弱

**解决：** 构建类似 ResNet 的多尺度主干：1× （高空间精度）...16×（高语义抽象）。每个 stage 通过 Sparse 2D Convolution 构建。

**为什么使用 Sparse Conv？**

BEV pseudo-image 高度稀疏，即很多 pillar 为空，而dense convolution 会对空区域进行无意义计算。

使用 sparse convolution 会跳过空 pillar，只对非空 pillar 卷积。

此时，计算量随 “非空区域” 而非 “总分辨率” 增长。

因此：可以使用更细粒度的 pillar size，主干网络可以做得更深而不增加不可接受的计算量。

---

## Neck Design

**目标：** 需要将不同尺度特征有效融合，使网络既能看局部细节又能理解全局语义。

**v1**：类似 FPN，将各层上采样到同一尺度并拼接。

**v2**：引入来自编码器的**空间特征**（来自稀疏卷积输出），引入来自额外 **16× 下采样密集特征图**的**高层语义特征**。使用**一组卷积层**在这两类特征之间进行充分的信息交换。

**v3**：先用**一组卷积层**对 16× 下采样的密集特征图进行**进一步抽象和增强**，以提取更丰富的语义信息。再用**另一组卷积层**将增强后的语义特征与空间特征进行融合。

------

## Orientation-Decoupled IoU Loss（OD-IoU）

**问题：** 在 BEV 3D 检测中，IoU 对 box 的角度敏感，IoU 下降会导致训练梯度震荡，回归不稳定。

**解决：** 把 **方向 $\theta$** 从回归问题中 **“解耦”** 出来。

设计一个**两部分的损失函数** $\mathcal{L}_{\text{IoU-Decoupled}}$ 来取代传统的 3D IoU 损失，将 7 个参数的回归任务拆分：

1、**无方向的 IoU 损失 (Location & Size Loss)**

这一部分使用标准的 IoU 损失（或其变体），但仅作用于目标框的**中心点 $(x, y, z)$** 和**尺寸 $(l, w, h)$**。

- **工作方式：** 在计算 IoU 时，它**忽略了角度 $\theta$**，可以把它想象成计算一个**与坐标轴对齐的 3D 边界框** （即 $\theta=0$ ）的 IoU 损失。
- **优势：** 这样一来，中心点和尺寸的回归就**不会受到角度误差剧烈波动的影响**，使得优化过程更稳定、梯度更平滑，网络可以更专注于学习目标框的位置和大小。

2、**独立的方向回归损失 (Orientation Loss)**

这一部分引入了一个独立的损失项 $\mathcal{L}_{\text{Orientation}}$ （通常是 $L_1$ 损失或 Smooth $L_1$ 损失），**只针对角度 $\theta$ 的预测误差**。

- **工作方式：** 它会使用特殊的**角度编码**，来确保**角度的周期性**（如 $-\pi/2$ 和 $\pi/2$ 在几何上可能代表相似的方向）得到正确的处理，避免边界问题。
- **优势：** 这解决了传统 3D IoU 损失中，角度误差对整体 IoU 贡献过大、导致优化不稳定的问题。

通过将 $\mathcal{L}\_{\text{IoU-Decoupled}} = \mathcal{L}\_{\text{IoU-No-Orientation}} + \lambda \mathcal{L}\_{\text{Orientation}}$ 结合。

------

