# SurroundOcc: Multi-Camera 3D Occupancy Prediction

**出处会议：** ICCV 2023
**是否开源：** https://github.com/weiyithu/SurroundOcc
**关键词：** 3D 占用预测、空间注意力、密集真值生成

------

## 1. 概述

解决自动驾驶中稀疏点云无法描述完整几何形状以及 BEV 架构丢失高度信息的问题。

通过将多视图图像特征投影到 **3D 体素查询 (Volume Queries)** 中，并利用 **3D U-Net** 结构进行多尺度特征融合，最终实现密集的 3D 占用（Occupancy）和语义预测。

$$\text{Multi-view Images} \rightarrow \text{2D Backbone} \rightarrow \text{Spatial Attention (2D-to-3D)} \rightarrow \text{3D U-Net} \rightarrow \text{Dense Occupancy}$$

**整体流程**：

1. **2D 特征提取**：从 $N$ 个摄像头提取多尺度图像特征。
2. **空间提升 (Lifting)**：利用空间注意力机制，将 2D 特征填充到 3D 体素网格中。
3. **多尺度融合**：通过 3D 反卷积逐步提升分辨率并融合特征。
4. **密集监督**：利用离线生成的密集点云真值进行多尺度深度监督。

------

## 2. 核心方法

### 2D-3D 空间注意力 (Spatial Attention)

这是将图像信息转换至物理 3D 空间的核心模块，遵循 **3D 为 Query，2D 为 KV** 的逻辑。

- **Query (Q)**：预定义的 3D 空间体素网格特征 $Q \in \mathbb{R}^{C \times H \times W \times Z}$。

- **Key / Value (KV)**：多摄像头的 2D 特征图。

- **投影机制**：

  1. 利用相机内外参矩阵 $P$ 将 3D 查询点投影至各 2D 视图。

  2. **可变形注意力 (Deformable Attention)**：仅在投影点附近的采样点提取特征，极大地降低了 $O(N^2)$ 的计算开销。

  3. **计算公式**：

     $$\text{Output}(Q^p) = \frac{1}{|\mathcal{V}_{hit}|} \sum_{i \in \mathcal{V}_{hit}} \text{DeformAttn}(Q^p, \mathcal{P}(q^p, i), X_i)$$

### 多尺度预测与融合 (Multi-scale Fusion)

为了在计算效率和精细度之间取得平衡，采用了类似 FPN 但基于 3D 算子的结构。

- **融合算子**：采用 **3D 反卷积 (3D Deconvolution)** 进行上采样，并使用 **逐元素相加 (Addition)** 融合不同层级的特征。

  $$Y_j = \text{Conv}_{3d}(F_j + \text{Upsample}_{3d}(Y_{j-1}))$$

- **优势**：相比拼接（Concatenation），加法融合在不增加通道数的前提下引入了高层语义，显著节省了 3D 卷积极其珍贵的显存空间。

------

## 3. 密集占用真值生成 (Dense GT Generation)

由于原始 LiDAR 点云极为稀疏，无法直接训练密集预测模型。SurroundOcc 提出了一套离线生成 Pipeline。

1. **多帧点云拼接**：利用位姿信息将多帧点云对齐。动态物体根据 Bounding Box 进行运动补偿，从而获得极其密集的点云。
2. **泊松曲面重建 (Poisson Reconstruction)**：通过求解泊松方程将离散点转换为连续曲面网格 $\mathcal{M}$，填补物理孔洞。
3. **体素化**：将网格 $\mathcal{M}$ 重新映射回离散的 3D 空间体素。
4. **语义传播**：利用 **最近邻 (NN)** 算法，将原始点云的语义标签传播给新生成的密集体素。

------

## 4. 损失函数 (Loss Function)

采用多尺度深度监督策略，确保每一层级的 3D 特征都能学习到正确的几何结构：

$$L_{total} = \sum_{j=1}^{M} \alpha_j (L_{ce}^j + L_{sem\_aff}^j)$$

- **$L_{ce}$**：多分类交叉熵，预测每个体素的语义类别（或是否占用）。
- **$L_{sem\_aff}$ (场景-类别亲和力)**：关注相邻体素之间的类别一致性，优化局部几何结构的连贯性。
- **权重衰减 $\alpha_j$**：高分辨率层级的权重更大 ($\alpha_j = 1/2^j$)，强制模型优化细节。

------

## 5. 实验分析

### 5.1 nuScenes 性能对比

在 3D 占用预测任务上，SurroundOcc 显著优于早期的 BEV 架构。

| **Method**      | **Input**  | **SC IoU (几何)** | **SSC mIoU (语义)** |
| --------------- | ---------- | ----------------- | ------------------- |
| BEVFormer       | Camera     | 30.50             | 16.75               |
| TPVFormer       | Camera     | 30.86             | 17.10               |
| **SurroundOcc** | **Camera** | **31.49**         | **20.30**           |

### 5.2 关键消融实验结论

- **密集监督的威力**：将监督信号从“稀疏点云”替换为“密集真值”后，SC IoU 从 **11.96% 飙升至 31.49%**。这证明了高质量数据对 3D 学习的决定性作用。
- **3D vs BEV**：显式的 3D 体素查询比将高度压缩的 BEV 查询在捕捉立交桥、电线杆等垂直结构上具有明显优势。

------

### 总结

1. **空间一致性**：通过 2D-3D 空间注意力，在 3D 坐标系下直接建模，避免了投影塌陷。
2. **数据是王道**：泊松重建生成的密集真值是该模型性能跨越式提升的核心“补药”。
3. **效率与精度**：利用 3D 反卷积和加法融合，在 RTX 3090 上实现了可接受的推理速度。