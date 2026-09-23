# Paper-Notes

关于计算机视觉、3D 感知与模型量化的论文笔记，共 **28 篇**。

每篇笔记统一以「出处会议（期刊）/ 是否开源 / 关键词」三行开头，正文以公式推导 + 直觉解释为主，配图统一放在 `assets/<论文名>/` 下。

## 📂 目录结构

| 目录 | 内容 | 篇数 |
| --- | --- | --- |
| [`Classical_Detection/`](Classical_Detection) | 经典 2D 检测、点云 3D 检测、4D 雷达检测 | 9 |
| [`Multimodal_Fusion/`](Multimodal_Fusion) | 多传感器 / 雷达-相机融合 | 5 |
| [`Quantization/`](Quantization) | 模型量化与压缩 | 8 |
| [`3D_Occupancy/`](3D_Occupancy) | 3D 语义占用预测 | 5 |
| [`Math_and_Notes/`](Math_and_Notes) | 数学与通用笔记 | 1 |
| [`assets/`](assets) | 论文插图与表格截图，按论文名分目录 | — |

## 经典检测

关于经典 2D / 点云 3D 目标检测与检测框架的笔记。

| 论文 | 出处 | 关键词 | 代码 |
| --- | --- | --- | --- |
| [CaDDN](Classical_Detection/CaDDN.md) | CVPR 2021 | 视锥神经网络、三线性插值 | [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) |
| [CornerNet](Classical_Detection/CornerNet.md) | ECCV 2018 | 关键点检测、角点池化、Associative Embedding、无锚框 | [princeton-vl/CornerNet](https://github.com/princeton-vl/CornerNet) |
| [CenterNet：Objects as Points](Classical_Detection/CenterNet%EF%BC%9AObjects%20as%20Points.md) | arXiv 2019 | 中心点检测、热力图、无锚框、无 NMS | [xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet) |
| [CenterNet：Keypoint Triplets](Classical_Detection/CenterNet%EF%BC%9AKeypoint%20Triplets.md) | ICCV 2019 | 关键点三元组、中心点验证、中心池化、级联角点池化 | [Duankaiwen/CenterNet](https://github.com/Duankaiwen/CenterNet) |
| [FCOS](Classical_Detection/FCOS.md) | ICCV 2019 | 无锚框检测、逐像素预测、Center-ness、FPN 多尺度 | [tianzhi0549/FCOS](https://github.com/tianzhi0549/FCOS) |
| [Fast R-CNN 和 Faster R-CNN](Classical_Detection/Fast%20R-CNN%20%E5%92%8C%20Faster%20R-CNN%20.md) | Fast R-CNN: ICCV 2015<br>Faster R-CNN: TPAMI 2017 | RoI Pooling、候选区域网络 (RPN)、锚框、多任务损失 | [ShaoqingRen/faster_rcnn](https://github.com/ShaoqingRen/faster_rcnn) |
| [PointNet & PointPillars](Classical_Detection/PointNet%20%26%20PointPillars.md) | CVPR 2017 / CVPR 2019 | 置换不变性、对称函数、Pillar 编码 | — |
| [R4Det](Classical_Detection/R4Det.md) | CVPR 2026 | 4D 雷达-相机融合、全景深度融合、可变形门控时序融合、动态精炼 | 未开源 |
| [SGDet3D](Classical_Detection/SGDet3D.md) | RA-L 2025 | 4D 毫米波雷达-相机融合、几何深度补全、语义雷达 PillarNet、面向对象的交叉注意力、特征解耦 | [shawnnnkb/SGDet3D](https://github.com/shawnnnkb/SGDet3D) |

## 多模态融合

关于融合多个传感器数据的笔记。

| 论文 | 出处 | 关键词 | 代码 |
| --- | --- | --- | --- |
| [BEVFusion](Multimodal_Fusion/Ali-BEVFusion.md) | NeurIPS 2022 | 多模态融合、BEV 空间、动态融合模块、激光雷达-摄像头融合 | [ADLab-AutoDrive/BEVFusion](https://github.com/ADLab-AutoDrive/BEVFusion) |
| [RCBEVDet](Multimodal_Fusion/RCBEVDet.md) | CVPR 2024 | 雷达-摄像头融合、BEV 特征提取、交叉注意力融合、RadarBEVNet | [VDIGPKU/RCBEVDet](https://github.com/VDIGPKU/RCBEVDet) |
| [RadarDistill](Multimodal_Fusion/RadarDistill.md) | CVPR 2024 | 知识蒸馏、雷达 3D 检测、跨模态对齐、LiDAR-Radar | [geonhobang/RadarDistill](https://github.com/geonhobang/RadarDistill) |
| [HGSFusion](Multimodal_Fusion/HGSFusion.md) | AAAI 2025 | 4D 毫米波雷达-相机融合、点云生成、深度同步、混合概率分布 | [garfield-cpp/HGSFusion](https://github.com/garfield-cpp/HGSFusion) |
| [EMC2](Multimodal_Fusion/EMC2.md) | ICCV 2025 | 混合专家模型、多模态融合、自适应调度、Jetson Orin | [LinshenLiu622/EMC2](https://github.com/LinshenLiu622/EMC2) |

## 量化

关于模型量化和压缩技术的笔记。

| 论文 | 出处 | 关键词 | 代码 |
| --- | --- | --- | --- |
| [MQBench](Quantization/MQBench.md) | NeurIPS 2021 | Benchmark、QAT、PTQ、计算图 | [ModelTC/MQBench](https://github.com/ModelTC/MQBench) |
| [RepQ-ViT](Quantization/RepQ-ViT.md) | ICCV 2023 | 视觉 Transformer (ViT)、训练后量化 (PTQ)、尺度重参数化、极低比特量化 (4-bit) | [zkkli/RepQ-ViT](https://github.com/zkkli/RepQ-ViT) |
| [PD-Quant](Quantization/PD-Quant.md) | CVPR 2023 | 训练后量化 (PTQ)、预测差异度量、分布修正、全局感知 | [hustvl/PD-Quant](https://github.com/hustvl/PD-Quant) |
| [QD-BEV](Quantization/QD-BEV.md) | ICCV 2023 | 量化感知训练、知识蒸馏、BEV 感知、渐进式量化 | [Niko-zyf/QD-BEV](https://github.com/Niko-zyf/QD-BEV) |
| [OmniQuant](Quantization/OmniQuant.md) | ICLR 2024 | LLM 量化、训练后量化 (PTQ)、可学习量化参数、LWC、LET | [OpenGVLab/OmniQuant](https://github.com/OpenGVLab/OmniQuant) |
| [VQ-Map](Quantization/VQ-Map.md) | NeurIPS 2024 | BEV 感知、语义地图构建、离散表示 (VQ-VAE)、Deformable Attention、跨模态对齐 (PV-BEV) | [Z1zyw/VQ-Map](https://github.com/Z1zyw/VQ-Map) |
| [QwT](Quantization/QwT.md) | CVPR 2025 | 线性补偿层 | [wujx2001/QwT](https://github.com/wujx2001/QwT) |
| [PTQAT](Quantization/PTQAT.md) | ICCV 2025 | 3D 感知网络、混合量化、PTQ+QAT、误差传播、自动驾驶部署 | 未开源 |

## 3D 占用预测

关于语义占用预测与雷达占用预测的笔记。

| 论文 | 出处 | 关键词 | 代码 |
| --- | --- | --- | --- |
| [SurroundOcc](3D_Occupancy/SurroundOcc.md) | ICCV 2023 | 3D 占用预测、空间注意力、密集真值生成 | [weiyithu/SurroundOcc](https://github.com/weiyithu/SurroundOcc) |
| [EchoFusion](3D_Occupancy/EchoFusion.md) | NeurIPS 2023 | 原始雷达数据融合、极坐标 BEV、极坐标对齐注意力 (PAA)、3D 目标检测 | [tusen-ai/EchoFusion](https://github.com/tusen-ai/EchoFusion) |
| [SparseOcc](3D_Occupancy/SparseOcc.md) | CVPR 2024 | 语义占用预测、稀疏潜变量表示、稀疏卷积、场景补全 | [VISION-SJTU/SparseOcc](https://github.com/VISION-SJTU/SparseOcc) |
| [NJU-SparseOcc](3D_Occupancy/NJU-SparseOcc.md) | ECCV 2024 | 全稀疏架构、占据预测、掩码引导稀疏采样、RayIoU、时序建模 | [MCG-NJU/SparseOcc](https://github.com/MCG-NJU/SparseOcc) |
| [RadarOcc](3D_Occupancy/RadarOcc.md) | NeurIPS 2024 | 4D 成像雷达、3D 占据预测、4D 雷达张量 (4DRT)、旁瓣感知、球坐标编码 | [Toytiny/RadarOcc](https://github.com/Toytiny/RadarOcc) |

> `SparseOcc`（CVPR 2024, VISION-SJTU）与 `NJU-SparseOcc`（ECCV 2024, MCG-NJU）是同名但不同的工作，引用时注意区分。

## 数学与通用笔记

| 笔记 | 关键词 |
| --- | --- |
| [刚体旋转表示法的深度解析](Math_and_Notes/%E5%9B%9B%E5%85%83%E6%95%B0.md) | 旋转矩阵、李群 SO(3)、四元数算法 |

## 📌 素材与待整理

`assets/` 下按论文名分目录存放插图与表格截图（命名如 `Figure3.png` / `Table5.png`，便于与论文对照）。以下目录已收集素材但**尚未成文**，欢迎按上面的格式补齐：

**VLA / 多模态大模型**：`LLaVA`、`OpenVLA`、`OpenDriveVLA`、`OneVL`、`Qwen3-VL`、`Alpamago`、`COTR`、`DrivePI`

**轻量骨干与压缩**：`MobileNet`、`MobileNetV2`、`MobileV3`、`MobileOne`、`ShuffleNet`、`LSQ`、`Deep Compression`、`BRCEQ`

**BEV 与多模态融合**：`BEVDet`、`MIT-BEVFusion`、`MSMDFusion`、`RCBEVDet++`、`PillarNet`

**参数高效微调**：`QLoRA`

**感知基础 / 其它**：`KITTI`、`CameraParams`、`Misc`

## 📝 笔记约定

- **头部三行**：`**出处会议：**`（期刊类用 `**出处期刊：**`）、`**是否开源：**`、`**关键词：**`，便于横向检索与 SOTA 对比表格引用。
- **命名**：一篇论文一个文件，文件名用论文原名；论文 PDF 与插图统一放 `assets/<论文名>/`。
- **层级**：单个一级标题 + 二级标题组织小节，便于在大纲面板中展开。

## 🕘 更新记录

| 日期 | 内容 |
| --- | --- |
| 2026-09-23 | 重建索引：补齐 3D 占用预测、R4Det、SGDet3D、PTQAT、VQ-Map 等此前漏收录的 13 篇笔记，新增素材待整理清单 |
