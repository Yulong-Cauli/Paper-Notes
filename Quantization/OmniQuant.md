# OmniQuant

**出处会议：** ICLR 2024  
**是否开源：** 是，https://github.com/OpenGVLab/OmniQuant  
**关键词：** LLM量化、训练后量化(PTQ)、可学习量化参数、LWC、LET

---

## 1. 概述

OmniQuant 冻结原始模型参数，引入少量**可学习的量化参数 (Learnable Quantization Parameters)**，通过梯度下降在少量数据上优化这些参数，从而在极低比特下实现高性能。

**保留 PTQ 的效率，引入 QAT 的梯度优化。**

---

## 2. 方法

OmniQuant 的优化框架是 **Block-wise Error Minimization (分块误差最小化)**。
它不是一次性优化整个网络，而是一个 Transformer Block 接一个 Block 地优化。

**优化目标公式：**
$$
\arg\min_{\Theta_1, \Theta_2} || Y_{FP} - Y_Q(\Theta_1, \Theta_2) ||^2
$$

*   $Y_{FP}$：全精度 Block 的输出。
*   $Y_Q$：量化后 Block 的输出。
*   $\Theta_1$：LWC 参数（针对权重）。
*   $\Theta_2$：LET 参数（针对激活）。

---

### 2.1 组件一：LWC (Learnable Weight Clipping) —— 可学习权重裁剪

#### 核心逻辑
传统的 Min-Max 量化直接取权重的最大/最小值作为边界，容易受离群值 (Outliers) 影响，拉大量化步长，导致大部分正常数值精度丢失。LWC 认为**最佳裁剪阈值**不应该是固定的，而应该是**通过梯度学习出来的**。

#### 数学公式推导
假设原始权重为 $W$，量化比特数为 $N$。
LWC 引入两个可学习的比例因子：$\gamma$ (控制上界) 和 $\beta$ (控制下界)。

1.  **确定裁剪边界 (Clipping Bounds)：**
    $$
     Upper = \gamma \cdot \max(W), \quad Lower = \beta \cdot \min(W) 
    $$
    *注：$\gamma, \beta$ 通过 Sigmoid 函数限制在 $(0, 1)$ 之间，即只缩小范围，不扩大。*
    
2.  **计算量化步长 (Step Size $h$)：**
    $$
    h = \frac{Upper - Lower}{2^N - 1}
    $$
    
3.  **计算零点 (Zero Point $z$)：**
    $$
    z = - \text{Round}(\frac{Lower}{h})
    $$
    
4.  **执行量化 (Quantization)：**
    $$
     W_q = \text{Clamp}\left( \text{Round}\left(\frac{W}{h}\right) + z, \ 0, \ 2^N - 1 \right) 
    $$
    

**总结：** LWC 让模型自动学会在哪里“剪一刀”，舍弃极少数离群值，换取 99% 参数的高精度。

---

### 2.2 组件二：LET (Learnable Equivalent Transformation) —— 可学习等价变换

#### 核心逻辑
LLM 的激活值 (Activation) 存在难以量化的**离群值**。LET 利用数学等价性，通过**缩放 (Scaling)** 和 **偏移 (Shifting)**，将激活值的离群压力转移给权重。

#### 场景 1：Linear 层变换
对于线性层 ，LET 引入缩放因子 $s$ 和偏移因子 $\delta$（均为可学习向量）：

$$
Y = XW + B = \underbrace{[(X - \delta) \oslash s]}_{\tilde{X}} \cdot \underbrace{[s \odot W]}_{\tilde{W}} + \underbrace{[B + \delta W]}_{\tilde{B}}
$$

*   **$\tilde{X}$ (新激活)**：先减 $\delta$ 再除 $s$。激活值变平滑，易于量化。
*   **$\tilde{W}$ (新权重)**：乘以 $s$。虽然变大了，但交给 LWC 处理。
*   符号说明：$\oslash$ 为逐元素除，$\odot$ 为逐元素乘。

#### 场景 2：Attention 模块变换 (Q/K/V)
对于注意力矩阵 $P = \text{Softmax}(QK^T)$，LET 引入缩放因子 $s_a$：

$$
P = \text{Softmax}\left( \underbrace{(Q \oslash s_a)}_{\tilde{Q}} \cdot \underbrace{(s_a \odot K^T)}_{\tilde{K}^T} \right)
$$

*   **目的**：平衡 Q 和 K 的分布，降低 KV Cache 的量化难度。

---

## 3. 深度解析

### 3.1 激活离群值与权重的关系 ("质量守恒")
*   **现象**：激活值 $X$ 中存在巨大的离群值（如 100），导致量化范围被撑大，小数值（0.1）精度丢失。
*   **操作**：LET 将激活值除以 $s$（如 $s=100$），同时将权重乘以 $s$。
    $$ Y = (X \div 100) \times (W \times 100) $$
*   **结果**：
    *   **激活值**：100 变成 1，离群值消失，变得**极其容易量化**。
    *   **权重**：原本的范围扩大了 100 倍，变得**更难量化**。
*   **为什么这么做？** 这是一个**“转移痛苦”**的过程。激活值是动态的，很难处理；权重是静态参数，我们有 **LWC** 这个强力工具来压制变大后的权重。总体收益 > 代价。

### 3.2 LWC 的作用范围
*   **LWC 仅用于权重 (Weights)**：因为权重是固定的，可以训练固定的裁剪阈值 $\gamma, \beta$。
*   **激活值如何量化？** 在经过 **LET** 变换（压平离群值）后，激活值分布变得很规范，直接使用最简单的 **Min-Max 量化** 即可获得很好的效果。

### 3.3 LET 的“零成本推理”魔法 (Zero-Cost Inference)
LET 在训练时引入了除法、减法、乘法，但在推理时这些操作**全部消失**，通过**参数融合 (Fusion)** 实现。

**1. 权重的融合 (Weight Fusion)**

*   训练得到的 $s$ 是常数。
*   **操作**：直接计算 $W' = W \odot s$，并保存到模型文件中。
*   **推理**：直接加载新权重 $W'$。

**2. 激活的融合 (Activation Fusion) —— 重点**
激活变换公式为 $\tilde{X} = (X - \delta) / s$。
输入 $X$ 通常来自上一层的 LayerNorm：
$$
 X = \text{LayerNorm}(Input) = \frac{Input - \mu}{\sigma} \cdot \gamma_{LN} + \beta_{LN} 
$$


将 LayerNorm 代入变换公式：
$$
 \tilde{X} = \frac{(\frac{Input - \mu}{\sigma} \cdot \gamma_{LN} + \beta_{LN}) - \delta}{s} 
$$


$$
\tilde{X} = \frac{Input - \mu}{\sigma} \cdot \mathbf{\frac{\gamma_{LN}}{s}} + \mathbf{\frac{\beta_{LN} - \delta}{s}}
$$

**结论**：这依然是一个 LayerNorm 的形式！我们只需要修改原 LayerNorm 的参数：
*   **新参数 $\gamma'_{LN} = \gamma_{LN} / s$**
*   **新参数 $\beta'_{LN} = (\beta_{LN} - \delta) / s$**

**推理时**：模型结构不变，只需加载修改后的 LayerNorm 参数，即可自动完成 LET 变换。**这就是零成本推理。**

---

## 4. 实验与分析 (Experiments)

### 实验 1：W2A16 (极限权重量化)
**场景**：2-bit 权重，16-bit 激活。这是对权重量化的极致考验。
**对比**：RTN, GPTQ, AWQ, OmniQuant。
**指标**：Perplexity (PPL，困惑度)，越低越好。

**表格：LLaMA-13B W2A16 PPL 对比**

| 方法          | Perplexity (困惑度) | 评价                     |
| :------------ | :------------------ | :----------------------- |
| **RTN**       | 6.8e4 (68000+)      | **完全失效**，输出乱码   |
| **GPTQ**      | 44.01               | **不可用**，精度损失巨大 |
| **AWQ**       | 2.6e5               | **失效**                 |
| **OmniQuant** | **9.72**            | **可用**，接近实用水平   |

**分析**：在 2-bit 这种极端条件下，死板的规则（GPTQ/AWQ）完全无法处理权重的精度损失。**LWC** 通过学习最佳裁剪点，挽救了模型精度。

---

### 实验 2：W4A4 (全量化)
**场景**：4-bit 权重，4-bit 激活。这是目前部署最关注的配置，激活量化是难点。
**对比**：SmoothQuant (SQ), LLM-QAT。
**指标**：LLaMA-7B 零样本任务平均准确率 (Accuracy)。

**表格：LLaMA-7B W4A4 准确率对比**

| 方法            | 平均准确率 (Avg Acc) | 评价                         |
| :-------------- | :------------------- | :--------------------------- |
| **FP16 (基准)** | 64.09%               | 原始模型                     |
| **SmoothQuant** | 38.41%               | **崩塌**，激活离群值未处理好 |
| **LLM-QAT**     | 41.27%               | 昂贵的训练也未能救回         |
| **OmniQuant**   | **52.65%**           | **显著提升** (+14%)          |

**分析**：W4A4 的瓶颈在于激活。OmniQuant 依靠 **LET** 的可学习参数，比 SmoothQuant 的人工规则更有效地平抑了激活离群值，甚至超越了昂贵的 QAT。

---

### 实验 3：消融实验 (Ablation Study)
**目的**：验证 LWC 和 LET 到底谁在起作用。
**场景**：LLaMA-7B W4A4。

**表格：组件贡献分析**

| 配置                      | PPL (困惑度) | 分析                                |
| :------------------------ | :----------- | :---------------------------------- |
| **SmoothQuant (基准)**    | 28.78        | 效果差                              |
| **仅加 LET**              | 16.97        | **提升巨大**，说明 LET 对 W4A4 关键 |
| **仅加 LWC**              | 15.80        | 有提升                              |
| **OmniQuant (LET + LWC)** | **12.87**    | **1+1 > 2**，两者协同效果最好       |

**分析**：LET 解决了激活问题，LWC 解决了权重问题，两者结合实现了最佳效果。

---

### 实验 4：真实推理加速
**场景**：NVIDIA A100-80G，使用 MLC-LLM 部署。
**模型**：LLaMA-13B。

**表格：W4A16 量化 vs. 原始模型**

| 指标         | 原始 FP16    | OmniQuant W4A16   | 提升           |
| :----------- | :----------- | :---------------- | :------------- |
| **显存占用** | ~26 GB       | **7.0 GB**        | **节省 ~73%**  |
| **推理速度** | ~40 tokens/s | **91.3 tokens/s** | **加速 > 2倍** |

**分析**：验证了“零成本推理”的有效性。参数融合后，模型体积大幅减小，且无需额外算子，直接转化为推理速度的提升。

---

## 5. 总结

OmniQuant 是一篇兼具**理论深度**和**工程价值**的论文。
1.  **理论上**：证明了通过梯度下降学习“量化参数”（裁剪阈值 & 等价变换系数），可以在不微调原始权重的前提下，达到接近 QAT 的效果。
2.  **方法上**：**LWC** 解决了极低比特权重的精度问题，**LET** 解决了激活离群值的问题。
3.  **工程上**：实现了**零成本推理**，对实际部署（如手机端、边缘端 LLM）具有极高的实用价值。



## 6. 附录详解 (Appendix Deep Dive)

附录部分补充了算法实现的细节、与同类方法的深度对比以及更多消融实验，进一步验证了 OmniQuant 的鲁棒性。

### 6.1 核心算法伪代码 (Algorithm)
附录 A1 提供了 OmniQuant 的整体训练流程，核心逻辑是**交替优化**。

*   **流程**：
    1.  **分块 (Block-wise)**：逐个 Transformer Block 进行处理。
    2.  **初始化**：
        *   LWC 参数 ($\gamma, \beta$) 初始化为 1 (不裁剪)。
        *   LET 参数 ($s, \delta$) 基于 SmoothQuant 的统计值初始化。
    3.  **循环优化 (Epochs)**：
        *   在前向传播中，先应用 LET (变换激活和权重)，再应用 LWC (量化权重)。
        *   计算量化输出与全精度输出的 MSE Loss。
        *   反向传播，同时更新 LWC 和 LET 的参数。
    4.  **参数融合**：训练结束后，将学到的参数融合进权重和上一层 LayerNorm，得到最终量化模型。

### 6.2 与其他“等价变换”方法的深度对比 (Appendix A2)
OmniQuant 的 LET 并不是唯一的等价变换方法，附录详细对比了它与 SmoothQuant (SQ)、AWQ、Outlier Suppression+ (OS+) 的区别。

| 特性         | SmoothQuant    | AWQ                    | Outlier Suppression+ | **OmniQuant (LET)**               |
| :----------- | :------------- | :--------------------- | :------------------- | :-------------------------------- |
| **变换操作** | 仅缩放 (Scale) | 仅缩放                 | 缩放 + 偏移 (Shift)  | **缩放 + 偏移**                   |
| **应用位置** | 仅 Linear 层   | 仅 Linear 层           | 仅 Linear 层         | **Linear 层 + Attention (Q/K)**   |
| **参数来源** | 人工设计规则   | 网格搜索 (Grid Search) | 网格搜索 + 部分人工  | **全梯度下降 (Gradient Descent)** |
| **适用场景** | W8A8           | 仅 Weight-only         | W8A8 / W6A6          | **全能 (Weight-only & W4A4)**     |

**核心优势**：OmniQuant 是唯一一个把 Attention 模块也纳入变换，并且**完全通过梯度优化**参数的方法，搜索空间最大，效果最好。

### 6.3 更多消融实验与分析

#### (1) LET 的组件拆解 (Appendix A3 - A5)
*   **注意力模块 (Attention) 的变换有用吗？**
    *   有用，但相比 Linear 层贡献较小。原因是 Attention (Q/K) 中的离群值不如 Linear 层激活值那么极端。
*   **偏移 (Shift) 有用吗？**
    *   非常关键。相比仅做缩放，增加偏移能更好地处理非对称分布的激活值（如 ReLU 后的正数分布）。
*   **应用位置消融**：
    *   论文发现，**LayerNorm 之后的第一层 Linear** (q_proj, k_proj, v_proj) 的离群值最严重，在这里应用 LET 收益最大。

#### (2) 与 LSQ、PACT 的对比 (Appendix A6)
LSQ 和 PACT 是经典的量化感知训练 (QAT) 方法，也会学习裁剪阈值。
*   **OmniQuant vs. LSQ/PACT**：
    *   在 W4A4 设置下，直接套用 LSQ/PACT 效果不如 LWC。
    *   **原因**：LET 在不断调整激活和权重的分布（动态变化），LSQ/PACT 学习的是**绝对截断值**，很难适应这种动态变化。而 LWC 学习的是**比例系数 ($\gamma$)**，对分布变化更鲁棒，收敛更稳。

#### (3) 训练时间与数据量 (Appendix A3)
*   **数据量**：实验证明，仅需 **128 个样本** 即可收敛。增加到 256 个样本收益极小。
*   **训练时间**：
    *   LLaMA-7B：W4A16 约 1.1 小时，W4A4 约 1.6 小时。
    *   LLaMA-65B：W4A16 约 9 小时，W4A4 约 14.4 小时。
    *   结论：相比需要数百 GPU 小时的 QAT，OmniQuant 极其高效。

### 6.4 补充实验结果 (Appendix A8)

#### (1) 巨型模型 Falcon-180B
*   在 180B 参数的模型上，OmniQuant 在 W3A16 配置下依然保持了极高的精度，甚至在某些任务上反超了 RTN 的 W4A16 效果。证明了方法的可扩展性。

#### (2) 非对称比特量化 (Asymmetric Bits)
*   论文测试了 **W4A8** (4-bit 权重，8-bit 激活) 和 **W8A4**。
*   **结论**：**W4A8 的效果远好于 W8A4**。这再次印证了 LLM 的特性：**激活值比权重更难量化**。即使给激活 4-bit，给权重 8-bit，效果依然很差；反之则很好。

#### (3) Softmax 量化 (Appendix A3)
*   OmniQuant 尝试了对 Attention 中的 Softmax 输出进行量化。
*   **结论**：8-bit 和 6-bit 的 Softmax 量化几乎无损，但 **4-bit Softmax 会导致精度严重下降**。这说明 Softmax 的长尾分布极其敏感，目前还需要保留较高精度 (通常保留 FP16 或 Int8)。





# BEVFusion网络可以用吗？

**可以，但需要适配。** OmniQuant 的核心思想（LWC 和 LET）在理论上完全适用于 BEVFusion 这类多模态 3D 检测模型，特别是针对其中 **Transformer 架构的部分**（如 Swin-Transformer 主干或 Transformer 检测头）效果会最好。

但是，BEVFusion 和纯文本 LLM 有显著的结构差异（卷积、稀疏卷积、BEV Pooling），直接“生搬硬套”会遇到一些特有的挑战。

我们来详细拆解一下，如果要把 OmniQuant 用在 BEVFusion 上，哪些能用，哪些需要改。

---

### 1. **LWC (可学习权重裁剪) —— 几乎可以无脑用**

**适用性：⭐⭐⭐⭐⭐ (非常高)**

*   **原理通用性：** LWC 处理的是“权重的长尾分布”问题。无论是 LLM 里的 `Linear` 层，还是 BEVFusion 里的 `Conv2d` (图像分支) 或 `SparseConv` (激光雷达分支)，权重矩阵里都有离群���。
*   **迁移策略：**
    *   对于 **Conv2d**：OmniQuant 的 LWC 公式可以直接套用。唯一的区别是 Linear 层的权重是二维的 $(C_{out}, C_{in})$，而卷积层权重是四维的 $(C_{out}, C_{in}, K, K)$。你只需要在计算 Max/Min 时注意维度规约即可（通常是对 Channel 维度做裁剪）。
    *   对于 **SparseConv (稀疏卷积)**：这是 LiDAR 分支的核心。稀疏卷积的权重本质上也是矩阵乘法，LWC 同样适用。
*   **价值：** BEVFusion 部署时通常也追求 W4A4 或 W8A8，LWC 能显著保护权重精度，特别是对于 LiDAR 分支这种对数值敏感的模块。

---

### 2. **LET (可学习等价变换) —— 需要针对性适配**

**适用性：⭐⭐⭐ (中等，视模块而定)**

LET 的核心是利用 $Y = (X/s) \cdot (sW)$ 将激活值的难度转移给权重。这在 Transformer 结构中非常有效，但在 BEVFusion 的不同部分表现不同：

#### A. 图像主干 (Image Backbone)

*   如果你的 BEVFusion 用的是 **Swin-Transformer** 或 **ViT**：
    *   **完美适用**。这部分和 LLM 结构几乎一样，LET 可以直接用来处理 Attention 和 Linear 层的激活离群值。
*   如果用的是 **ResNet (CNN)**：
    *   **可以适配**。CNN 中也有激活离群值（通常在 ReLU 之后）。LET 的数学原理依然成立。
    *   **零成本推理：** 在 LLM 里，LET 的参数融合进了 LayerNorm。在 CNN 里，LET 的缩放参数 $s$ 和偏移 $\delta$ 可以**融合进 BatchNorm (BN)** 层。这在 CNN 量化里叫 "BN Folding"，完全可行。

#### B. 激光雷达主干 (LiDAR Backbone - SparseConv)

*   **难点：** 稀疏卷积的激活值大部分是 0（空的空间）。
*   **适配：** 这里的激活离群值分布可能与稠密网络不同。LET 如果强行对所有通道做 Shifting（偏移），可能会破坏“稀疏性”（把原本是 0 的空区域变成非 0，导致计算量爆炸）。
*   **建议：** 在稀疏卷积层，**慎用 Shifting ($\delta$)**，可以只保留 Scaling ($s$)，或者必须保证 Shifting 操作仅作用于非空体素（Active Voxels）。

#### C. **关键瓶颈：BEV Pooling (融合模块)**

这是 BEVFusion 最特殊的地方，也是 OmniQuant **最难直接套用**的地方。

*   **原理：** BEV Pooling 把图像特征投影到 3D 空间。这是一个几何操作，不是简单的矩阵乘法。
*   **问题：** 如果你在 Pooling 之前用 LET 对图像特征做了 Scaling ($X/s$)，你需要确保后续的融合操作能感知这个 $s$。
    *   如果 LiDAR 特征没有做同样的缩放，那么直接 Concat（拼接）或者 Add（相加）就会导致**特征量级不匹配**，融合效果直接崩盘。
*   **解决方案：** 你必须在融合（Fusion）发生之前，把 LET 的 Scaling 效果**逆转回去（De-quantize）**，或者强制让 LiDAR 分支和 Camera 分支学习**同一个** Scaling Factor（但这很难收敛）。

---

### 3. Block-wise Optimization (分块优化) —— 依然是神技

BEVFusion 虽然结构复杂，但也是模块化的（Backbone -> Neck -> Head）。

*   **OmniQuant 的分块策略完全适用。** 你可以先冻住其他部分，只优化 Image Backbone，再优化 LiDAR Backbone，最后优化 Head。
*   **数据问题：** LLM 用 WikiText 只有几千个 Token。BEVFusion 需要用 nuScenes 数据集。因为是 PTQ（训练后量化），你只需要几百帧校准数据（Calibration Data），���如 100-200 个场景，就能在几个小时内跑完量化参数搜索。这比重新训练（QAT）要快得多。

---

### 总结：如何在 BEVFusion 上落地 OmniQuant？

如果你想做这个实验，我建议的路线图是：

1.  **第一阶段（低垂的果实）：**
    *   只在 **检测头（Detection Head）** 和 **Swin-Transformer 主干** 上应用 OmniQuant（LWC + LET）。
    *   这部分全是 Dense 计算，和 LLM 最像，收益最稳。

2.  **第二阶段（攻坚）：**
    *   在 **Conv/SparseConv** 权重上应用 **LWC**。
    *   这能让你在 W4A4 下保住 LiDAR 特征的精度。

3.  **第三阶段（小心地雷）：**
    *   **不要在 BEV Fusion 层即刻前后** 使用复杂的 LET 变换，除非你非常清楚特征对齐的关系。
    *   保持 Fusion 层的输入是经过反量化（Dequantized）回到 FP16/Int8 统一量级的，或者确保两路分支缩放比例一致。

**一句话总结：**
LWC 是通用的“万金油”，BEVFusion 肯定能用；LET 是“精密手术刀”，在 BEVFusion 的 Transformer 部分好用，但在 CNN 和融合层需要小心修改。
