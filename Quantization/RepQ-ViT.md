# RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers

**出处会议：** ICCV 2023  
**是否开源：** https://github.com/zkkli/RepQ-ViT  
**关键词：** 视觉Transformer (ViT)，训练后量化 (PTQ)，尺度重参数化 (Scale Reparameterization)，极低比特量化 (4-bit)

---

## 1. 概述

本文直击 ViT 在极低比特（如 4-bit）量化时精度崩溃的痛点。

ViT 中存在两个具有“极端分布”的激活值：**LayerNorm 后的剧烈通道间波动** 和 **Softmax 后的极端幂律分布**。

传统 PTQ 方法受限于硬件部署要求，在量化阶段就强行使用简单的硬件友好量化器（如逐层量化、$\log_2$ 量化），导致严重的量化误差。

RepQ-ViT 提出了一种全新的**“量化-推理解耦范式”**：在量化阶段使用**复杂的量化器**（保精度），在推理阶段通过**尺度重参数化（Scale Reparameterization）**将其无缝转换为**简单的量化器**（保速度）。该方法无需超参数调节和昂贵的重建过程，在 4-bit 下首次将 ViT 的精度提升到了可用水平。

<div align="center"><img src="https://raw.githubusercontent.com/Yulong-Cauli/Paper-Notes/main/assets/RepQ-ViT/Figure1.jpeg" alt="Overview of RepQ-ViT" width="60%"></div>

---

## 2. 方法：两大尺度重参数化魔法

RepQ-ViT 的核心在于用严谨的数学等价/近似变换，将复杂的量化参数“揉”进网络的前后层结构中。

### 2.1 针对 LayerNorm 的重参数化：逐通道 $\rightarrow$ 逐层

**痛点：** LayerNorm 后的激活值在不同通道间差异巨大，必须用**逐通道量化 (Channel-wise)** 才能保精度，但这在硬件激活运算上不被支持，硬件仅支持**逐层量化 Layer-wise**。

<div align="center"><img src="https://raw.githubusercontent.com/Yulong-Cauli/Paper-Notes/main/assets/RepQ-ViT/Figure2.jpeg" alt="Overview of RepQ-ViT" width="60%"></div>

**解决思路：**
假设原逐通道的比例尺为 $s$，零点为 $z$ ，$s$ 和 $z$ 是形状为 `[D]` 的一维数组（长度等于通道数）。目标是转换为统一的逐层参数 $\tilde{s}$ 和 $\tilde{z}$ 。
定义变异因子：倍数关系 $r_1 = s/\tilde{s}$，差值关系 $r_2 = z - \tilde{z}$。

原本我们输入的激活值是 $X'$ ，是一个形状为 `[N, D]` 的二维矩阵。

在原量化公式中，量化过程大致是：

$$
Q = \text{round}(\frac{X'}{s}) + z
$$

把我们第一步的关系代进去：

$$
Q = \text{round}(\frac{X'}{\tilde{s} \cdot r_1}) + \tilde{z} + r_2
$$

把 $r_2$ 放进括号里，因为当 $b$ 是整数时，有 $\text{round}(a) + b = \text{round}(a+b)$ ，且 零点 $z$ 定义为整数：

$$
Q = \text{round}(\frac{X'}{\tilde{s} \cdot r_1} + r_2) + \tilde{z}
$$

通分一下：

$$
Q = \text{round}(\frac{X' + \tilde{s} \cdot r_1 \cdot r_2}{\tilde{s} \cdot r_1}) + \tilde{z}
$$

因为 $\tilde{s} \cdot r_1 = s$，代进去，写成标准的逐层量化的形式：

$$
Q = \text{round}(\frac{\frac{X' + s \cdot r_2}{r_1}}{\tilde{s}}) + \tilde{z}
$$

这样，我们直接对 $\widetilde{X}' = \frac{X' + s \cdot r_2}{r_1}$ 做逐层量化，就等于对原来的 $X'$ 做逐通道量化！

我们怎么把 $X'$ 变成 $\widetilde{X}'$ 呢？直接改它前面的 LayerNorm 的参数就行了。

LayerNorm 的原公式是： 

$$
X' = \text{Norm}(X) \cdot \gamma + \beta
$$
我们想要的输出是 $\widetilde{X}' = \frac{X' + s \cdot r_2}{r_1}$，把 $X'$ 代进去：

$$
\widetilde{X}' = \frac{(\text{Norm}(X) \cdot \gamma + \beta) + s \cdot r_2}{r_1}
$$

把除以 $r_1$ 拆开：

$$
\widetilde{X}' = \text{Norm}(X) \cdot \left(\frac{\gamma}{r_1}\right) + \left(\frac{\beta + s \cdot r_2}{r_1}\right)
$$


对比原来的 LayerNorm 公式，**新的参数诞生了！**

$$
\tilde{\gamma} = \frac{\gamma}{r_1}\\
\tilde{\beta} = \frac{\beta + s \cdot r_2}{r_1}
$$

虽然我们成功骗过了中间的量化过程，但原本给 Linear层 输入的是 $X'$，现在变成了 $\widetilde{X}'$，下一层的结果会变错。所以我们要修改下一层的权重来抵消。

下一层的原计算是：

$$
Y = X' \cdot W + b
$$

由前面知道：$X' = \widetilde{X}' \cdot r_1 - s \cdot r_2$，我们把它代入 $Y$：

$$
Y = (\widetilde{X}' \cdot r_1 - s \cdot r_2) \cdot W + b
$$

把括号展开：

$$
Y = \widetilde{X}' \cdot (r_1 \cdot W) - (s \cdot r_2) \cdot W + b
$$

整理一下，把后面两项合并：

$$
Y = \widetilde{X}' \cdot \underbrace{(r_1 \cdot W)}_{\text{新权重 } \widetilde{W}} + \underbrace{(b - s \cdot r_2 \cdot W)}_{\text{新偏置 } \tilde{b}}
$$

这就得出了**新的下一层参数：**

$$
\widetilde{W} = r_1 \cdot W \\ \tilde{b} = b - s \cdot r_2 \cdot W
$$

#### 结论

1. **构造等效的伪激活值**：
   为了让后续量化器能用统一的 $\tilde{s}, \tilde{z}$，我们需要把输入 $X'$ 偷偷替换为 $\widetilde{X}'$：
   
   $$
   \widetilde{X}' = \frac{X' + s \odot r_2}{r_1}
   $$

2. **向前吸收（修改 LayerNorm 参数）**：
   将上述变换融合进前置的 LayerNorm 的仿射参数 $\gamma, \beta$ 中，得到新的部署参数：
   
   $$
   \widetilde{\gamma} = \frac{\gamma}{r_1}, \quad \widetilde{\beta} = \frac{\beta + s \odot r_2}{r_1}
   $$

3. **向后补偿（修改下一层 Linear 的权重）**：
   由于输入给下一层的内容变了，为了保证最终输出不变，必须修改下一层（QKV投影层）的权重 $W^{qkv}$ 和偏置 $b^{qkv}$：
   
   $$
   \widetilde{W}^{qkv} = r_1 \odot W^{qkv}, \quad \widetilde{b}^{qkv} = b^{qkv} - (s \odot r_2) W^{qkv}
   $$
   
   *(注：由于这里用 $r_1$ 放缩了权重 $W$，改变了其分布，因此 $\widetilde{W}$ 需要在极少量校准数据上重新校准一下量化参数，这会带来极微小的精度损失，但换来了硬件的完美支持。)*

---

### 2.2 针对 Softmax 的重参数化：$\log_{\sqrt{2}}$ $\rightarrow$ $\log_2$

**痛点：** Attention 矩阵呈幂律分布，大部分值接近0，极少数大值非常关键。硬件友好的 $\log_2$ 量化器分辨率不足，会将大量关键分数粗暴截断。底数为 $\sqrt{2}$ 的量化器精度高，但**不支持硬件移位加速（Bit-shifting）**。

<div align="center"><img src="https://raw.githubusercontent.com/Yulong-Cauli/Paper-Notes/main/assets/RepQ-ViT/Figure3.jpeg" alt="Overview of RepQ-ViT" width="60%"></div>

**解决思路：**
利用对数与指数的数学性质，将 $\log_{\sqrt{2}}$ 强行拆解为纯整数的 $\log_2$ 和位移操作。

1. **量化阶段**：
   $$
   A^{(\mathbb{Z})} = \text{clip}\left(\left\lfloor -\log_{\sqrt{2}} \frac{A}{s} \right\rceil, \dots\right) = \text{clip}\left(\left\lfloor -2\log_2 \frac{A}{s} \right\rceil, \dots\right)
   $$
   
2. **反量化阶段（奇偶性拆解）**：
   原本的反量化公式为 $\hat{A} = s \cdot 2^{-\frac{A^{(\mathbb{Z})}}{2}}$。
   此时指数 $-\frac{A^{(\mathbb{Z})}}{2}$ 可能不是整数，无法使用硬件移位。作者引入向下取整和奇偶指示函数 $\mathbb{1}(\cdot)$（偶数为0，奇数为1）：
   $$
   \hat{A} = s \cdot 2^{\lfloor -A^{(\mathbb{Z})}/2 \rfloor} \cdot \left[ \mathbb{1}(A^{(\mathbb{Z})}) \cdot (\sqrt{2} - 1) + 1 \right]
   $$
   将后面那一坨多出来的常数直接吸收到全新的比例尺 $\tilde{s}$ 中：
   $$
   \tilde{s} = s \cdot \left[ \mathbb{1}(A^{(\mathbb{Z})}) \cdot (\sqrt{2} - 1) + 1 \right]
   $$
   最终反量化变成 $\hat{A} = \tilde{s} \cdot 2^{\text{整数}}$。

---

## 3. 实验结果与深度分析

### 3.1 极限 4-bit 图像分类 (ImageNet)

在 W4A4 的极限设置下，此前的方法全面崩盘，RepQ-ViT 实现了“起死回生”。

| **方法 (W4A4)**     | **No HP** | **No REC** | **ViT-B** | **DeiT-S** | **Swin-S** |
| :------------------ | :-------: | :--------: | :-------: | :--------: | :--------: |
| FP32 (基准)         |     -     |     -      |   84.54   |   79.85    |   83.23    |
| FQ-ViT              |     ×     |     ✓      |   0.10    |    0.10    |    0.10    |
| PTQ4ViT             |     ×     |     ×      |   30.69   |   34.08    |   76.09    |
| APQ-ViT             |     ×     |     ×      |   41.41   |   43.55    |   77.15    |
| **RepQ-ViT (Ours)** |   **✓**   |   **✓**    | **68.48** | **69.03**  | **79.45**  |

**分析**：在无超参 (No HP) 且无昂贵重建 (No REC) 的纯 PTQ 设定下，RepQ-ViT 在 DeiT-S 上将准确率从 43.55% 暴涨至 69.03% (+25.48%)，首次将 ViT 的 4-bit 量化推向可用水平。


### 3.2 消融实验

验证重参数化的实际收益（以 DeiT-S 为例）。

**LayerNorm 激活量化消融：**

*   纯逐层量化 (Layer-wise)：33.17%（崩塌）
*   纯逐通道量化 (Channel-wise)：70.28%（精度高，但硬件不支持）
*   **Scale Reparam (Ours)**：**69.03%**（硬件完美支持，精度仅因权重重新校准损失 1.25%）

**Softmax 激活量化消融：**

*   普通的 $\log_2$ 量化：67.71%
*   完美的 $\log_{\sqrt{2}}$ 量化：69.03%
*   **Scale Reparam (Ours)**：**69.03%**（100% 数学等价，精度无损转移到硬件位移运算）

### 3.3 部署与校准效率 (Efficiency)

| **方法**     | **校准数据量** | **耗时 (单卡 3090)** |
| :----------- | :------------- | :------------------- |
| FQ-ViT       | 1000 张        | 0.5 分钟             |
| PTQ4ViT      | 32 张          | 3.2 分钟             |
| **RepQ-ViT** | **32 张**      | **1.3 分钟**         |

- **分析**：因为没有任何梯度重建和复杂的超参搜索，RepQ-ViT 只需要 32 张校准图，在 1 分钟出头即可完成一整个 ViT 模型的量化，极其契合工业界快速落地的需求。





没问题！这绝对是最能把底层逻辑理顺的方式。

为了最直观地展示，我们以论文中最硬核的 **4-bit 量化（W4/A4）** 为例。
假设我们的硬件是标准的 AI 芯片（支持 INT4 乘加运算、移位运算和 FP16 浮点运算）。

**【预设维度说明】**
*   $N$：序列长度（Token 数量，比如 197 张图像 Patch）
*   $D$：特征通道维度（比如 768）
*   $h$：注意力头数（比如 12），每个头的维度 $D_h = D/h = 64$

以下是数据在一个完整的 **RepQ-ViT Block** 中的全生命周期流动过程（极其详细版）：

---

### 第一阶段：MSA（多头自注意力模块）计算

#### 1. 输入阶段
*   **当前数据**：上一层的输出 $X_{in}$
*   **数据形状**：`[N, D]`
*   **数据类型**：**FP16（浮点）**

#### 2. LayerNorm 1（包含重参数化魔法的第一处！）
*   **动作**：执行 LayerNorm。注意，这里的 $\gamma, \beta$ 已经是**被修改过（吸收了 $r_1, r_2$）**的 $\tilde{\gamma}, \tilde{\beta}$。
*   **当前数据**：激活值 $X'$
*   **数据形状**：`[N, D]`
*   **数据类型**：**FP16（浮点）**

#### 3. 准备生成 Q, K, V（第一次量化与矩阵乘法）
*   **量化激活 $X'$**：采用 **逐层均匀量化（Layer-wise MinMax/Percentile）**。
    *   将 FP16 的 $X'$ 压缩成 $X'_{int}$。
    *   **数据类型变为：INT4**。获得全局比例尺 $s_x$。
*   **加载权重 $W^{qkv}$**：硬盘里存的已经是经过 **逐通道均匀量化（Channel-wise）** 且吸收了 $r_1$ 的权重 $W^{qkv}_{int}$。
    *   **数据类型：INT4**。带有通道比例尺 $s_{wqkv}$。
*   **执行矩阵乘法**：$X'_{int} \times W^{qkv}_{int}$
    *   纯整数乘法，累加结果在芯片内部暂存为 **INT32**。
*   **立刻反量化（De-quantization）**：芯片将 INT32 的结果乘以 $(s_x \times s_{wqkv})$，再加上偏置 $b$。
*   **当前数据**：切分为 $Q, K, V$ 三个矩阵。
*   **数据形状**：三个 `[N, h, D_h]`
*   **数据类型**：**重回 FP16（浮点）**

#### 4. 计算注意力分数（Attention Score）
*   **量化 Q 和 K**：采用 **逐层均匀量化**，将 FP16 的 Q 和 K 压缩为 $Q_{int}$ 和 $K_{int}$（**INT4**）。
*   **执行矩阵乘法**：$Q_{int} \times K_{int}^T$（INT32 累加）。
*   **立刻反量化**：乘以比例尺 $(s_q \times s_k)$，并除以 $\sqrt{D_h}$。
*   **当前数据**：未归一化的注意力分数 $Score$。
*   **数据形状**：`[h, N, N]`
*   **数据类型**：**重回 FP16（浮点）**

#### 5. Softmax 激活
*   **动作**：算指数并归一化。
*   **当前数据**：注意力概率 $A$。
*   **数据形状**：`[h, N, N]`
*   **数据类型**：**FP16（浮点）**

#### 6. 注意力上下文加权（包含重参数化魔法的第二处！）
*   **量化概率 $A$**：采用 **$\log_2$ 量化（由 $\log_{\sqrt{2}}$ 重参数化而来）**。
    *   将 FP16 的 $A$ 转化为代表移位位数的整数 $A_{int}$。
    *   **数据类型变为：INT4**。获得包含所有常数渣渣的新比例尺 $\tilde{s}_a$。
*   **量化 $V$**：采用 **逐层均匀量化**，压缩为 $V_{int}$（**INT4**），获得比例尺 $s_v$。
*   **执行位移乘法**：硬件直接根据 $A_{int}$ 的值，对 $V_{int}$ 进行向右位移操作（极速！）并累加（存为 INT32）。
*   **立刻反量化**：将 INT32 结果乘以 $(\tilde{s}_a \times s_v)$。
*   **当前数据**：上下文矩阵 $Context$。
*   **数据形状**：`[N, D]`
*   **数据类型**：**重回 FP16（浮点）**

#### 7. 线性投影（Linear Proj）与第一次残差
*   **量化 Context**：逐层量化为 **INT4**。
*   **加载权重**：逐通道量化的 $W^{proj}_{int}$（**INT4**）。
*   **矩阵乘法 -> 反量化** -> 输出 FP16。
*   **残差相加**：$X_{in}$ (FP16) + $Proj\_Out$ (FP16)。
*   **当前数据**：MSA阶段总输出 $Y_{in}$。
*   **数据形状**：`[N, D]`
*   **数据类型**：**FP16（浮点）**

---

### 第二阶段：MLP（前馈神经网络）计算

#### 8. LayerNorm 2
*   **动作**：使用重参数化修改过的 $\tilde{\gamma}, \tilde{\beta}$ 执行归一化。
*   **当前数据**：激活值 $Y'$
*   **数据形状**：`[N, D]`
*   **数据类型**：**FP16（浮点）**

#### 9. MLP 第一层扩维 (Linear 1)
*   **量化 $Y'$**：逐层均匀量化为 **INT4**。
*   **加载权重**：逐通道量化的 $W^{fc1}_{int}$（**INT4**，特征维度通常从 D 放大到 4D）。
*   **矩阵乘法 -> 反量化** -> 输出 FP16。
*   **GELU 激活**：在 FP16 下计算非线性激活函数。
*   **当前数据**：隐藏层特征 $H$。
*   **数据形状**：`[N, 4D]`
*   **数据类型**：**FP16（浮点）**

#### 10. MLP 第二层缩维 (Linear 2) 与第二次残差
*   **量化 $H$**：逐层均匀量化为 **INT4**。
*   **加载权重**：逐通道量化的 $W^{fc2}_{int}$（**INT4**，维度从 4D 缩回 D）。
*   **矩阵乘法 -> 反量化** -> 输出 FP16。
*   **残差相加**：$Y_{in}$ (FP16) + $MLP\_Out$ (FP16)。
*   **当前数据**：当前 Block 的最终输出 $X_{out}$！
*   **数据形状**：`[N, D]`
*   **数据类型**：**FP16（浮点）**

---

### 核心总结（帮你理清大乱炖）：
1. **网络的主干血液永远是 FP16 浮点数。** 输入 Block 是 FP16，经过 LayerNorm/Softmax 也是 FP16，加残差也是 FP16，输出也是 FP16。
2. **只有在需要“做沉重的矩阵乘法”前夕，数据才会被临时“量化（掐头去尾截断）”成 INT4。**
3. **“反量化”永远紧紧跟在“矩阵乘法”的屁股后面！** 算完 INT32 后，立刻乘比例尺弹回 FP16，绝不带着 INT32 往下走。
4. **两处魔法的落脚点：** LayerNorm 魔法发生在**第 2 步**，修改了 LayerNorm 的公式；Softmax 魔法发生在**第 6 步**，把繁琐的对数乘法彻底变成了纯底层的硬件移位。