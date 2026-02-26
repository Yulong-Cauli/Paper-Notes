# MQBench

**出处会议：** NeurIPS 2021  
**是否开源：** https://github.com/ModelTC/MQBench  2025年1月10号还在更新  
**关键词：** Benchmark，QAT，PTQ，计算图

---

## 1. 概述 

本文直击模型量化领域的两大核心痛点：**“不可复现”** 和**“不可部署”**。

为此，作者提出了 MQBench，在软件层面拉齐了评价标准（统一数据增强、超参搜索）；在硬件层面强制对齐真实算子和计算图约束。

是**首个真正面向工业级落地、确保训练与部署完全一致的模型量化基准测试框架**。

<div align="center"><img src="https://raw.githubusercontent.com/Yulong-Cauli/Paper-Notes/main/assets/MQBench/_page_1_Figure_0.jpeg""></div>

---

## 2. 方法

要想让学术界的量化模型在真实的手机 NPU / 显卡 GPU 上跑通且不掉点，必须跨越三个维度的鸿沟：**算子对齐、模块对齐、架构对齐**。

----

### 2.1 算子对齐：伪量化与真量化

假设原浮点数为 $x$，缩放因子为 $s$，零点为 $z$。

**真实量化 (Quantize)**，也就是转为真实 INT8：
$$
q = \text{clip}\left(\text{round}\left(\frac{x}{s}\right) + z, \ 0, \ 255\right)
$$

**反量化 (Dequantize)**， 也就是转回 FP32：

$$
\hat{x} = (q - z) \cdot s
$$

**伪量化 (FakeQuant)** = 真实量化 + 反量化，即人为在 FP32 计算流中引入 INT8 的截断误差：

$$
\text{FakeQuant}(x) = \hat{x} = \left( \text{clip}\left(\text{round}\left(\frac{x}{s}\right) + z, \ 0, \ 255\right) - z \right) \cdot s
$$

- **问题：“伪量化”到底伪在哪？为什么要保持 FP32？** 
  - 数据类型依然是 PyTorch 中的 `float32`，但其数值空间被强行限制在了离散的阶梯值上，从而在数学上完美模拟了 INT8 的截断误差。
- **问题：为什么在 QAT 中需要保持 FP32？** 
  - 因为量化函数 `round()` 不可导（导数为0）。如果直接转成物理 INT8，梯度无法传递，且微小的权重更新（如 $0.001$）会被直接抹杀。因此必须保留 FP32 权重接收梯度，仅在前向传播时制造“模拟误差”。
- **问题：为什么在 PTQ 中也需要保持 FP32？** 
  - 第一，目前的深度学习框架缺乏灵活的原生 INT8/INT32 混合图执行生态；第二，高级 PTQ 需要计算量化前后的输出重建误差（Reconstruction Loss），这必须在统一的高精度 FP32 域中进行比较和优化。

---

真实硬件执行卷积时（**真量化**），是纯整型交叉计算。在上图中，卷积算子执行的是浮点乘加：

$$
y = \sum(\hat{x} \cdot \hat{w})
$$

为了能在硬件上运行，下图必须将其转换为纯整数运算。我们将反量化公式代入，提取常数缩放因子到求和号外部，展开多项式：

$$
y &=& \sum(\hat{x} \cdot \hat{w}) \\
  &=& \sum \left (s_x(q_x - z_x)\cdot s_w(q_w - z_w)\right ) \\  
  &=&(s_x \cdot s_w) \sum \big( q_x q_w - z_w q_x - z_x q_w + z_x z_w \big)
$$

其中 $q_x q_w$ 是物理电路上的 `INT8 x INT8 -> INT32` 累加。MQBench 必须在 PyTorch 中通过设定严格的伪量化参数，来使得模拟计算的结果与物理硬件上述公式完全一致。

<div align="center"><img src="https://raw.githubusercontent.com/Yulong-Cauli/Paper-Notes/main/assets/MQBench/_page_3_Figure_0.jpeg""></div>

<div align="center"><img src="https://raw.githubusercontent.com/Yulong-Cauli/Paper-Notes/main/assets/MQBench/_page_3_Figure_1.jpeg""></div>

---

### 2.2 模块对齐：BN 层折叠

在部署时，Conv 和 BN在物理上已经被合并成了一个算子。如果我们在 QAT 时不严格按照合并后的数学形式去模拟量化误差，训练出来的模型在部署时精度就会崩溃。

在 FP32 的标准神经网络中，前向传播依次经过 Conv 和 BN：

**Conv 公式：**

$$
y_{conv} = W \cdot x + b
$$

**BN 公式： ** 这里 $\mu$ 是均值， $\sigma$ 是标准差， $\gamma$ 和 $\beta$ 是 BN 层的可学习缩放和偏移参数

$$
y_{bn} = \gamma \cdot \frac{y_{conv} - \mu}{\sigma} + \beta
$$

**折叠 (Folding) 的数学推导：**
为了在部署时加速，推理引擎，如 TensorRT 会把 $y_{conv}$ 代入 $y_{bn}$ 中展开：

$$
y_{bn} = \gamma \cdot \frac{(W \cdot x + b) - \mu}{\sigma} + \beta
$$

我们把带有 $x$ 的项和常数项分离开来：

$$
y_{bn} = \left( W \cdot \frac{\gamma}{\sigma} \right) \cdot x + \left( (b - \mu) \cdot \frac{\gamma}{\sigma} + \beta \right)
$$

**结论：** 在实际部署的物理芯片上，不存在独立的 BN 层。它执行的是一个全新的 Conv 算子，其权重和偏置变成了：

*   **折叠后的权重 (Folded Weight)：** $W_{fold} = W \cdot \frac{\gamma}{\sigma}$
*   **折叠后的偏置 (Folded Bias)：** $b_{fold} = (b - \mu) \cdot \frac{\gamma}{\sigma} + \beta$

---

既然部署时用的是 $W_{fold}$，那么我们在做 量化 时，理所当然应该去量化 $W_{fold}$，而不是原来的 $W$。

- **错误的做法（早期的一些 QAT 方案）：**
  先对原始权重 $W$ 做伪量化，算出卷积后，再经过常规的 BN 层。
  此时模型经历的等效权重是： $\text{FakeQuant}(W) \cdot \frac{\gamma}{\sigma}$

- **正确的做法（MQBench 及工业界标准，图3-e）：**
  必须先把 $W$ 和 BN 参数乘起来得到 $W_{fold}$，然后再对 $W_{fold}$ 做伪量化。
  此时模型经历的等效权重是： $\text{FakeQuant}\left( W \cdot \frac{\gamma}{\sigma} \right)$

举个例子，假设当前通道的参数如下：
*   原始卷积权重: $W = 1.2$
*   BN层的参数: $\gamma = 0.2$ , $\sigma = 1.0$ (即缩放系数 $\frac{\gamma}{\sigma} = 0.2$)

**情况 A（错误做法：先量化 $W$，再乘 BN）：**

1. 量化 $W$ : $\text{FQ}(1.2) = \text{round}(1.2 / 0.5) \times 0.5 = \text{round}(2.4) \times 0.5 = 2 \times 0.5 = \mathbf{1.0}$
2. 作用 BN: 最终等效权重 $= 1.0 \times 0.2 = \mathbf{0.20}$

**情况 B（部署态真实情况 / MQBench 的做法：先折叠，再量化）：**
1. 先计算物理折叠权重: $W_{fold} = W \cdot \frac{\gamma}{\sigma} = 1.2 \times 0.2 = \mathbf{0.24}$
2. 对 $W_{fold}$ 进行伪量化: $\text{FQ}(0.24) = \text{round}(0.24 / 0.5) \times 0.5 = \text{round}(0.48) \times 0.5 = 0 \times 0.5 = \mathbf{0.0}$

**结论**：
情况 A 算出来是 $0.20$，情况 B 算出来是 $0.0$。
如果你的 QAT 框架采用情况 A，你的模型在训练时会认为这个权重还有微小的贡献（0.20），从而把 Loss 降下去了。
但是！一旦把这个模型导出到手机芯片上（芯片必定执行情况 B），这个权重直接变成了 **0.0**，特征图在这里直接断流，导致部署后的精度暴跌。

---

既然证明了必须用情况 B（先折叠，再量化），那为什么图 3-e 画得那么复杂呢？直接用情况 B 计算不就好了吗？

**训练时的变量困境：**
在推理（部署）时，BN 的 $\mu$ 和 $\sigma$ 是固定的（跑完训练定死的 Running Mean/Var）。
但在 QAT 训练时，每一轮 Batch 送进来的数据都在变，导致当步的 $\mu_{batch}$ 和 $\sigma_{batch}$ 剧烈波动。
如果我们在每次 Forward 传播时，直接彻底把波动的 $\mu_{batch}$ 吸收到偏置 $b$ 里去更新参数，反向传播的梯度会极其混乱，模型根本无法收敛，这对应了论文中提到的图 3-a 在训练时不可行。

<div align="center"><img src="https://raw.githubusercontent.com/Yulong-Cauli/Paper-Notes/main/assets/MQBench/_page_4_Figure_0.jpeg""></div>

**5 张核心演进图解析：**

*   **图 (a) 纯部署态**：理想终点。无独立 BN 层，卷积直接使用提前折叠好的参数（ $W_{fold}, b_{fold}$ ）运行。
*   **图 (b) 朴素折叠**：最天真的 QAT。前向传播直接强行用当步 Batch 的波动的均值和方差去更新偏置，导致训练极其不稳定，Loss 无法收敛。
*   **图 (c) 双路计算**：另开一条无量化误差的全精度支路去计算干净的 BN 统计量。缺点是计算量翻倍，且数学上与部署态存在微小不匹配。
*   **图 (d) 解耦偏置**：保留全局方差更新权重，另跑影子网络获取均值更新偏置，同时加一个尺度补偿因子。缺点是在低比特下依然会暴露精度损失。
*   **图 (e) MQBench 终极方案**：
    *   **乘法路径（模拟真实误差）**：用当前缩放系数 $\frac{\gamma}{\sigma}$ 对权重进行折叠并伪量化，卷积运算严格承担真实的量化噪音。
    *   **加法路径（保留稳定梯度）**：卷积输出后，逆向乘以 $\frac{\sigma}{\gamma}$ 抵消缩放，接入一个正常的 BN 层去计算平滑统计量（EMA）和偏置更新。此方案完美兼顾了“量化误差同构”与“梯度更新稳定”。

**总结 2.2 节：**
这一节的科学价值在于：它在 PyTorch 中构建了一个**数学上极其严谨的计算子图**（图 3-e）。

*   在**前向的权重乘法**上，它严格模拟了芯片底层的**先折叠后量化**的舍入误差；
*   在**统计量和梯度回传**上，它又通过数学上的逆运算保留了标准 BN 层的作用，使得优化器能够稳定工作。

这就是为什么用 MQBench 训出来的模型，能够和 PyTorch 里测出来的精度做到几乎 $1:1$ 无损。

---

### 2.3 架构对齐：硬件计算图约束

不同的硬件编译器对节点拓扑极其苛刻。学术界随意插量化节点会导致真实芯片无法进行算子融合，或者因类型不匹配直接报错。

<div align="center"><img src="https://raw.githubusercontent.com/Yulong-Cauli/Paper-Notes/main/assets/MQBench/_page_5_Figure_0.jpeg""></div>

**3 张核心拓扑图解析（以残差结构为例）：**
*   **图 (1) 学术设定**：无脑在算子后插 FakeQuant。导致 Add 节点的两个输入一个是 INT8（带误差），一个是 FP32（未截断），硬件底层根本无法异构相加；且打破了底层 `Conv+ReLU` 的硬融合规则。
*   **图 (2) 张量推理引擎 (如 TensorRT)**：要求节点前移到 Conv 输入端。允许 Add 节点以 INT32/FP32 的高精度累加，加完再统一量化。部署相对友好，掉点少。
*   **图 (3) 极低功耗 DSP (如 高通 SNPE)**：极其苛刻。不仅要求 `Conv+ReLU` 必须连在一起物理融合，还强制要求 Add 节点的左右两个分支在相加前必须都是 INT8，且 **Scale（缩放因子）必须强行绑定为一致**，从而通过极简汇编指令完成同构相加。在此设定下极易掉大点。

---

## 实验部分

MQBench 在强制拉齐所有条件的“绝对公平”赛场上，得出了以下惊人结论：

*   **5.1 剥除“训练外挂”的公平比拼**：当所有算法使用相同的超参搜索、数据增强后，差距大幅缩小。比起复杂的截断公式，**量化步长（Step Size）的学习率调整最为关键**。基于梯度的 **LSQ 算法** 展现出最强的普适性。
*   **5.2 硬件图约束的残酷审判**：在学术图（Academic）下各算法均表现极好，但在 TensorRT 中会有轻微掉点。而一旦切入 SNPE (DSP) 的强图约束（强制绑定 Scale），大部分算法精度发生断崖式暴跌。证明了脱离硬件谈量化毫无意义。
*   **5.3 BN 融合的照妖镜**：验证了 3.2 节的各 BN Folding 策略。图(b)无法收敛，图(c)(d)与真实部署间存在 1-2% 的精度差。只有采用了**图(e) 策略**的模型，实现了训练态与部署态的 **Zero Drop（零掉点）**。
*   **5.4 测试4-bit 量化**：在 INT8 下拉不开差距的算法，在极度缺乏表达空间的 4-bit 遭受降维打击。统计学方法彻底失效，只有 LSQ / PACT 这种**把截断边界作为参数参与反向传播学习**的方法活了下来。

---

## 附录C部分：PTQ相关实验

PTQ 用于在无梯度、无算力、只有极少量无标签校准集情况下的快速量化部署。

虽然基础统计法（如 Min-Max）容易掉点，但引入高级方法（如 AdaRound 局部权重寻优、BN 重新校准）能大幅拉回精度。

并且，**先跑一次 PTQ，拿其参数作为初始化再跑 QAT**，是提升 QAT 上限的极佳工程 Trick。

---

### C.3 BatchNorm 校准题。

如我们在 2.2 节所说，QAT 训练时可以平滑处理 BN 折叠。

但在 PTQ 中，我们是直接拿着 FP32 模型算好的 BN 均值 ($\mu$) 和方差 ($\sigma$) 强行折叠到权重里去的。因为量化引入了截断误差，导致原本特征图的分布发生了微小的偏移。此时，如果你还用 FP32 统计出来的 $\mu$ 和 $\sigma$，就会误差累积。

**MQBench 的解法 (BN Re-calibration)**：
在完成权重的 PTQ 量化后，将网络的 BN 层重新打开，但不做反向传播。把几百张校准图片重新跑一遍网络（纯 Forward）。
利用这次带有量化误差的前向传播，**重新计算并更新所有 BN 层的 Running Mean 和 Running Var**。

---

**基于 Min-Max 方法的完整流程解析**

假设你现在跑Min-Max，流程如下：

1.  **BN 层折叠**：首先从数学上把 FP32 模型的 BN 层彻底吸收合并进前一层的 Conv 权重 $W_{fold}$ 和偏置中，消除独立 BN。
2.  **插 Observers**：在模型的每一层权重矩阵和输入特征图上安插“Observers”。
3.  **跑校准数据**：将 100 张图片作为 Batch 喂入模型，**仅做前向传播（Forward）**。
    *   权重探头：找出固定权重的最大值与最小值。
    *   特征图探头：不断刷新记录，找出这 100 张图流过时产生过的历史绝对最大值和最小值。
4.  **计算量化参数 (核心数学)**：
    *   假设某层特征图监控到的物理范围是 $Min = -3.2$, $Max = 8.4$。
    *   **计算缩放因子 $s$**：跨度映射 $\Rightarrow s = \frac{8.4 - (-3.2)}{255} \approx 0.0455$。
    *   **计算零点 $z$**：求 FP32 的 $0.0$ 落在 INT8 的哪里 $\Rightarrow z = \text{round}\left(0 - \frac{-3.2}{0.0455}\right) = 70$。
5.  **离线转换与部署**：
    *   利用算出的 $s_w, z_w$ 将模型硬盘里的浮点权重永久转化为 `int8` 存储，模型体积压缩 4 倍。
    *   将算出的特征图 $s_a, z_a$ 写入部署配置。硬件 NPU 在推理时，通过提取这些参数进行底层的纯整数矩阵乘加运算。
