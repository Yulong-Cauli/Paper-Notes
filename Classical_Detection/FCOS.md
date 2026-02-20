# FCOS 

**核心思想**：将目标检测从传统的“基于锚框（Anchor-based）”回归问题，重构为类似语义分割的**全卷积逐像素预测（Per-pixel Prediction）**问题。

---

## Architecture

$$
\text{Image} \rightarrow \text{Backbone (e.g., ResNet)} \rightarrow \text{FPN} \rightarrow \text{Shared Heads}
$$

<div align="center">
    <img src="../../assets/FCOS/Figure2_a.png" width="90%">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9; display: inline-block; color: #999; padding: 2px;">
      注意：关于 Center-ness 的归属问题，后续实验证明，将其移到回归分支后，AP 进一步提升
    </div>
    <img src="../../assets/FCOS/Figure2_Tips.png" width="70%">
</div>


### 预测头 (The Head)

FPN 输出的每一层特征图（$P_3$ 到 $P_7$）都共享同一个预测头结构。它由**全卷积层**构成，分为两个“塔”：

1.  **分类塔 (Classification Tower)**：
    * **输出 1**：类别概率张量，尺寸 $H \times W \times C$（使用 Focal Loss）。
    * **输出 2**：Center-ness 张量，尺寸 $H \times W \times 1$（使用 BCE Loss）。
2.  **回归塔 (Regression Tower)**：
    * **输出**：距离向量张量，尺寸 $H \times W \times 4$（使用 IOU Loss）。

---

## 方法

### Anchor-free 的逐像素回归

**传统痛点**：Anchor 设计复杂，超参数敏感（尺寸、长宽比），且导致正负样本极度不平衡。

**FCOS 解法**：彻底抛弃 Anchor，将特征图上的点视为训练样本。

* **定义样本**：特征图上的点 $(x, y)$ 映射回原图，只要落在真实框（Ground Truth）内部，即视为**正样本**。
* **回归目标**：直接预测该点到真实框四条边的距离 $t = (l, t, r, b)$。
* **物理意义**：“位置（Location）”是指特征图上的像素，对应原图感受野的中心点。

---

### FPN 解决重叠

**痛点**：逐像素预测容易遇到“一个点落在两个重叠物体里”的模糊问题（Ambiguity）。

**FCOS 解法**：利用特征金字塔（FPN）进行“分流”。

* **分层规则**：根据**回归距离的范围**（而不是锚框大小）进行限制。
  * $P_3$ (小感受野) $\rightarrow$ 预测小物体（距离范围 $[0, 64]$）
  * ...
  * $P_7$ (大感受野) $\rightarrow$ 预测大物体（距离范围 $[512, \infty]$）
* **效果**：绝大多数重叠物体因尺寸不同被分到了不同层级，模糊样本比例大幅下降。若仍有重叠，则取面积最小的框。

---

### Center-ness 分支 

**痛点**：由于所有“框内点”都被视为正样本，**远离中心的边缘位置**产生的预测框质量通常很差，拉低了检测精度。

**FCOS 解法**：增加一个分支预测“中心度”。

* **定义**：衡量当前像素距离物体几何中心的程度（范围 0~1）。

* **计算公式**：
  $$
  \text{centerness}^* = \sqrt{\frac{\min(l^*, r^*)}{\max(l^*, r^*)} \times \frac{\min(t^*, b^*)}{\max(t^*, b^*)}}
  $$

* **作用**：在推理阶段：
  $$
  \text{最终得分} = \text{分类置信度} \times \text{Center-ness}
  $$
  这使得低质量的边缘框得分变低，从而在 NMS（非极大值抑制）过程中被过滤掉。

------

### Loss Function

FCOS 的训练目标由三部分组成：**分类损失**、**回归损失**以及**中心度损失**。最终的总损失函数定义如下：
$$
L_{total} = \frac{1}{N_{pos}} \sum_{x,y} L_{cls}(p_{x,y}, c^*_{x,y}) + \frac{1}{N_{pos}} \sum_{x,y} \mathbb{I}_{\{c^*_{x,y} > 0\}} L_{reg}(t_{x,y}, t^*_{x,y}) + \frac{1}{N_{pos}} \sum_{x,y} \mathbb{I}_{\{c^*_{x,y} > 0\}} L_{center}(p_{ctr}, c^*_{ctr})
$$


其中 $N_{pos}$ 表示正样本的数量，用于归一化。求和是对特征图上所有位置 $(x, y)$ 进行的。

---

**分类损失 $L_{cls}$**：**Focal Loss** 。

在 FCOS 这种单阶段检测器中，特征图上绝大多数位置都是背景（负样本）。如果使用普通的交叉熵损失（CE Loss），大量的简单负样本（Easy Negatives）会淹没少数正样本的梯度，导致训练失败。

- **原理**：Focal Loss 通过降低简单样本（容易分类的背景）的权重，强制模型专注于那些“难分样本”（Hard Examples）的训练。
- **细节**：FCOS 训练的是 $C$ 个二元分类器，而不是一个多分类器。

因为激活函数是 Sigmoid 函数，而不是 Softmax。

根本原因是负样本太多，COCO里面只有80个物体，背景类的样本数量是前景的几万倍。Softmax 会让所有类别去竞争，导致背景类占据绝对主导，淹没其他类别的梯度。

---

**回归损失 $L_{reg}$** ：**IOU Loss** 。$L_{reg} = - \ln(\text{IOU})$

仅针对**正样本**计算（$\mathbb{I}_{\{c^* > 0\}}$ 表示只有落在物体内的点才计算此损失）。

- **为什么不用 L1/L2 Loss？** 直接回归距离数值（如 Smooth L1）对物体的尺度很敏感。例如，对于小物体，5个像素的误差可能很严重；但对于大物体，5个像素误差微不足道。
- **IOU Loss 的优势**：它直接优化预测框与真实框的**交并比（Intersection over Union）**。IOU 是尺度不变的（Scale Invariant），无论物体大小，其值都在 0~1 之间，能更好地反映检测框的质量。

---

**中心度损失** $L_{center}$ ：**BCE Loss** 。$$L_{center} = - \left[ \text{centerness}^* \cdot \log(p_{ctr}) + (1 - \text{centerness}^*) \cdot \log(1 - p_{ctr}) \right]$$

仅针对**正样本**计算 。

- **目标值构建**：对于每一个正样本位置，根据其到四边的回归目标 $(l^*, t^*, r^*, b^*)$ 计算出一个 0~1 之间的实数作为“标签”：
  $$
  \text{centerness}^* = \sqrt{\frac{\min(l^*, r^*)}{\max(l^*, r^*)} \times \frac{\min(t^*, b^*)}{\max(t^*, b^*)}}
  $$

- **原理**：

  - 这是一个二分类回归任务（逻辑回归）。
  - 使用**根号**（Sqrt）是为了减缓中心度随距离衰减的速度，让中心附近的区域都能保持相对较高的权重，不至于只有正中心一点是。

- **作用**：迫使网络学习区分“高质量中心点”和“低质量边缘点”。


