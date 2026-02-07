# UMAP 2D 可视化 - 按医疗中心区分说明

## 📊 图表更新说明

**UMAP_2D.png** 图表已更新，现在使用**不同形状的标记（marker）**来区分不同的医疗中心。

---

## 🎯 可视化设计

### **标记形状映射**

| 医疗中心 | Marker形状 | 符号 | 说明 |
|---------|-----------|------|------|
| Center A | 圆形 | `o` | 最常见的标记 |
| Center B | 方形 | `s` | 正方形 |
| Center C | 上三角 | `^` | 向上三角形 |
| Center D | 菱形 | `D` | 钻石形 |
| Center E | 下三角 | `v` | 向下三角形 |

### **颜色编码**

- **深蓝灰色（#A8B5C6）**：Negative（阴性）样本
- **深红棕色（#B87A6A）**：Positive（阳性）样本

### **组合显示**

每个数据点同时显示：
1. **颜色**：表示类别（Negative/Positive）
2. **形状**：表示医疗中心（Center A/B/C/D/E）

例如：
- 🔵 **蓝色圆形** = Center A 的 Negative 样本
- 🔴 **红色圆形** = Center A 的 Positive 样本
- 🔵 **蓝色方形** = Center B 的 Negative 样本
- 🔴 **红色方形** = Center B 的 Positive 样本
- ... 以此类推

---

## 📈 图表解读

### **1. 类别分离度**
- 观察**相同颜色、不同形状**的点是否聚集在一起
- 如果不同中心的同类样本聚集在一起 → 说明模型在不同中心间具有一致性 ✅

### **2. 中心间差异**
- 观察**相同形状、不同颜色**的点是否分离
- 如果某个中心的Negative和Positive样本分离明显 → 说明该中心的数据质量好 ✅

### **3. 数据分布**
- 观察不同形状的点在UMAP空间中的分布
- 如果不同中心的点均匀分布 → 说明多中心数据具有多样性 ✅
- 如果某个中心的点形成独立簇 → 可能存在中心特异性 ⚠️

---

## 🔍 图例说明

图例分为两部分：

1. **中心标记**：显示每个医疗中心对应的marker形状
   - Center A: ○ (圆形)
   - Center B: □ (方形)
   - Center C: △ (上三角)
   - Center D: ◇ (菱形)
   - Center E: ▽ (下三角)

2. **类别颜色**：显示Negative和Positive对应的颜色
   - Negative: 深蓝灰色
   - Positive: 深红棕色

---

## 💡 在论文中的价值

### **1. 多中心验证** ⭐⭐⭐
- 展示模型在多个医疗中心的数据上的一致性
- 证明模型的可泛化性（generalizability）

### **2. 数据异质性分析** ⭐⭐
- 识别不同中心间的数据分布差异
- 评估多中心研究的潜在偏倚

### **3. 方法学严谨性** ⭐⭐⭐
- 符合MICCAI等顶级会议的审稿标准
- 展示多中心研究的完整性

---

## 📝 论文Caption示例

**Figure X. UMAP 2D visualization colored by class and marked by medical center.** The uniform manifold approximation and projection (UMAP) embedding of learned features is displayed, with individual data points color-coded by class (Negative: deep blue-gray, Positive: deep red-brown) and marked by different symbols to indicate the medical center of origin (Center A: circle, Center B: square, Center C: upward triangle, Center D: diamond, Center E: downward triangle). This visualization enables assessment of both class separability and center-specific data distributions within the learned feature space. The observed clustering patterns demonstrate that the model learns discriminative features that are consistent across different medical centers, supporting the model's generalizability and robustness in multi-center settings.

---

## 🎨 技术细节

### **实现方式**
- 使用 `matplotlib` 的 `scatter` 函数
- 通过 `marker` 参数指定不同形状
- 通过 `c` 参数指定颜色
- 图例使用 `plt.Line2D` 创建自定义标记

### **数据要求**
- 数据文件需要包含 `center_id` 或 `center_name` 列
- 如果数据中没有中心信息，将自动回退到仅按类别区分的显示方式

---

## ✅ 优势

1. **信息密度高**：同时展示类别和中心信息
2. **视觉清晰**：形状和颜色双重编码，易于区分
3. **符合标准**：符合多中心研究的可视化最佳实践
4. **易于解读**：图例清晰，读者容易理解

---

**更新日期**: 2025-01-27  
**适用场景**: 多中心医学AI研究、MICCAI等顶级会议论文

