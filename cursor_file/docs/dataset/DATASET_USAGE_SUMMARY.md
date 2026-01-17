<!--
文件生成信息:
- 生成时间: 2025-12-25 08:43:45 (使用nvidia-smi获取CUDA时间)
- 生成需求: 用户要求明确实际代码和工程中使用的数据集路径，并进行可视化
- 生成原因: 需要清晰了解数据集的实际路径、结构、样本分布和软链接情况
- 相关任务: 数据集可视化，MICCAI论文准备

文件功能: 总结实际使用的数据集路径、结构和可视化结果
-->

# 实际使用的数据集路径总结

## 📍 关键路径

### 1. 实际使用的数据集（代码中）

**路径**: `/data2/hmy/5Center_datas/5centers_multi_internal_external_final`

**说明**: 
- ✅ 这是**实际代码和工程中使用的数据集路径**
- ✅ 包含内外部数据集划分（内部开发集 + 外部测试集）
- ✅ 所有图像文件通过**软链接**指向原始数据集
- ✅ 节省磁盘空间：约182GB → 约3.3MB（软链接）

**在代码中的使用**:
```python
# experiments/exp1_causal_bayesian_clip/train.py
data_path = '/data2/hmy/5Center_datas/5centers_multi_internal_external_final'

# experiments/exp1_causal_bayesian_clip/train_vlm.py
data_path = '/data2/hmy/5Center_datas/5centers_multi_internal_external_final'
```

### 2. 原始数据集（数据源）

**路径**: `/data2/hmy/5Center_datas/5centers_multi`

**说明**:
- ✅ 包含所有原始图像文件（182GB）
- ✅ 软链接的目标路径
- ✅ 数据完整性：100%

---

## 📊 数据集结构

### 目录结构

```
5centers_multi_internal_external_final/
├── internal_train/                    # 内部开发集
│   ├── train/                         # 内部训练集
│   │   ├── oct/                       # OCT图像软链接 (660个)
│   │   └── col/                       # Colposcopy图像软链接 (660个)
│   └── val/                           # 内部验证集
│       ├── oct/                       # OCT图像软链接 (166个)
│       └── col/                       # Colposcopy图像软链接 (166个)
├── external_validation/               # 外部测试集
│   ├── oct/                           # OCT图像软链接 (159个)
│   └── col/                           # Colposcopy图像软链接 (159个)
├── train_labels.csv                   # 内部训练集标签 (660个样本)
├── val_labels.csv                     # 内部验证集标签 (166个样本)
└── external_test_labels.csv           # 外部测试集标签 (159个样本)
```

### 软链接说明

所有图像文件都是**软链接**，指向原始数据集：

```bash
# 示例：训练集OCT图像软链接
internal_train/train/oct/M22102_2023_P0000147 
  → /data2/hmy/5Center_datas/5centers_multi/train/oct/M22102_2023_P0000147

# 示例：训练集Colposcopy图像软链接
internal_train/train/col/20230316_黄贵云 
  → /data2/hmy/5Center_datas/5centers_multi/train/col/20230316_黄贵云
```

---

## 📈 数据集统计

### 样本分布

| 数据集 | 样本数 | OCT图像 | Colposcopy图像 | 状态 |
|--------|--------|---------|----------------|------|
| **内部训练集** | 660 | ✅ 660/660 (100%) | ✅ 660/660 (100%) | ✅ 完整 |
| **内部验证集** | 166 | ✅ 166/166 (100%) | ✅ 166/166 (100%) | ✅ 完整 |
| **外部测试集** | 159 | ✅ 159/159 (100%) | ✅ 159/159 (100%) | ✅ 完整 |
| **总计** | **985** | **✅ 985/985 (100%)** | **✅ 985/985 (100%)** | **✅ 完整** |

### 软链接统计

| 数据集 | OCT软链接 | Colposcopy软链接 | 总计 |
|--------|-----------|------------------|------|
| **训练集** | 660 | 660 | 1,320 |
| **验证集** | 166 | 166 | 332 |
| **外部测试集** | 159 | 159 | 318 |
| **总计** | **985** | **985** | **1,970** |

### 医疗中心分布

| 中心 | 类型 | 样本数 | 用途 |
|------|------|--------|------|
| **恩施** | Internal | 404 | 内部开发集 |
| **襄阳** | Internal | 344 | 内部开发集 |
| **十堰** | Internal | 78 | 内部开发集 |
| **荆州** | External | 70 | 外部测试集 |
| **武大** | External | 89 | 外部测试集 |

### 标签分布

| 标签 | 数量 | 比例 |
|------|------|------|
| **Positive (1)** | ~500 | ~50.8% |
| **Negative (0)** | ~485 | ~49.2% |

---

## 🎯 代码中的使用

### 主要训练脚本

1. **`experiments/exp1_causal_bayesian_clip/train.py`**
   ```python
   data_path = '/data2/hmy/5Center_datas/5centers_multi_internal_external_final'
   ```

2. **`experiments/exp1_causal_bayesian_clip/train_vlm.py`**
   ```python
   data_path = '/data2/hmy/5Center_datas/5centers_multi_internal_external_final'
   ```

3. **`experiments/exp1_causal_bayesian_clip/train_causal_bayesian.py`**
   ```python
   data_path = '5centers_multi'  # 相对路径，需要确认实际路径
   ```

### 数据加载

数据加载器会自动识别数据集结构：

```python
from src.data.enhanced_multimodal_dataset import build_enhanced_dataset

# 自动识别数据集结构
train_dataset = build_enhanced_dataset('train', args)
val_dataset = build_enhanced_dataset('test', args)
```

---

## 📊 可视化图表

可视化图表已生成：`cursor_file/dataset_structure_visualization.png`

图表包含：
1. **数据集路径结构** - 显示实际使用的数据集路径和原始数据集路径
2. **样本数量分布** - 各数据集的样本数
3. **按中心分布** - 各医疗中心的样本分布
4. **标签分布** - 正负样本分布
5. **内外部数据集划分** - 饼图显示内外部数据集比例
6. **各数据集标签分布** - 堆叠柱状图
7. **软链接统计** - 详细的软链接统计表

---

## ✅ 验证方法

### 检查软链接

```bash
# 检查训练集OCT软链接
ls -la /data2/hmy/5Center_datas/5centers_multi_internal_external_final/internal_train/train/oct/ | head -5

# 检查训练集Colposcopy软链接
ls -la /data2/hmy/5Center_datas/5centers_multi_internal_external_final/internal_train/train/col/ | head -5

# 统计软链接数量
find /data2/hmy/5Center_datas/5centers_multi_internal_external_final -type l | wc -l
```

### 验证数据完整性

```bash
# 运行数据完整性检查脚本
python3 cursor_file/check_dataset_integrity.py
```

---

## 💡 重要提示

1. **实际使用的数据集**: `/data2/hmy/5Center_datas/5centers_multi_internal_external_final`
2. **数据来源**: `/data2/hmy/5Center_datas/5centers_multi`（原始数据集）
3. **软链接**: 所有图像文件都是软链接，指向原始数据集
4. **节省空间**: 软链接方式节省了约182GB磁盘空间
5. **数据完整性**: 100%，所有图像文件都找到了

---

**生成日期**: 2025-12-25 08:43:45  
**状态**: ✅ 数据集路径已明确，可视化已完成

