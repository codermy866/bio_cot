# 数据集大小分析报告

## 📊 数据集大小总览

### 各数据集目录大小对比

| 数据集目录 | 总大小 | 文件数 | 目录数 | 说明 |
|-----------|--------|--------|--------|------|
| **5centers_multi** | **182 GB** | 121,400 | 1,977 | 原始完整数据集 |
| **5centers_multi_internal_external_final** | **1.2 GB** | 1,546 | 487 | 最终划分版本（推荐）✅ |
| **5centers_multi_internal_external_recommended** | **642 MB** | 496 | 160 | 推荐版本（不推荐使用）❌ |
| **5centers_multi_internal_external** | - | - | - | 可能已删除或不存在 |

---

## 📁 详细大小分析

### 1. 原始数据集: `5centers_multi` (182 GB)

```
5centers_multi/
├── train/                   145 GB  (训练数据)
├── test/                    38 GB   (测试数据)
├── train_labels.csv         40 KB   (训练标签)
├── test_labels.csv          12 KB   (测试标签)
├── dataset_info.json        4 KB    (数据集信息)
└── datasets_distribution.xlsx 12 KB (分布统计)
```

**特点**:
- ✅ 完整数据集，包含所有5个中心的数据
- ✅ 包含训练集和测试集
- ⚠️ 占用空间最大（182 GB）

---

### 2. 最终划分版本: `5centers_multi_internal_external_final` (1.2 GB) ✅ **推荐**

```
5centers_multi_internal_external_final/
├── external_validation/      1.2 GB  (外部测试集数据)
├── internal_train/          2.1 MB  (内部训练数据)
├── train_labels.csv         36 KB   (训练标签)
├── val_labels.csv           12 KB   (验证标签)
├── external_test_labels.csv  8 KB    (外部测试标签)
├── visualizations/           1.5 MB  (可视化图表)
├── README.md                 8 KB    (说明文档)
├── split_report.md          4 KB    (划分报告)
└── split_statistics.json    4 KB    (统计信息)
```

**特点**:
- ✅ 符合要求的划分（内部=恩施+襄阳+十堰，外部=荆州+武大）
- ✅ 有完整文档和可视化
- ✅ 占用空间适中（1.2 GB）
- ✅ 结构清晰，便于使用

**文件统计**:
- 总文件数: 1,546
- 总目录数: 487

---

### 3. 推荐版本: `5centers_multi_internal_external_recommended` (642 MB) ❌ **不推荐**

```
5centers_multi_internal_external_recommended/
├── external_validation/      638 MB  (外部验证数据)
├── internal_train/          3.4 MB  (内部训练数据)
├── train_labels.csv         48 KB   (训练标签)
├── test_labels.csv          12 KB   (测试标签)
└── split_statistics.json   4 KB    (统计信息)
```

**特点**:
- ❌ 划分不符合要求（武大在内部，十堰在外部）
- ❌ 缺少文档和可视化
- ⚠️ 占用空间较小（642 MB），但结构不完整

**文件统计**:
- 总文件数: 496
- 总目录数: 160

---

## 💾 磁盘使用情况

### 当前磁盘状态

```
磁盘: /dev/sdc1
总容量: 3.6 TB
已使用: 3.2 TB (92%)
剩余空间: 294 GB
```

### 数据集占用空间

- **原始数据集**: 182 GB
- **划分后数据集**: ~1.2 GB (final版本)
- **总占用**: ~183 GB

---

## 📈 文件类型统计

### 原始数据集 (`5centers_multi`)

- **OCT图像文件** (.npy): 238 个
- **Colposcopy图像文件** (.jpg): 2,955 个
- **CSV标签文件**: 2 个
- **JSON配置文件**: 1 个

### 数据分布

- **训练集**: 145 GB (约80%的数据)
- **测试集**: 38 GB (约20%的数据)

---

## 🎯 使用建议

### 推荐使用: `5centers_multi_internal_external_final`

**理由**:
1. ✅ **符合要求**: 内部=恩施+襄阳+十堰，外部=荆州+武大
2. ✅ **文档完整**: 有README、统计报告、可视化
3. ✅ **结构清晰**: train/val/external_test划分明确
4. ✅ **空间适中**: 1.2 GB，便于管理和传输
5. ✅ **数据完整**: 包含所有必要的图像和标签文件

### 在新项目中使用

```python
# 推荐路径
DATA_PATH = "/data2/hmy/5Center_datas/5centers_multi_internal_external_final"

# 或在新项目中（需要创建软链接）
DATA_PATH = "/data2/hmy/VLM_Caus_Rm/data/5centers_multi_internal_external_final"
```

---

## 📋 数据大小对比总结

| 项目 | 原始数据集 | 最终划分版本 | 推荐版本 |
|------|-----------|-------------|---------|
| **总大小** | 182 GB | 1.2 GB | 642 MB |
| **文件数** | 121,400 | 1,546 | 496 |
| **目录数** | 1,977 | 487 | 160 |
| **是否推荐** | 完整数据源 | ✅ **推荐使用** | ❌ 不推荐 |

---

## ⚠️ 注意事项

1. **原始数据集很大**: 182 GB，包含所有原始数据
2. **划分版本较小**: 1.2 GB，只包含划分后的数据
3. **软链接使用**: 在新项目中可以使用软链接，避免重复占用空间
4. **磁盘空间**: 当前磁盘使用率92%，剩余294 GB，足够使用

---

## 🔧 创建软链接（在新项目中）

```bash
cd /data2/hmy/VLM_Caus_Rm/data

# 创建最终版本的软链接
ln -s /data2/hmy/5Center_datas/5centers_multi_internal_external_final \
      5centers_multi_internal_external_final

# 验证
ls -lh 5centers_multi_internal_external_final
```

---

## 📊 内存占用估算（训练时）

### 数据加载到内存

假设使用以下配置：
- Batch size: 10
- OCT frames: 120
- Image size: 224x224
- 数据类型: float32

**单batch内存占用**:
- OCT: 10 × 120 × 3 × 224 × 224 × 4 bytes ≈ 1.4 GB
- Colposcopy: 10 × 3 × 3 × 224 × 224 × 4 bytes ≈ 0.02 GB
- Clinical: 10 × 7 × 4 bytes ≈ 0.0003 GB
- **总计**: ~1.42 GB per batch

**训练时总内存**:
- 数据加载: ~1.5 GB
- 模型参数: ~0.3 GB (78M参数)
- 优化器状态: ~0.6 GB
- 中间激活: ~2-4 GB
- **总计**: ~4-6 GB (FP32) 或 ~2-3 GB (FP16)

---

## ✅ 总结

1. **原始数据集**: 182 GB，包含完整数据
2. **最终划分版本**: 1.2 GB，**推荐使用** ✅
3. **推荐版本**: 642 MB，不推荐使用 ❌
4. **磁盘空间**: 充足（剩余294 GB）
5. **训练内存**: 约4-6 GB (FP32) 或 2-3 GB (FP16)

**建议**: 使用 `5centers_multi_internal_external_final` 作为最终数据集划分。

---

**最后更新**: 2025-12-17

