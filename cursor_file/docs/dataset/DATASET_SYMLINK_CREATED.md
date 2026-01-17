<!--
文件生成信息:
- 生成时间: 2025-12-25 11:12:00 (使用nvidia-smi获取CUDA时间)
- 生成需求: 用户在data目录下没有看到最新划分的数据集的软链接
- 生成原因: 需要在data目录下创建指向Leave-Centers-Out数据集的软链接
- 相关任务: 数据集软链接创建

文件功能: 记录Leave-Centers-Out数据集软链接的创建情况
-->

# Leave-Centers-Out数据集软链接创建完成

## ✅ 已创建的软链接

### 位置
`/data2/hmy/VLM_Caus_Rm_Mics/data/5centers_multi_leave_centers_out`

### 指向
`/data2/hmy/5Center_datas/5centers_multi_leave_centers_out`

---

## 📊 data目录下的所有软链接

现在 `data/` 目录下共有3个软链接：

| 软链接名称 | 指向路径 | 用途 |
|-----------|----------|------|
| `5centers_multi` | `/data2/hmy/5Center_datas/5centers_multi` | 原始数据集（182GB） |
| `5centers_multi_internal_external_final` | `/data2/hmy/5Center_datas/5centers_multi_internal_external_final` | 旧的内外部划分数据集 |
| `5centers_multi_leave_centers_out` | `/data2/hmy/5Center_datas/5centers_multi_leave_centers_out` | **最新的Leave-Centers-Out数据集** ✅ |

---

## ✅ 验证结果

### 软链接状态
- ✅ 软链接已创建
- ✅ 目标路径存在
- ✅ 可以正常访问

### 数据集内容检查
- ✅ `train_labels.csv` - 训练集标签文件（669 samples）
- ✅ `val_labels.csv` - 验证集标签文件（168 samples）
- ✅ `external_test_labels.csv` - 外部测试集标签文件（148 samples）
- ✅ `internal_train/` - 内部训练和验证集目录
- ✅ `external_validation/` - 外部测试集目录

---

## 🎯 使用方式

### 方式1: 使用完整路径（当前训练脚本使用）
```python
data_path = '/data2/hmy/5Center_datas/5centers_multi_leave_centers_out'
```

### 方式2: 使用data目录下的软链接（推荐）
```python
from pathlib import Path
project_root = Path('/data2/hmy/VLM_Caus_Rm_Mics')
data_path = str(project_root / 'data' / '5centers_multi_leave_centers_out')
```

### 方式3: 使用相对路径（如果在项目根目录）
```python
data_path = 'data/5centers_multi_leave_centers_out'
```

---

## 📝 训练脚本中的使用

当前训练脚本使用的路径：
- `train.py`: `/data2/hmy/5Center_datas/5centers_multi_leave_centers_out`
- `train_vlm.py`: `/data2/hmy/5Center_datas/5centers_multi_leave_centers_out`
- `train_causal_bayesian.py`: `/data2/hmy/5Center_datas/5centers_multi_leave_centers_out`

**注意**: 这些路径仍然有效，因为软链接指向同一个数据集。如果需要，可以更新训练脚本使用 `data/5centers_multi_leave_centers_out` 相对路径。

---

## 🔍 验证命令

```bash
# 查看软链接
ls -la /data2/hmy/VLM_Caus_Rm_Mics/data/

# 验证软链接指向
readlink -f /data2/hmy/VLM_Caus_Rm_Mics/data/5centers_multi_leave_centers_out

# 检查数据集内容
ls -la /data2/hmy/VLM_Caus_Rm_Mics/data/5centers_multi_leave_centers_out/
```

---

**创建完成日期**: 2025-12-25 11:12:00  
**状态**: ✅ **软链接已成功创建并验证**

