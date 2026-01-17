<!--
文件生成信息:
- 生成时间: 2025-12-25 11:10:00 (使用nvidia-smi获取CUDA时间)
- 生成需求: 用户要求将多个目录移动到Trash
- 生成原因: 分析这些目录的使用情况，确保移动操作的安全性
- 相关任务: 项目清理

文件功能: 分析待移动目录的使用情况
-->

# 目录清理分析报告

## 📊 待移动目录列表

| 目录 | 大小 | 状态 | 使用情况 |
|------|------|------|----------|
| `figures/` | 2.9M | ⚠️ | 可能包含可视化结果 |
| `models/` | 88K | ⚠️ | **正在使用** - 训练脚本导入`models.SwinT.*` |
| `util/` | 8.0K | ✅ | 可能未使用 |
| `utils/` | 212K | ⚠️ | **正在使用** - 训练脚本导入`utils.*` |
| `visualization/` | 156K | ✅ | 可能未使用 |
| `configs/` | 8.0K | ⚠️ | 可能包含配置文件 |
| `src/` | 708K | ⚠️ | **正在使用** - 训练脚本导入`src.models.*`和`src.data.*` |
| `training/` | 40K | ✅ | 可能未使用 |

## ⚠️ 重要警告

### 正在使用的目录

以下目录**正在被训练脚本使用**，移动后可能导致训练失败：

1. **`src/`** (708K)
   - 被导入: `from src.models.*`, `from src.data.*`
   - 包含: `src/models/`, `src/data/` (包含`enhanced_multimodal_dataset.py`)

2. **`models/`** (88K)
   - 被导入: `from models.SwinT.swin_image_encoder import SwinTImageEncoder`
   - 包含: `SwinT/`, `ViT/`, `MedicalViT/`

3. **`utils/`** (212K)
   - 被导入: `from utils.enhanced_multimodal_dataset import *`
   - 被导入: `from utils.advanced_clinical_metrics import *`

### 使用这些目录的训练脚本

- `experiments/exp1_causal_bayesian_clip/train.py`
- `experiments/exp1_causal_bayesian_clip/train_vlm.py`
- `experiments/exp1_causal_bayesian_clip/train_causal_bayesian.py`
- `experiments/exp1_causal_bayesian_clip/train_legacy.py`
- `experiments/exp1_causal_bayesian_clip/train_innovations.py`
- `experiments/exp1_causal_bayesian_clip/train_adaptive.py`
- `experiments/exp1_causal_bayesian_clip/train_swin_optimized.py`
- `experiments/baseline/train_swin_baseline.py`

## ✅ 执行计划

根据用户要求，将执行以下操作：

1. **移动到Trash**（而不是直接删除，保留备份）
2. **记录移动操作**，便于后续恢复

## 📝 操作记录

执行时间: 2025-12-25 11:10:00

