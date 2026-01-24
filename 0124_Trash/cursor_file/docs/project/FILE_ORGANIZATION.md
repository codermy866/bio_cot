<!--
文件生成信息:
- 生成时间: 2025-12-27
- 生成需求: 整理experiments文件夹，将无关训练的文件移动到cursor_file
- 生成原因: 用户要求清理experiments文件夹，只保留训练相关文件

文件功能: 记录文件整理情况
-->

# 📁 文件整理报告

**整理时间**: 2025-12-27  
**状态**: ✅✅✅ **已完成文件整理**

---

## 📋 整理规则

### 保留在 `experiments/` 文件夹中的文件

以下文件与训练参数或数据相关，**保留在原位置**：

1. **训练脚本**:
   - `train_causal_bayesian.py` - 主训练脚本（当前使用）
   - `train.py` - 训练脚本
   - `train_adaptive.py` - 自适应训练脚本
   - `train_innovations.py` - 创新训练脚本
   - `train_legacy.py` - 遗留训练脚本
   - `train_swin_optimized.py` - Swin优化训练脚本
   - `train_vlm.py` - VLM训练脚本

2. **启动脚本**:
   - `quick_start_multicenter.sh` - 快速启动训练脚本

3. **Python包文件**:
   - `__init__.py` - Python包初始化文件

---

### 移动到 `cursor_file/` 文件夹中的文件

以下文件与训练参数或数据无关，**已移动到cursor_file**：

1. **调试脚本** (移动到 `cursor_file/debug_scripts/`):
   - `debug_data_loading.py` - 数据加载调试脚本
   - `quick_test_data_fix.py` - 快速测试数据修复脚本

---

## 📊 整理结果

### 移动的文件

| 原路径 | 新路径 | 类型 |
|--------|--------|------|
| `experiments/exp1_causal_bayesian_clip/debug_data_loading.py` | `cursor_file/debug_scripts/debug_data_loading.py` | 调试脚本 |
| `experiments/exp1_causal_bayesian_clip/quick_test_data_fix.py` | `cursor_file/debug_scripts/quick_test_data_fix.py` | 测试脚本 |

### 保留的文件

| 文件 | 说明 |
|------|------|
| `train_causal_bayesian.py` | 主训练脚本（当前使用） |
| `train.py` | 训练脚本 |
| `train_adaptive.py` | 自适应训练脚本 |
| `train_innovations.py` | 创新训练脚本 |
| `train_legacy.py` | 遗留训练脚本 |
| `train_swin_optimized.py` | Swin优化训练脚本 |
| `train_vlm.py` | VLM训练脚本 |
| `quick_start_multicenter.sh` | 快速启动脚本 |
| `__init__.py` | Python包文件 |

---

## 📁 目录结构

### experiments/exp1_causal_bayesian_clip/

```
experiments/exp1_causal_bayesian_clip/
├── __init__.py                    # Python包文件
├── train_causal_bayesian.py       # 主训练脚本 ⭐
├── train.py                       # 训练脚本
├── train_adaptive.py              # 自适应训练脚本
├── train_innovations.py           # 创新训练脚本
├── train_legacy.py                # 遗留训练脚本
├── train_swin_optimized.py        # Swin优化训练脚本
├── train_vlm.py                   # VLM训练脚本
├── quick_start_multicenter.sh     # 快速启动脚本
├── exp_multicenter_alignment/     # 训练结果目录
├── exp_multicenter_alignment_fixed/ # 训练结果目录
└── oct_cache_optimized/           # OCT缓存目录
```

### cursor_file/debug_scripts/

```
cursor_file/debug_scripts/
├── debug_data_loading.py          # 数据加载调试脚本
└── quick_test_data_fix.py         # 快速测试数据修复脚本
```

---

## ✅ 整理完成

- ✅ 已移动2个调试/测试脚本到 `cursor_file/debug_scripts/`
- ✅ 保留了所有训练相关脚本在 `experiments/` 文件夹
- ✅ 文件结构更加清晰，便于管理

---

**整理完成日期**: 2025-12-27  
**核心结论**: ✅✅✅ **文件整理完成，experiments文件夹现在只包含训练相关的脚本和文件。**

