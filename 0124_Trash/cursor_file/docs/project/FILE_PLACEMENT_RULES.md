<!--
文件生成信息:
- 生成时间: 2025-12-27
- 生成需求: 建立文件放置规则，确保所有生成的文件都放在正确的位置
- 生成原因: 用户要求以后所有生成的文件都要放在对应的文件夹内
- 相关任务: 文件分类规则制定

文件功能: 记录文件放置规则，供AI助手和开发者参考
-->

# 📋 文件放置规则

**制定时间**: 2025-12-27  
**状态**: ✅✅✅ **规则已建立，必须严格遵守**

---

## 🎯 核心原则

**所有生成的文件必须按照类型分类，放在对应的文件夹内，禁止在根目录或experiments文件夹中乱放！**

---

## 📁 文件分类规则

### 1. 文档文件 (docs/)

所有 `.md` 文档文件必须根据内容分类到对应的子目录：

#### `docs/analysis/` - 分析报告
- **放置内容**: 深度分析报告、任务分析、背景分析、创新分析等
- **文件示例**:
  - `CCF_A_TASK_ANALYSIS.md`
  - `CLINICAL_BACKGROUND_ANALYSIS.md`
  - `ORIGINAL_INNOVATION_ANALYSIS.md`
  - `PROJECT_LOGIC_ANALYSIS.md`

#### `docs/training/` - 训练相关文档
- **放置内容**: 训练状态、训练报告、训练问题修复、批次大小调整、内存分析等
- **文件示例**:
  - `TRAINING_*.md`
  - `BATCH_SIZE_*.md`
  - `MEMORY_ANALYSIS.md`
  - `DATA_LOADING_FIXES.md`
  - `CLASS_DETECTION_FIX_VERIFIED.md`

#### `docs/dataset/` - 数据集相关文档
- **放置内容**: 数据集分析、数据集设置、数据集划分、数据路径验证等
- **文件示例**:
  - `DATASET_*.md`
  - `DATA_*.md`
  - `LEAVE_CENTERS_OUT_SUMMARY.md`
  - `SCIENTIFIC_DATASET_SPLIT_SUMMARY.md`

#### `docs/miccai/` - MICCAI相关文档
- **放置内容**: MICCAI论文相关文档、实验计划、故事线等
- **文件示例**:
  - `MICCAI_*.md`
  - `NEW_EXPERIMENT_PLAN_EVALUATION.md`
  - `DETAILED_EXPERIMENT_PLAN.md`

#### `docs/project/` - 项目组织文档
- **放置内容**: 文件组织、项目清理、目录重组、项目结构等
- **文件示例**:
  - `FILE_*.md`
  - `PROJECT_*.md`
  - `*_CLEANUP_*.md`
  - `REORGANIZATION_*.md`
  - `DIRECTORY_*.md`

---

### 2. 脚本文件 (scripts/)

所有 `.py` 脚本文件必须根据功能分类到对应的子目录：

#### `scripts/dataset/` - 数据集处理脚本
- **放置内容**: 数据集创建、数据集处理、数据集检查等脚本
- **文件示例**:
  - `create_*.py`
  - `redesign_*.py`
  - `check_dataset_*.py`

#### `scripts/utils/` - 工具脚本
- **放置内容**: 通用工具脚本，如时间戳、文件操作、查找等
- **文件示例**:
  - `get_timestamp.py`
  - `find_empty_dirs.py`
  - `move_*.py`

#### `scripts/visualization/` - 可视化脚本
- **放置内容**: 数据可视化、结构可视化等脚本
- **文件示例**:
  - `visualize_*.py`
  - `create_links_*.py`

---

### 3. 配置文件 (config/)

- **放置内容**: 所有配置文件
- **文件示例**:
  - `project_config.py`
  - `visualization_config.py`
  - `*.config`
  - `*.yaml`
  - `*.json` (配置文件)

---

### 4. 图片文件 (images/)

- **放置内容**: 所有图片和PDF文件
- **文件示例**:
  - `*.png`
  - `*.jpg`
  - `*.jpeg`
  - `*.pdf`
  - `*.svg`

---

### 5. 调试脚本 (debug_scripts/)

- **放置内容**: 调试和测试相关脚本
- **文件示例**:
  - `debug_*.py`
  - `test_*.py`
  - `quick_test_*.py`

---

## 🚫 禁止行为

### ❌ 禁止在以下位置放置文件：

1. **cursor_file根目录** (除了README.md)
   - ❌ 禁止直接放置 `.md` 文件
   - ❌ 禁止直接放置 `.py` 文件
   - ❌ 禁止直接放置图片文件

2. **experiments文件夹** (除了训练相关文件)
   - ❌ 禁止放置调试脚本
   - ❌ 禁止放置测试脚本
   - ❌ 禁止放置文档文件
   - ✅ 只保留训练脚本 (`train_*.py`) 和启动脚本 (`*.sh`)

3. **项目根目录**
   - ❌ 禁止放置临时文件
   - ❌ 禁止放置文档文件
   - ❌ 禁止放置脚本文件（除非是项目核心文件）

---

## ✅ 正确做法

### 生成新文件时的步骤：

1. **确定文件类型**
   - 文档 → `docs/` 子目录
   - 脚本 → `scripts/` 子目录
   - 配置 → `config/`
   - 图片 → `images/`
   - 调试 → `debug_scripts/`

2. **确定具体子目录**
   - 根据文件内容选择对应的子目录
   - 如果不确定，参考现有文件的分类

3. **使用完整路径生成文件**
   - ✅ `cursor_file/docs/training/TRAINING_REPORT.md`
   - ❌ `cursor_file/TRAINING_REPORT.md`

---

## 📝 示例

### ✅ 正确示例：

```python
# 生成训练报告
write(
    file_path="/data2/hmy/VLM_Caus_Rm_Mics/cursor_file/docs/training/TRAINING_REPORT.md",
    contents="..."
)

# 生成数据集分析
write(
    file_path="/data2/hmy/VLM_Caus_Rm_Mics/cursor_file/docs/dataset/DATASET_ANALYSIS.md",
    contents="..."
)

# 生成调试脚本
write(
    file_path="/data2/hmy/VLM_Caus_Rm_Mics/cursor_file/debug_scripts/debug_model.py",
    contents="..."
)
```

### ❌ 错误示例：

```python
# ❌ 错误：放在根目录
write(
    file_path="/data2/hmy/VLM_Caus_Rm_Mics/cursor_file/TRAINING_REPORT.md",
    contents="..."
)

# ❌ 错误：放在experiments文件夹
write(
    file_path="/data2/hmy/VLM_Caus_Rm_Mics/experiments/exp1_causal_bayesian_clip/debug_script.py",
    contents="..."
)
```

---

## 🔄 特殊情况处理

### 如果文件类型不明确：

1. **分析文件内容**：根据文件的主要功能确定分类
2. **参考现有文件**：查看类似文件放在哪里
3. **优先选择最相关的目录**：如果不确定，选择最相关的子目录

### 如果目录不存在：

1. **先创建目录**：使用 `mkdir -p` 创建必要的目录
2. **再生成文件**：在正确的目录中生成文件

---

## 📋 快速参考表

| 文件类型 | 放置位置 | 示例 |
|---------|---------|------|
| 训练报告 | `docs/training/` | `TRAINING_REPORT.md` |
| 数据集分析 | `docs/dataset/` | `DATASET_ANALYSIS.md` |
| MICCAI文档 | `docs/miccai/` | `MICCAI_PAPER.md` |
| 项目组织 | `docs/project/` | `FILE_ORGANIZATION.md` |
| 分析报告 | `docs/analysis/` | `TASK_ANALYSIS.md` |
| 数据集脚本 | `scripts/dataset/` | `create_dataset.py` |
| 工具脚本 | `scripts/utils/` | `get_timestamp.py` |
| 可视化脚本 | `scripts/visualization/` | `visualize_data.py` |
| 配置文件 | `config/` | `project_config.py` |
| 图片文件 | `images/` | `chart.png` |
| 调试脚本 | `debug_scripts/` | `debug_model.py` |

---

## ⚠️ 重要提醒

1. **每次生成文件前，先确定文件类型和放置位置**
2. **使用完整路径，不要使用相对路径**
3. **如果不确定，参考本规则文档**
4. **禁止在根目录或experiments文件夹中乱放文件**

---

**规则制定日期**: 2025-12-27  
**核心原则**: ✅✅✅ **所有文件必须按照类型分类，放在对应的文件夹内！**

