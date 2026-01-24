# 项目重组使用指南

## 📋 一、分析依据说明

### 我分析时使用的关键文件：

1. **核心文档**（理解项目整体）：
   - `DEEP_TECHNICAL_ANALYSIS.md` - 技术深度分析（1327行）
   - `TECHNICAL_WORK_SUMMARY.md` - 技术工作总结
   - `README.md` - 项目概述
   - `exp1_Causal_Bayesian_clip/README_exp1.md` - 实验详细说明

2. **模型架构文件**（理解技术实现）：
   - `exp1_Causal_Bayesian_clip/code/enhanced_causal_clip.py` - 核心模型
   - `src/models/causal_bayesian_clip_framework.py` - 因果贝叶斯CLIP框架
   - `models/cnn_multimodal_model.py` - CNN多模态模型
   - `models/vmamba_multimodal_model.py` - VMamba模型
   - `models/SwinT/` - Swin-T模型

3. **训练脚本**（理解工作流程）：
   - `exp1_Causal_Bayesian_clip/code/train_enhanced_causal_clip.py` - 主训练脚本
   - `training/train_enhanced_causal_clip.py` - 训练脚本
   - `src/train/train_causal_bayesian_clip.py` - 训练入口

4. **项目结构文档**：
   - `PROJECT_STRUCTURE.md` - 项目结构说明
   - `docs/` - 各种技术文档

---

## 🚀 二、如何使用重组脚本

### 步骤1：查看重组计划

```bash
# 查看详细的重组计划
cat PROJECT_REORGANIZATION_PLAN.md
```

### 步骤2：备份当前项目（重要！）

```bash
# 脚本会自动创建备份，但建议您也手动备份
cp -r /data2/hmy/VLM_Caus_Rm /data2/hmy/VLM_Caus_Rm_backup_$(date +%Y%m%d)
```

### 步骤3：运行重组脚本

```bash
cd /data2/hmy/VLM_Caus_Rm

# 运行重组脚本
python reorganize_project.py

# 或者指定项目路径
python reorganize_project.py /data2/hmy/VLM_Caus_Rm
```

### 步骤4：检查重组结果

```bash
# 查看新的目录结构
tree -L 3 -I '__pycache__|*.pyc|backup*'

# 检查关键文件是否移动成功
ls -la src/models/
ls -la experiments/exp1_causal_bayesian_clip/
```

### 步骤5：更新导入路径

重组后，您需要更新一些导入路径。脚本会生成迁移报告：

```bash
# 查看迁移报告
cat REORGANIZATION_REPORT.json
```

**需要手动更新的常见导入**：

```python
# 旧导入
from utils.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
from models.cnn_multimodal_model import CNNMultimodalTransformer

# 新导入
from src.data.dataset import EnhancedMultimodalCervicalDataset
from src.models.backbones.cnn_encoder import CNNMultimodalTransformer
```

---

## 📁 三、重组后的标准结构

```
VLM_Caus_Rm/
├── README.md                    # 项目主README
├── README_GITHUB.md            # GitHub README（英文）
├── LICENSE                      # 开源许可证
├── requirements.txt            # 依赖包
├── setup.py                    # 安装脚本
├── .gitignore                  # Git忽略文件
│
├── src/                        # 源代码（核心代码库）
│   ├── models/                 # 模型定义
│   │   ├── backbones/          # 基础编码器
│   │   ├── fusion/             # 融合模块
│   │   ├── causal/             # 因果推理
│   │   └── uncertainty/        # 不确定性量化
│   ├── data/                   # 数据处理
│   ├── training/               # 训练相关
│   ├── evaluation/             # 评估相关
│   └── utils/                  # 工具函数
│
├── experiments/                # 实验代码
│   ├── exp1_causal_bayesian_clip/  # 主实验
│   └── baseline/              # 基线方法
│
├── scripts/                     # 启动脚本
├── configs/                     # 配置文件
├── results/                     # 实验结果
├── figures/                     # 论文图表
├── tests/                       # 测试代码
└── docs/                        # 文档
```

---

## ⚠️ 四、注意事项

### 1. 备份重要

重组脚本会自动创建备份到 `backup_before_reorganize/`，但建议您也手动备份整个项目。

### 2. 导入路径更新

重组后，所有导入路径需要更新。建议：
- 使用IDE的全局替换功能
- 或者使用脚本批量替换

### 3. 测试验证

重组后，务必测试：
```bash
# 测试导入
python -c "from src.models.enhanced_causal_clip import EnhancedCausalBayesianCLIP; print('✅ 导入成功')"

# 测试训练脚本
python experiments/exp1_causal_bayesian_clip/train.py --help
```

### 4. 逐步迁移

如果项目很大，建议：
1. 先重组核心代码（`src/`）
2. 再重组实验代码（`experiments/`）
3. 最后整理文档和结果

---

## 🔧 五、常见问题

### Q1: 重组后训练脚本报错"找不到模块"

**A**: 需要更新导入路径。检查：
```python
# 确保项目根目录在Python路径中
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
```

### Q2: 如何恢复重组前的结构？

**A**: 从备份恢复：
```bash
cp -r backup_before_reorganize/* .
```

### Q3: 有些文件没有被移动？

**A**: 检查 `REORGANIZATION_REPORT.json`，查看哪些文件被跳过。可能需要手动移动。

---

## 📝 六、后续工作

重组完成后，建议：

1. **更新README**
   - 更新项目结构说明
   - 更新快速开始指南

2. **创建配置文件**
   - `configs/default.yaml` - 默认配置
   - `configs/exp1_config.yaml` - 实验配置

3. **编写测试**
   - `tests/test_models.py` - 模型测试
   - `tests/test_training.py` - 训练测试

4. **准备GitHub**
   - 完善README_GITHUB.md
   - 添加示例代码
   - 准备发布说明

---

## 📞 需要帮助？

如果遇到问题，请：
1. 查看 `REORGANIZATION_REPORT.json`
2. 检查备份目录
3. 查看错误日志

---

**祝您项目重组顺利！** 🎉

