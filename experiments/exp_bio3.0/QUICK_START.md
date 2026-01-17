# Bio-COT 3.0 快速开始指南

## 🚀 5分钟快速开始

### Step 1: 构建医学知识库

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0
python knowledge_base/build_knowledge_base.py \
    --output knowledge_base/medical_guidelines.json
```

**预期输出**：
```
✅ 医学知识库已构建完成！
   文件路径: knowledge_base/medical_guidelines.json
   条目数量: 12
   文件大小: 5.23 KB
```

### Step 2: 验证知识库

```python
import json
with open('knowledge_base/medical_guidelines.json', 'r') as f:
    kb = json.load(f)
print(f"知识库条目数: {len(kb)}")
print(f"示例键: {list(kb.keys())[:3]}")
```

### Step 3: 测试模型（待实现训练脚本后）

```bash
# 待创建训练脚本后运行
python training/train_bio_cot_v3.py \
    --data_root /data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal \
    --knowledge_base knowledge_base/medical_guidelines.json \
    --output_dir results \
    --batch_size 16 \
    --num_epochs 100
```

---

## 📁 文件结构

```
exp_bio3.0/
├── README.md                          # 项目说明
├── ARCHITECTURE.md                    # 架构文档
├── IMPLEMENTATION_ROADMAP.md          # 实施路线图
├── QUICK_START.md                     # 本文件
├── config.py                          # 配置文件
├── knowledge_base/
│   ├── medical_guidelines.json        # 医学知识库（需构建）
│   └── build_knowledge_base.py       # 构建脚本
├── models/
│   ├── __init__.py
│   ├── bio_cot_v3.py                 # ✅ 主模型
│   ├── knowledge_notes.py             # ✅ 知识笔记模块
│   └── visual_notes.py                # ✅ 视觉笔记模块
├── data/
│   └── __init__.py
├── training/
│   └── __init__.py                    # ⬜ 待创建训练脚本
├── evaluation/
│   └── __init__.py                    # ⬜ 待创建评估脚本
├── checkpoints/                       # 模型检查点
├── logs/                              # 训练日志
└── results/                           # 实验结果
    ├── knowledge_notes/               # 知识笔记可视化
    ├── visual_notes/                  # 视觉笔记可视化
    └── metrics/                       # 性能指标
```

---

## ✅ 已完成的工作

- [x] 创建项目结构
- [x] 创建配置文件 (`config.py`)
- [x] 创建知识库构建脚本 (`knowledge_base/build_knowledge_base.py`)
- [x] 创建知识笔记模块 (`models/knowledge_notes.py`)
- [x] 创建视觉笔记模块 (`models/visual_notes.py`)
- [x] 创建Bio-COT v3主模型 (`models/bio_cot_v3.py`)
- [x] 创建文档（README, ARCHITECTURE, ROADMAP）

## ⬜ 待完成的工作

- [ ] 创建数据集类（支持知识库）
- [ ] 创建训练脚本 (`training/train_bio_cot_v3.py`)
- [ ] 创建评估脚本 (`evaluation/evaluate_v3.py`)
- [ ] 创建可视化脚本 (`evaluation/visualize_notes.py`)
- [ ] 测试模型前向传播
- [ ] 运行消融实验

---

## 🔧 下一步行动

1. **立即行动**：构建知识库
   ```bash
   python knowledge_base/build_knowledge_base.py
   ```

2. **创建训练脚本**：参考`exp_5centers/train_bio_cot_v2.py`

3. **测试模型**：创建简单的测试脚本验证模型功能

4. **开始训练**：运行第一个实验（Baseline vs Knowledge Notes）

---

**提示**：详细实施步骤请参考 `IMPLEMENTATION_ROADMAP.md`

