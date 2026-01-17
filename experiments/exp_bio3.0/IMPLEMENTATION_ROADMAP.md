# Bio-COT 3.0 实施路线图

## 📋 实施步骤

### 第一步：数据准备

#### 1.1 构建医学知识库

```bash
cd experiments/exp_bio3.0
python knowledge_base/build_knowledge_base.py \
    --output knowledge_base/medical_guidelines.json
```

**预期输出**：
- `knowledge_base/medical_guidelines.json`：包含医学指南的JSON文件

**验证**：
```python
import json
with open('knowledge_base/medical_guidelines.json', 'r') as f:
    kb = json.load(f)
print(f"知识库条目数: {len(kb)}")
print(f"示例: {list(kb.keys())[:3]}")
```

---

### 第二步：特征提取（保持ViT不变）

#### 2.1 使用现有的ViT特征提取

Bio-COT 3.0使用与v2.0相同的ViT特征提取方法：

```python
from experiments.exp_5centers.train_bio_cot_5centers_multimodal import extract_features_with_vit

# 提取OCT特征
oct_features = extract_features_with_vit(oct_images, device)  # [B, 768]

# 提取Colposcopy特征
colpo_features = extract_features_with_vit(colposcopy_images, device)  # [B, 768]
```

**注意**：如果需要Patch特征（用于Visual Notes），需要修改ViT提取函数。

---

### 第三步：代码修改

#### 3.1 创建Bio-COT v3模型

✅ **已完成**：`models/bio_cot_v3.py`

**关键修改点**：
1. 添加`KnowledgeNotesModule`：生成知识笔记
2. 添加`VisualNotesModule`：生成视觉笔记
3. 修改前向传播流程：
   - 先生成知识笔记 z_sem
   - 再生成视觉笔记 F_note
   - 最后进行因果解耦和分类

#### 3.2 修改Forward函数流程

```python
# 新流程：
1. Knowledge Notes: z_sem = KnowledgeNotesModule(clinical_data)
2. Visual Notes: F_note, mask = VisualNotesModule(F_img, z_sem)
3. Causal Decoupling: z_causal, z_noise = DualHead(F_note)
4. Fusion: fused_feat = Fusion(z_causal, z_sem)
5. Classification: logits = Classifier(fused_feat)
```

---

### 第四步：消融实验

#### 4.1 实验1: Knowledge Notes vs Raw Clinical Data

**配置A（Baseline）**：
```python
use_knowledge_notes = False
use_visual_notes = False
```

**配置B（+ Knowledge Notes）**：
```python
use_knowledge_notes = True
use_visual_notes = False
```

**预期结果**：
- Knowledge Notes提升 +1-2% AUC

#### 4.2 实验2: Visual Note Masking vs Global Attention

**配置C（+ Visual Notes）**：
```python
use_knowledge_notes = True
use_visual_notes = True
```

**预期结果**：
- Visual Notes提升 +2-3% AUC

#### 4.3 实验3: 完整Bio-COT 3.0

**配置D（Full v3.0）**：
```python
use_knowledge_notes = True
use_visual_notes = True
use_ot = True
use_dual = True
use_cross_attn = True
```

**预期结果**：
- 总体提升 +3-5% AUC

---

## 🚀 快速开始脚本

### 创建训练脚本模板

**文件**：`training/train_bio_cot_v3.py`（待创建）

**关键步骤**：
1. 加载配置
2. 创建模型
3. 训练循环（注意：每个epoch调用`model.set_epoch(epoch)`）
4. 验证（使用优化后的预测）

---

## 📊 预期时间表

| 步骤 | 内容 | 时间 | 状态 |
|------|------|------|------|
| Step 1 | 构建知识库 | 0.5天 | ⬜ 待实施 |
| Step 2 | 特征提取（复用） | 0天 | ✅ 已完成 |
| Step 3 | 代码修改 | 2-3天 | ⬜ 待实施 |
| Step 4 | 消融实验 | 3-5天 | ⬜ 待实施 |
| **总计** | - | **5-8天** | - |

---

## ✅ 检查清单

### 数据准备
- [ ] 构建医学知识库
- [ ] 验证知识库格式
- [ ] 测试知识检索功能

### 模型实现
- [x] 创建KnowledgeNotesModule
- [x] 创建VisualNotesModule
- [x] 创建BioCOT_v3主模型
- [ ] 测试模型前向传播
- [ ] 测试损失计算

### 训练脚本
- [ ] 创建训练脚本
- [ ] 实现Warm-up策略
- [ ] 实现稀疏性损失
- [ ] 实现可视化功能

### 评估脚本
- [ ] 创建评估脚本
- [ ] 实现知识笔记可视化
- [ ] 实现视觉笔记可视化
- [ ] 实现性能指标计算

---

**文档版本**：v1.0  
**最后更新**：2025-01-08

