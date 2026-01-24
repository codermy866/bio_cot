# Bio-COT 3.0 执行检查清单

## 📋 按天执行的详细步骤

### Day 1: 构建知识库和生成Knowledge Note Embeddings

#### ✅ Step 1.1: 构建医学指南库

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

python knowledge_base/build_knowledge_base.py \
    --output knowledge_base/medical_guidelines.json
```

**验证**：
```bash
# 检查文件是否存在
ls -lh knowledge_base/medical_guidelines.json

# 检查内容
python -c "import json; kb = json.load(open('knowledge_base/medical_guidelines.json')); print(f'条目数: {len(kb)}'); print(f'示例键: {list(kb.keys())[:3]}')"
```

**预期输出**：
- 文件大小：~5 KB
- 条目数量：12条

---

#### ✅ Step 1.2: 生成Knowledge Note Embeddings

```bash
python knowledge_base/generate_knowledge_notes.py \
    --csv_paths \
        /data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal/internal_train/labels.csv \
        /data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal/internal_val/labels.csv \
    --guidelines knowledge_base/medical_guidelines.json \
    --output data/knowledge_embeddings.npy \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
    --device cuda:1 \
    --batch_size 32
```

**验证**：
```bash
# 检查文件
ls -lh data/knowledge_embeddings.npy
ls -lh data/knowledge_embeddings.json

# 检查维度
python -c "import numpy as np; emb = np.load('data/knowledge_embeddings.npy'); print(f'形状: {emb.shape}'); print(f'维度: {emb.shape[1]}')"
```

**预期输出**：
- 嵌入形状：`(N, 768)`，N为样本数
- 文件大小：~3-5 MB（取决于样本数）

---

### Day 2: 测试VisualNoteLayer

#### ✅ Step 2.1: 创建并运行测试脚本

创建文件：`training/test_visual_note_layer.py`（已在COMPLETE_IMPLEMENTATION_GUIDE.md中提供）

```bash
python training/test_visual_note_layer.py
```

**预期输出**：
```
输入形状: img_feats torch.Size([4, 196, 768]), text_feats torch.Size([4, 768])
输出形状: img_focused torch.Size([4, 196, 768]), attn_map torch.Size([4, 196, 1])
✅ VisualNoteLayer测试通过！
```

---

### Day 3: 测试模型前向传播

#### ✅ Step 3.1: 创建并运行模型测试

创建文件：`training/test_model_forward.py`（已在COMPLETE_IMPLEMENTATION_GUIDE.md中提供）

```bash
python training/test_model_forward.py
```

**预期输出**：
```
输出键: ['pred', 'z_causal', 'z_sem', 'z_noise', 'attn_maps', 'loss_components']
pred形状: torch.Size([4, 2])
z_causal形状: torch.Size([4, 768])
z_sem形状: torch.Size([4, 768])
attn_maps数量: 2
attn_oct形状: torch.Size([4, 196, 1])
attn_colpo形状: torch.Size([4, 196, 1])
损失组件: ['L_ot', 'L_sparse', 'L_consist', 'L_adv']
✅ 模型前向传播测试通过！
```

---

### Day 4: 开始训练

#### ✅ Step 4.1: 检查配置文件

编辑 `config.py`，确保路径正确：
```python
data_root = '/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal'
knowledge_embed_path = 'data/knowledge_embeddings.npy'
```

#### ✅ Step 4.2: 运行训练

```bash
python training/train_bio_cot_v3.py
```

**监控指标**：
- 训练损失（应该逐渐下降）
- 验证AUC（应该逐渐上升）
- 稀疏性损失（应该逐渐减小）
- Beta值（应该按Warm-up策略变化）

---

## 🔍 关键验证点

### 验证点1: Knowledge Note生成

**检查**：生成的Note文本是否合理
```python
import json
with open('data/knowledge_embeddings.json', 'r') as f:
    data = json.load(f)
    print(f"示例Note文本: {list(data['note_texts'].values())[0][:200]}...")
```

### 验证点2: Visual Note Masking

**检查**：Attention Map是否合理
- 大部分值应该接近0（背景）
- 少数值应该接近1（病灶区域）
- 稀疏性损失应该逐渐减小

### 验证点3: 动态Beta

**检查**：Beta值是否按策略变化
- Epoch 1-5: Beta = 1.0
- Epoch 6-20: Beta线性递减
- Epoch 21+: Beta = 0.1

---

## 📊 预期训练曲线

### 训练损失曲线
- **初期**（Epoch 1-5）：损失较高，因为Beta=1.0（全图保留）
- **中期**（Epoch 6-20）：损失逐渐下降，Beta线性递减
- **后期**（Epoch 21+）：损失稳定，Beta=0.1（强过滤）

### 验证AUC曲线
- **预期**：逐渐上升，最终达到0.87-0.89

### 稀疏性损失曲线
- **预期**：逐渐减小，说明注意力越来越聚焦

---

## ✅ 完成标准

### Day 1完成标准
- [x] 知识库文件存在且格式正确
- [x] Knowledge Note Embeddings文件存在
- [x] 嵌入维度正确（768）
- [x] ID映射文件存在

### Day 2完成标准
- [x] VisualNoteLayer测试通过
- [x] 输入输出形状正确
- [x] 不同Beta值测试通过

### Day 3完成标准
- [x] 模型前向传播测试通过
- [x] 所有输出键存在
- [x] 损失组件计算正确

### Day 4完成标准
- [x] 训练脚本运行成功
- [x] 第一个epoch完成
- [x] 动态Beta策略工作正常
- [x] 稀疏性损失计算正确

---

**文档版本**：v1.0  
**最后更新**：2025-01-08

