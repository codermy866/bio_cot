# Bio-COT 3.0 完整实施指南

## 📋 按照用户方案实施的完整步骤

### ✅ 已完成的工作

1. **知识库构建脚本**：`knowledge_base/build_knowledge_base.py`
   - 使用简化的Key格式（如`HPV_High_Age_30+`）
   - 包含12条医学指南

2. **Knowledge Note生成脚本**：`knowledge_base/generate_knowledge_notes.py`
   - 离线预处理所有样本
   - 生成Knowledge Note文本
   - 编码为.npy格式

3. **VisualNoteLayer模块**：`models/visual_notes.py`
   - 使用Soft Masking（Sigmoid）
   - 支持动态Beta
   - 支持Warm-up策略

4. **Bio-COT v3主模型**：`models/bio_cot_v3.py`
   - 整合VisualNoteLayer
   - 支持Patch特征输入
   - 返回attention maps用于损失计算

5. **数据集类**：`data/dataset_v3.py`
   - 支持加载.npy格式的Knowledge Note Embeddings
   - 自动匹配patient_id

6. **训练脚本**：`training/train_bio_cot_v3.py`
   - 实现动态Beta策略
   - 实现稀疏性损失
   - 完整的训练循环

---

## 🚀 实施步骤（按顺序执行）

### Day 1: 构建知识库和生成Knowledge Note Embeddings

#### Step 1.1: 构建医学指南库

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

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

#### Step 1.2: 生成Knowledge Note Embeddings（离线预处理）

```bash
python knowledge_base/generate_knowledge_notes.py \
    --csv_paths \
        /data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal/internal_train/labels.csv \
        /data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal/internal_val/labels.csv \
        /data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal/external_test/labels.csv \
    --guidelines knowledge_base/medical_guidelines.json \
    --output data/knowledge_embeddings.npy \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
    --device cuda:1 \
    --batch_size 32
```

**预期输出**：
```
✅ 预处理完成！
   成功: 985 个样本
   失败: 0 个样本
   输出文件: data/knowledge_embeddings.npy
   ID映射文件: data/knowledge_embeddings.json
   嵌入形状: (985, 768)
   文件大小: 2.89 MB
```

**注意**：这个步骤可能需要一些时间（取决于数据集大小和LLM模型）。

---

### Day 2: 测试VisualNoteLayer（单元测试）

#### Step 2.1: 创建测试脚本

**文件**：`training/test_visual_note_layer.py`

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试VisualNoteLayer"""

import torch
from models.visual_notes import VisualNoteLayer

def test_visual_note_layer():
    """测试VisualNoteLayer"""
    B, N, D = 4, 196, 768
    
    # 创建模型
    layer = VisualNoteLayer(img_dim=D, text_dim=D, hidden_dim=256)
    
    # 创建测试数据
    img_feats = torch.randn(B, N, D)
    text_feats = torch.randn(B, D)
    
    # 前向传播
    img_focused, attn_map = layer(img_feats, text_feats, beta=0.1)
    
    print(f"输入形状: img_feats {img_feats.shape}, text_feats {text_feats.shape}")
    print(f"输出形状: img_focused {img_focused.shape}, attn_map {attn_map.shape}")
    
    # 验证形状
    assert img_focused.shape == (B, N, D), f"img_focused形状错误: {img_focused.shape}"
    assert attn_map.shape == (B, N, 1), f"attn_map形状错误: {attn_map.shape}"
    
    print("✅ VisualNoteLayer测试通过！")

if __name__ == '__main__':
    test_visual_note_layer()
```

**运行测试**：
```bash
cd experiments/exp_bio3.0
python training/test_visual_note_layer.py
```

---

### Day 3: 修改Bio-COT主模型，跑通Forward pass

#### Step 3.1: 测试模型前向传播

**文件**：`training/test_model_forward.py`

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试Bio-COT v3模型前向传播"""

import torch
from models.bio_cot_v3 import BioCOT_v3
from config import BioCOT_v3_Config

def test_model_forward():
    """测试模型前向传播"""
    config = BioCOT_v3_Config()
    
    # 创建模型
    model = BioCOT_v3(
        embed_dim=768,
        num_classes=2,
        num_centers=5,
        input_dim=768,
        use_visual_notes=True,
        use_ot=True,
        use_dual=True,
        use_cross_attn=True
    )
    
    # 创建测试数据
    B, N = 4, 196
    f_oct = torch.randn(B, N, 768)
    f_colpo = torch.randn(B, N, 768)
    note_embeds = torch.randn(B, 768)
    center_labels = torch.randint(0, 5, (B,))
    
    # 前向传播
    outputs = model(
        f_oct=f_oct,
        f_colpo=f_colpo,
        note_embeds=note_embeds,
        center_labels=center_labels,
        return_loss_components=True,
        current_beta=0.1
    )
    
    print(f"输出键: {list(outputs.keys())}")
    print(f"pred形状: {outputs['pred'].shape}")
    print(f"z_causal形状: {outputs['z_causal'].shape}")
    print(f"z_sem形状: {outputs['z_sem'].shape}")
    
    if 'attn_maps' in outputs:
        print(f"attn_maps数量: {len(outputs['attn_maps'])}")
        print(f"attn_oct形状: {outputs['attn_maps'][0].shape}")
        print(f"attn_colpo形状: {outputs['attn_maps'][1].shape}")
    
    if 'loss_components' in outputs:
        print(f"损失组件: {list(outputs['loss_components'].keys())}")
    
    print("✅ 模型前向传播测试通过！")

if __name__ == '__main__':
    test_model_forward()
```

**运行测试**：
```bash
python training/test_model_forward.py
```

---

### Day 4: 调整训练脚本，开始训练

#### Step 4.1: 修改配置文件

**文件**：`config.py`

确保以下路径正确：
```python
data_root = '/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal'
knowledge_embed_path = 'data/knowledge_embeddings.npy'  # 预计算的嵌入
```

#### Step 4.2: 开始训练

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

python training/train_bio_cot_v3.py
```

**训练过程**：
- 每个epoch会自动调整Beta（Warm-up策略）
- 会计算稀疏性损失（L_sparse）
- 会保存最佳模型到`checkpoints/`

---

## 📊 关键实现细节

### 1. 动态Beta策略

**实现位置**：`training/train_bio_cot_v3.py` 中的 `get_dynamic_beta()` 函数

```python
def get_dynamic_beta(epoch: int, max_epochs: int = 100) -> float:
    if epoch < 5:
        current_beta = 1.0  # 前5个epoch：全图保留
    elif epoch < 20:
        current_beta = 1.0 - (0.9 * (epoch - 5) / 15)  # 线性衰减
    else:
        current_beta = 0.1  # 强过滤模式
    return current_beta
```

**在训练循环中使用**：
```python
current_beta = get_dynamic_beta(epoch, config.num_epochs)
outputs = model(..., current_beta=current_beta)
```

### 2. 稀疏性损失

**实现位置**：`models/bio_cot_v3.py` 中的 `sparse_loss()` 函数

```python
def sparse_loss(attn_maps: List[torch.Tensor]) -> torch.Tensor:
    loss = 0
    for attn in attn_maps:
        loss += torch.mean(torch.abs(attn))  # L1正则化
    return loss
```

**在训练中使用**：
```python
if 'L_sparse' in outputs['loss_components']:
    L_sparse = outputs['loss_components']['L_sparse']
    total_loss = total_loss + config.lambda_sparse * L_sparse
```

### 3. VisualNoteLayer（Soft Masking）

**核心公式**：
```python
# 计算注意力
attn_map = sigmoid((img_proj @ text_proj^T) / sqrt(d))

# 应用Soft Masking
mask_weight = attn_map + (1 - attn_map) * beta
img_focused = img_feats * mask_weight
```

---

## 🔧 注意事项

### 1. Patch特征获取

**当前实现**：使用全局特征扩展为Patch特征（简化处理）

```python
oct_features = extract_features_with_vit(oct_images, device)  # [B, 768]
oct_features_patch = oct_features.unsqueeze(1).expand(-1, 196, -1)  # [B, 196, 768]
```

**改进建议**：从ViT中间层获取真正的Patch特征

```python
# 需要修改extract_features_with_vit函数，返回Patch特征
oct_features_patch = extract_patch_features_with_vit(oct_images, device)  # [B, 196, 768]
```

### 2. Knowledge Note生成

**当前实现**：使用Prompt作为Note（简化版）

**改进建议**：使用生成模型（GPT-4或Qwen-Med）生成真正的Note文本

### 3. 批处理大小

**建议**：根据显存调整batch_size
- A6000 (48GB): batch_size=16-32
- 较小GPU: batch_size=8-16

---

## ✅ 检查清单

### Day 1检查清单
- [ ] 构建医学知识库（`medical_guidelines.json`）
- [ ] 生成Knowledge Note Embeddings（`knowledge_embeddings.npy`）
- [ ] 验证嵌入文件格式和维度

### Day 2检查清单
- [ ] 测试VisualNoteLayer（输入输出形状正确）
- [ ] 测试不同Beta值的效果
- [ ] 验证Soft Masking工作正常

### Day 3检查清单
- [ ] 测试Bio-COT v3模型前向传播
- [ ] 验证所有输出键存在
- [ ] 验证损失组件计算正确

### Day 4检查清单
- [ ] 修改配置文件路径
- [ ] 运行第一个训练epoch
- [ ] 验证动态Beta策略工作
- [ ] 验证稀疏性损失计算

---

## 📈 预期效果

| 阶段 | 配置 | 预期AUC | 说明 |
|------|------|---------|------|
| **Baseline** | 无Knowledge Notes, 无Visual Notes | 0.84 | v2.0基线 |
| **+ Knowledge Notes** | 使用Knowledge Notes | 0.85-0.86 | +1-2% |
| **+ Visual Notes** | Knowledge Notes + Visual Notes | 0.87-0.89 | +3-5% |

---

## 🐛 故障排除

### 问题1: Knowledge Note Embeddings文件不存在

**解决**：
```bash
# 先运行生成脚本
python knowledge_base/generate_knowledge_notes.py --csv_paths ... --output data/knowledge_embeddings.npy
```

### 问题2: 维度不匹配

**检查**：
- Knowledge Note Embeddings维度应该是768
- 图像特征维度应该是768
- Patch数量应该是196（14x14）

### 问题3: 显存不足

**解决**：
- 减小batch_size
- 使用梯度累积
- 使用混合精度训练

---

**文档版本**：v1.0  
**最后更新**：2025-01-08  
**实施状态**：✅ 代码已完成，待测试和训练

