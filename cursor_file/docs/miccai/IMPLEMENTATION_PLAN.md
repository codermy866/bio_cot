# 🚀 CADIL方法实施计划

## 📋 概述

**目标**：将当前的`CausalBayesianCLIP`（A+B组合）改进为`CADIL`（A'+B'→C'，真正的创新）

**核心创新**：
1. **因果对齐机制**（Causal Alignment）
2. **因果不确定性分解**（Causal Uncertainty Decomposition）
3. **域不变性学习**（Domain-Invariant Learning）

---

## 📁 文件结构

```
src/models/causal/
├── bayesian_clip_framework.py          # 原始方法（A+B）
├── causal_alignment.py                 # ✅ 新增：核心创新模块
│   ├── CausalAlignment                 # 因果对齐机制
│   ├── CausalUncertaintyDecomposition  # 因果不确定性分解
│   └── DomainInvariantLearning        # 域不变性学习
└── cadil_framework.py                  # ✅ 新增：改进版方法（A'+B'→C'）
```

---

## 🔧 实施步骤

### 阶段1：核心模块实现（已完成✅）

**文件**：`src/models/causal/causal_alignment.py`

**包含模块**：
1. ✅ `CausalStructureExtractor` - 因果结构提取器
2. ✅ `CausalAlignment` - 因果对齐机制
3. ✅ `CausalUncertaintyDecomposition` - 因果不确定性分解
4. ✅ `DomainInvariantLearning` - 域不变性学习

---

### 阶段2：集成到模型框架（待实施）

**文件**：`src/models/causal/cadil_framework.py`

**任务**：
1. 创建`CADIL`类（改进版`CausalBayesianCLIP`）
2. 集成三个核心创新模块
3. 修改前向传播流程
4. 修改损失函数

**代码结构**：
```python
class CADIL(nn.Module):
    """
    Causal-Aligned Domain-Invariant Learning
    改进版：A'+B'→C'（真正的创新）
    """
    def __init__(self, ...):
        # 1. 基础模块（保留）
        self.oct_encoder = BayesianCLIPEncoder(...)
        self.colposcopy_encoder = BayesianCLIPEncoder(...)
        self.clinical_encoder = BayesianCLIPEncoder(...)
        
        # 2. 核心创新模块（新增）
        self.causal_alignment = CausalAlignment(...)
        self.causal_uncertainty = CausalUncertaintyDecomposition(...)
        self.domain_invariant = DomainInvariantLearning(...)
        
        # 3. 分类器
        self.classifier = nn.Sequential(...)
    
    def forward(self, oct_feat, colpo_feat, clinical_feat, domain_labels=None):
        # 1. 特征提取（保留）
        oct_mean, oct_var = self.oct_encoder(oct_feat)
        colpo_mean, colpo_var = self.colposcopy_encoder(colpo_feat)
        clinical_mean, clinical_var = self.clinical_encoder(clinical_feat)
        
        # 2. 采样
        oct_feat_sampled = self.oct_encoder.sample(oct_mean, oct_var, self.training)
        colpo_feat_sampled = self.colposcopy_encoder.sample(colpo_mean, colpo_var, self.training)
        clinical_feat_sampled = self.clinical_encoder.sample(clinical_mean, clinical_var, self.training)
        
        # 3. 因果对齐（核心创新1）
        oct_aligned, alignment_loss_oct = self.causal_alignment(oct_feat_sampled, clinical_feat_sampled)
        colpo_aligned, alignment_loss_colpo = self.causal_alignment(colpo_feat_sampled, clinical_feat_sampled)
        
        # 4. 因果不确定性分解（核心创新2）
        clinical_structure = self.causal_alignment.extract_causal_structure(clinical_feat_sampled)
        uncertainty_dict = self.causal_uncertainty(clinical_feat_sampled, clinical_structure)
        
        # 5. 域不变性学习（核心创新3）
        domain_loss = None
        if domain_labels is not None:
            domain_loss_oct = self.domain_invariant(oct_aligned, domain_labels)
            domain_loss_colpo = self.domain_invariant(colpo_aligned, domain_labels)
            domain_loss = domain_loss_oct + domain_loss_colpo
        
        # 6. 多模态融合
        fused = self.fusion(torch.cat([oct_aligned, colpo_aligned, clinical_feat_sampled], dim=-1))
        
        # 7. 分类
        logits = self.classifier(fused)
        
        return {
            'logits': logits,
            'uncertainty': uncertainty_dict,
            'alignment_loss': alignment_loss_oct + alignment_loss_colpo,
            'domain_loss': domain_loss,
            ...
        }
```

---

### 阶段3：修改损失函数（待实施）

**文件**：`src/models/causal/cadil_framework.py`

**任务**：
1. 创建`CADILLoss`类
2. 集成三个新的损失项

**代码结构**：
```python
class CADILLoss(nn.Module):
    """
    CADIL损失函数
    """
    def __init__(
        self, 
        alignment_weight: float = 0.1,
        uncertainty_weight: float = 0.01,
        domain_weight: float = 0.1,
        ...
    ):
        self.alignment_weight = alignment_weight
        self.uncertainty_weight = uncertainty_weight
        self.domain_weight = domain_weight
        ...
    
    def forward(self, predictions, targets, domain_labels=None):
        # 1. 分类损失
        ce_loss = F.cross_entropy(predictions['logits'], targets)
        
        # 2. 因果对齐损失（核心创新1）
        alignment_loss = predictions.get('alignment_loss', 0.0)
        
        # 3. 因果不确定性损失（核心创新2）
        uncertainty_dict = predictions.get('uncertainty', {})
        uncertainty_loss = uncertainty_dict.get('total_uncertainty', 0.0).mean()
        
        # 4. 域不变性损失（核心创新3）
        domain_loss = predictions.get('domain_loss', 0.0)
        
        # 5. 总损失
        total_loss = (
            ce_loss + 
            self.alignment_weight * alignment_loss +
            self.uncertainty_weight * uncertainty_loss +
            self.domain_weight * domain_loss
        )
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'alignment_loss': alignment_loss,
            'uncertainty_loss': uncertainty_loss,
            'domain_loss': domain_loss
        }
```

---

### 阶段4：创建训练脚本（待实施）

**文件**：`experiments/exp2_cadil/train_cadil.py`

**任务**：
1. 复制`train_causal_bayesian.py`
2. 修改为使用`CADIL`模型和`CADILLoss`
3. 添加域标签支持

**关键修改**：
```python
# 导入新模型
from src.models.causal.cadil_framework import CADIL, CADILLoss

# 创建模型
model = CADIL(
    embed_dim=768,
    clinical_dim=256,
    num_classes=num_classes,
    num_centers=num_centers  # 用于域不变性学习
)

# 创建损失函数
criterion = CADILLoss(
    alignment_weight=0.1,
    uncertainty_weight=0.01,
    domain_weight=0.1
)

# 训练循环中需要提供域标签
for batch in train_loader:
    oct_feat, colpo_feat, clinical_feat, labels, domain_labels = batch
    outputs = model(oct_feat, colpo_feat, clinical_feat, domain_labels=domain_labels)
    loss_dict = criterion(outputs, labels, domain_labels=domain_labels)
```

---

### 阶段5：修改数据集（待实施）

**文件**：`src/data/enhanced_multimodal_dataset.py`

**任务**：
1. 添加域标签（center_id）到数据集
2. 在`__getitem__`中返回域标签

**关键修改**：
```python
def __getitem__(self, idx):
    # ... 现有代码 ...
    
    # 获取域标签（center_id）
    domain_label = self.data.iloc[idx]['center_id']  # 假设有center_id列
    
    return {
        'oct_features': oct_features,
        'colposcopy_features': colposcopy_features,
        'clinical_features': clinical_features,
        'label': label,
        'domain_label': domain_label  # 新增
    }
```

---

## 📊 实验设计

### 实验1：消融实验

**目标**：验证三个核心创新模块的贡献

| 方法 | 因果对齐 | 因果不确定性分解 | 域不变性学习 | AUC |
|------|---------|-----------------|-------------|-----|
| Baseline | ✗ | ✗ | ✗ | - |
| + 因果对齐 | ✓ | ✗ | ✗ | - |
| + 因果不确定性 | ✗ | ✓ | ✗ | - |
| + 域不变性 | ✗ | ✗ | ✓ | - |
| CADIL (完整) | ✓ | ✓ | ✓ | - |

### 实验2：对比实验

**目标**：与现有方法对比

| 方法 | 创新点 | AUC (Source) | AUC (Unseen) |
|------|--------|-------------|-------------|
| Simple Fusion | 特征拼接 | - | - |
| Standard CLIP | 跨模态对比 | - | - |
| CausalBayesianCLIP | 因果+贝叶斯 | - | - |
| **CADIL (Ours)** | **因果对齐+域不变性** | **-** | **-** |

### 实验3：跨中心泛化

**目标**：验证Zero-Shot泛化能力

| 方法 | Enshi (Source) | Shiyan (Unseen) | Jingzhou (Unseen) |
|------|---------------|----------------|------------------|
| Baseline | - | - | - |
| CADIL (Ours) | - | - | - |

---

## ⏱️ 时间表

| 阶段 | 任务 | 时间 | 状态 |
|------|------|------|------|
| 阶段1 | 核心模块实现 | 1-2天 | ✅ 已完成 |
| 阶段2 | 集成到模型框架 | 2-3天 | ⏳ 待实施 |
| 阶段3 | 修改损失函数 | 1天 | ⏳ 待实施 |
| 阶段4 | 创建训练脚本 | 1-2天 | ⏳ 待实施 |
| 阶段5 | 修改数据集 | 1天 | ⏳ 待实施 |
| 阶段6 | 实验验证 | 2-3周 | ⏳ 待实施 |

**总计**：约3-4周

---

## ✅ 检查清单

### 代码实现
- [x] 核心模块实现（`causal_alignment.py`）
- [ ] CADIL模型实现（`cadil_framework.py`）
- [ ] CADIL损失函数实现
- [ ] 训练脚本创建
- [ ] 数据集修改（添加域标签）

### 实验验证
- [ ] 消融实验
- [ ] 对比实验
- [ ] 跨中心泛化实验
- [ ] 不确定性分析

### 论文写作
- [ ] Introduction部分（强调创新点）
- [ ] Methods部分（详细描述三个核心创新）
- [ ] Results部分（展示实验效果）

---

## 🎯 下一步行动

1. **立即执行**：创建`cadil_framework.py`，集成三个核心创新模块
2. **修改数据集**：添加域标签支持
3. **创建训练脚本**：`train_cadil.py`
4. **开始实验**：先在小数据集上验证，再扩展到完整数据集

---

## 📝 注意事项

1. **向后兼容**：保留原始`CausalBayesianCLIP`，作为Baseline
2. **参数调优**：三个损失权重（`alignment_weight`, `uncertainty_weight`, `domain_weight`）需要仔细调优
3. **计算开销**：因果对齐和不确定性分解会增加计算开销，需要评估
4. **实验对比**：确保与原始方法在相同条件下对比

