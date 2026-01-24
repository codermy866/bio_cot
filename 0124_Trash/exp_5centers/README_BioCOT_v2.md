# Bio-COT 2.0: 基于LLM语义锚点与因果最优传输的多模态分类框架

## 📋 项目概述

Bio-COT 2.0是对原有Bio-COT框架的重大升级，核心变动在于将简单的"临床数据MLP编码"升级为**"基于医学LLM的语义锚点生成"**，并以Strong Baseline (ViT + Cross-Attention)为基础进行模块化消融实验。

## 🎯 核心升级点

1. **Text Encoder**: MLP (One-hot) → Frozen Med-LLM (Offline Embeddings)
2. **Baseline Fusion**: Concat → Cross-Attention Fusion
3. **Alignment**: MSE/KL → Sinkhorn Optimal Transport (OT)
4. **Modular Design**: 支持消融实验（use_ot, use_dual, use_llm, use_cross_attn）

## 📁 文件结构

```
experiments/exp_5centers/
├── preprocess_llm.py              # Step 1: 离线LLM预处理
├── dataset_v2.py                   # Step 2: 支持LLM嵌入的数据集
├── train_bio_cot_v2.py            # Step 4: 训练脚本
└── data/
    └── clinical_embeddings.pkl    # 预计算的LLM嵌入（8.98 MB, 3009样本）

src/models/bida/
├── bio_cot_v2.py                   # Step 3: Bio-COT 2.0模型架构
│   ├── TextProjector               # LLM嵌入投影器
│   ├── CrossModalFusion            # Cross-Attention融合模块
│   ├── DualHeadImageEncoder        # 双头图像编码器
│   └── BioCOT_v2                   # 主模型类
└── losses.py                       # 损失函数（SinkhornDistance已存在）
```

## 🚀 快速开始

### Step 1: 离线LLM预处理

**重要**：使用实际训练/验证/测试集的CSV文件生成嵌入，确保数据一致性！

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics
source my_retfound/bin/activate

python experiments/exp_5centers/preprocess_llm.py \
    --csv_paths data/5centers_multi_leave_centers_out/train_labels.csv \
                 data/5centers_multi_leave_centers_out/val_labels.csv \
                 data/5centers_multi_leave_centers_out/external_test_labels.csv \
    --output_path experiments/exp_5centers/data/clinical_embeddings_from_csv.pkl \
    --model_name bert-base-uncased \
    --device cuda:1 \
    --batch_size 32
```

**输出**：
- 成功处理985个样本（train: 669, val: 168, test: 148）
- 嵌入维度：768（bert-base-uncased）
- 文件大小：2.94 MB
- 使用`OCT`列作为标识符（与训练时使用的`oct_id`匹配）

### Step 2: 训练Bio-COT 2.0模型

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics
source my_retfound/bin/activate

python experiments/exp_5centers/train_bio_cot_v2.py
```

**配置参数**（在`BioCOT_v2_Args`类中）：
- `use_llm=True`: 使用LLM嵌入（否则使用传统MLP）
- `use_cross_attn=True`: 使用Cross-Attention融合（否则使用Concat）
- `use_ot=True`: 使用Sinkhorn OT损失
- `use_dual=True`: 使用Dual-Head结构

## 🏗️ 架构详解

### 1. TextProjector（临床语义映射器）

**功能**：将离线提取的LLM嵌入（768维）映射到与图像特征对齐的维度（768维）

**结构**：
```python
TextProjector(
    input_dim=768,      # LLM嵌入维度（bert-base-uncased）
    embed_dim=768       # 输出维度（与ViT对齐）
)
```

**网络层**：
- Linear(768 → 2048)
- LayerNorm(2048)
- GELU()
- Dropout(0.2)
- Linear(2048 → 768)

### 2. CrossModalFusion（跨模态融合模块）

**功能**：使用Cross-Attention融合图像和文本特征

**结构**：
- Query: Image features `z_causal` [B, 768]
- Key/Value: Text features `z_sem` [B, 768]
- Multi-Head Attention (8 heads)
- Residual connection + LayerNorm
- Feed-Forward Network (FFN)

**输出**：融合后的特征 `fused_feat` [B, 768]

### 3. DualHeadImageEncoder（双头图像编码器）

**功能**：从图像特征中提取因果特征和噪声特征

**结构**：
- Feature Projection: Linear(768 → 1536 → 768)
- Causal Head: 用于分类的因果特征 `z_causal`
- Noise Head: 用于对抗训练的噪声特征 `z_noise`

### 4. BioCOT_v2（主模型）

**模块化设计**：
- `use_llm`: True=LLM嵌入, False=传统MLP
- `use_cross_attn`: True=Cross-Attention, False=Concat
- `use_ot`: True=Sinkhorn OT损失
- `use_dual`: True=Dual-Head, False=单头

**损失函数**：
- `L_cls`: 分类损失（CrossEntropy/FocalLoss）
- `L_ot`: Sinkhorn OT损失（如果`use_ot=True`）
- `L_consist`: 反事实一致性损失（如果`use_dual=True`）
- `L_adv`: 对抗损失（如果`use_dual=True`）

## 📊 消融实验配置

### 配置1: Strong Baseline（仅Cross-Attention）
```python
use_llm=False      # 使用传统MLP
use_cross_attn=True # Cross-Attention融合
use_ot=False        # 无OT损失
use_dual=False     # 单头结构
```

### 配置2: Baseline + LLM
```python
use_llm=True       # 使用LLM嵌入
use_cross_attn=True
use_ot=False
use_dual=False
```

### 配置3: Baseline + LLM + OT
```python
use_llm=True
use_cross_attn=True
use_ot=True        # 添加Sinkhorn OT
use_dual=False
```

### 配置4: Full Bio-COT 2.0
```python
use_llm=True
use_cross_attn=True
use_ot=True
use_dual=True     # 完整Dual-Head结构
```

## 🔧 技术细节

### LLM模型选择

**当前使用**: `bert-base-uncased` (768维)
- 优点：轻量级，快速，适合调试
- 缺点：非医学专用，语义理解有限

**推荐升级**: `epfl-llm/meditron-7b` (4096维)
- 优点：医学专用，语义理解更强
- 缺点：需要更大显存，需要8bit量化

**使用方法**：
```bash
python preprocess_llm.py \
    --model_name epfl-llm/meditron-7b \
    --use_8bit \
    --device cuda:1
```

### 数据流程

1. **离线预处理**：
   - Excel → Prompt生成 → LLM嵌入提取 → 保存为.pkl

2. **在线训练**：
   - 加载图像（OCT + Colposcopy）
   - 加载预计算的LLM嵌入
   - ViT特征提取 → 图像编码 → 文本投影 → 跨模态融合 → 分类

### 损失函数权重

**默认配置**：
- `L_cls`: 1.0（分类损失）
- `L_ot`: 1.0（OT损失，如果启用）
- `L_consist`: 0.5（一致性损失，如果启用）
- `L_adv`: 1.0（对抗损失，如果启用）

可在`train_bio_cot_v2.py`的`train_epoch`函数中调整。

## 📈 预期效果

### 性能提升（理论）

1. **LLM嵌入 vs MLP**：
   - 更强的语义理解能力
   - 更好的泛化性能
   - 减少过拟合风险

2. **Cross-Attention vs Concat**：
   - 更灵活的跨模态交互
   - 注意力机制自动学习重要特征
   - 更好的特征对齐

3. **Sinkhorn OT vs MSE/KL**：
   - 更灵活的分布对齐
   - 对称性（不受参考分布影响）
   - 几何直观性

### 消融实验表格（示例）

| 配置 | LLM | Cross-Attn | OT | Dual-Head | Val AUC | Val Acc |
|------|-----|------------|----|-----------|---------|---------|
| Baseline | ✗ | ✗ | ✗ | ✗ | 0.65 | 0.72 |
| + Cross-Attn | ✗ | ✓ | ✗ | ✗ | 0.68 | 0.74 |
| + LLM | ✓ | ✓ | ✗ | ✗ | 0.71 | 0.76 |
| + OT | ✓ | ✓ | ✓ | ✗ | 0.73 | 0.77 |
| Full v2.0 | ✓ | ✓ | ✓ | ✓ | 0.75 | 0.79 |

## 🐛 故障排除

### 问题1: LLM嵌入文件不存在
**解决**：先运行`preprocess_llm.py`生成嵌入文件

### 问题2: 维度不匹配
**解决**：检查`llm_embed_dim`参数是否与实际LLM嵌入维度匹配

### 问题3: 显存不足
**解决**：
- 使用8bit量化：`--use_8bit`
- 减小batch_size
- 使用更小的LLM模型

### 问题4: 训练速度慢
**解决**：
- 增加`num_workers`
- 使用`persistent_workers=True`
- 增加`prefetch_factor`

## 📝 注意事项

1. **不终止当前训练**：Bio-COT 2.0训练进程独立运行，不会影响现有的Bio-COT v1训练
2. **模块化设计**：所有组件都可以独立开关，方便消融实验
3. **向后兼容**：`use_llm=False`时，模型行为与v1类似（但使用Cross-Attention）

## 🔗 相关文档

- `BioCOT_Methodology_Analysis.md`: Bio-COT v1的详细技术分析
- `train_bio_cot_5centers_multimodal.py`: Bio-COT v1的训练脚本
- `src/models/bida/bio_cot_model.py`: Bio-COT v1的模型定义

## 📧 联系方式

如有问题，请检查日志文件：`experiments/exp_5centers/logs/train_bio_cot_v2_*.log`

