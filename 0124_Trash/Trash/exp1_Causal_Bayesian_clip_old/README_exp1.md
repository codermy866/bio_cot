# 实验1：增强因果约束贝叶斯CLIP（Enhanced Causal Bayesian CLIP）

## 📋 实验概述

### 研究目标
本研究提出**增强因果约束贝叶斯CLIP框架**，用于多模态医学影像（OCT + Colposcopy + Clinical）的宫颈病变智能诊断。核心创新在于：

1. **可学习因果图发现**：数据驱动学习模态间因果关系，结合医学领域先验知识
2. **不确定性分解**：区分认知不确定性（模型）和偶然不确定性（数据）
3. **贝叶斯CLIP框架**：在CLIP基础上引入不确定性量化，提升模型可靠性

### 核心问题
- **传统CLIP的局限性**：学习关联关系而非因果关系，存在虚假关联
- **缺乏不确定性量化**：无法评估预测置信度，难以识别困难样本
- **缺乏可解释性**：无法理解模态间关系，难以用于临床决策

### 解决方案
- **因果约束**：使用可学习因果图约束注意力机制，消除虚假关联
- **不确定性量化**：贝叶斯编码器输出均值和方差，实现不确定性估计
- **可解释性**：因果图可视化，提供模型决策依据

---

## 🚀 快速开始

### 环境配置

```bash
# 激活虚拟环境
source /data2/hmy/VLM_Caus_Rm/my_retfound/bin/activate

# 安装依赖
pip install -r requirements_enhanced.txt
```

### 训练模型

```bash
cd code/

# 基础训练（推荐配置）
python train_enhanced_causal_clip.py \
    --data_path /data2/hmy/VLM_Caus_Rm/data/5centers_multi \
    --output_dir ../results/enhanced_causal_clip_results/run_exp1 \
    --batch_size 24 \
    --num_epochs 100 \
    --learning_rate 3e-4 \
    --no_amp \
    --causal_loss_weight 0.001 \
    --kl_weight 0.001 \
    --label_smoothing 0.01

# 或使用脚本
bash run_optimized_training.sh
```

### 导出因果图

```bash
# 导出训练好的模型的因果邻接矩阵
python export_causal_adj.py \
    --model_path ../results/enhanced_causal_clip_results/run_exp1/best_model.pth \
    --output_dir ../visualization/causal_analysis_detailed/
```

### 生成可视化

```bash
# 生成因果图可视化
python generate_causal_visualizations.py

# 生成详细医学因果图
python generate_detailed_medical_causal_graph.py

# 生成训练过程可视化
python generate_training_causal_visualizations.py
```

---

## 🏗️ 方法架构

### 整体框架

```
输入数据
├── OCT图像 [B, 120, 3, 224, 224]
├── Colposcopy图像 [B, 3, 3, 224, 224]
└── Clinical特征 [B, 7]
        ↓
特征提取（Swin-T，部分微调）
├── OCT编码器 → [B, 768]
├── Colposcopy编码器 → [B, 768]
└── Clinical投影 → [B, 768]
        ↓
贝叶斯编码器（变分推断）
├── OCT: μ_oct, σ²_oct
├── Colposcopy: μ_col, σ²_col
└── Clinical: μ_clin, σ²_clin
        ↓
可学习因果图发现
├── 数据驱动学习（MLP）
├── 领域知识融合（硬约束）
├── DAG约束（NOTEARS + 上三角）
└── 稀疏正则化
        ↓
不确定性分解
├── 认知不确定性（Epistemic）
└── 偶然不确定性（Aleatoric）
        ↓
因果约束的多头注意力融合
        ↓
分类器（2-class）
```

### 核心模块

#### 1. 可学习因果图发现（LearnableCausalGraph）

**功能**：动态学习三个模态（OCT、Colposcopy、Clinical）之间的因果关系

**关键特性**：
- **因果发现网络**：MLP(2304 → 512 → 256 → 9)
- **可学习权重矩阵**：Parameter([3, 3])
- **DAG约束**：上三角矩阵 + NOTEARS惩罚
- **稀疏正则化**：L1/L0近似
- **干预反馈**：伪干预训练（50%概率）

**数学公式**：
```
causal_adj = Sigmoid(MLP(concat([oct_feat, colpo_feat, clinical_feat])))
dag_penalty = ReLU(trace(exp(A ∘ A)) - 3)²
sparsity_penalty = mean(|causal_adj|)
```

#### 2. 贝叶斯编码器（BayesianCLIPEncoder）

**功能**：为每个模态输出均值和方差，实现不确定性量化

**关键特性**：
- **变分推断**：q(z|x) = N(μ(x), σ²(x))
- **重参数化技巧**：z = μ + ε·σ, ε ~ N(0,1)
- **KL正则化**：防止方差过大

**数学公式**：
```
μ = Linear(feat)
log_σ² = Softplus(Linear(feat))
z = μ + ε·σ, ε ~ N(0,1)
KL = 0.5 * Σ(σ² + μ² - 1 - log(σ²))
```

#### 3. 不确定性分解（UncertaintyDecomposition）

**功能**：区分认知不确定性和偶然不确定性

**关键特性**：
- **认知不确定性**：基于融合特征的MLP估计（模型参数不确定性）
- **偶然不确定性**：基于贝叶斯编码器方差（数据固有噪声）
- **总不确定性**：epistemic + aleatoric

**数学公式**：
```
epistemic = MLP(fusion_feat) → Softplus → [B, 1]
aleatoric = MLP(mean(variances)) → Softplus → [B, 1]
total_uncertainty = epistemic + aleatoric
```

---

## 📁 文件结构

```
exp1_Causal_Bayesian_clip/
├── README_exp1.md                    # 本文件
├── requirements_enhanced.txt         # 依赖包列表
│
├── code/                             # 代码实现
│   ├── train_enhanced_causal_clip.py          # 主训练脚本
│   ├── enhanced_causal_clip.py                # 模型定义
│   ├── visualize_causal_graph.py             # 因果图可视化
│   ├── generate_causal_visualizations.py       # 生成因果可视化
│   ├── generate_detailed_medical_causal_graph.py  # 详细医学因果图
│   ├── generate_training_causal_visualizations.py # 训练过程可视化
│   ├── export_causal_adj.py                   # 导出因果邻接矩阵
│   ├── run_optimized_training.sh              # 训练脚本
│   ├── auto_start_training.py                 # 自动启动训练
│   ├── start_optimized_clip_training.sh       # 优化训练启动
│   └── start_stage4_training.sh              # 阶段4训练启动
│
├── docs/                             # 文档
│   ├── TECHNICAL_WORK_SUMMARY.md              # 技术工作总结
│   ├── EXPERIMENTAL_METHODS.md                # 实验方法详细说明
│   ├── CAUSAL_CONSTRAINED_CLIP_DEEP_ANALYSIS.md  # 因果CLIP深度分析
│   ├── DEEP_TECHNICAL_ANALYSIS.md              # 技术模块深度分析
│   ├── TECHNICAL_MODULES_ANALYSIS.md           # 技术模块分析
│   ├── CAUSAL_VISUALIZATION_OUTPUT_LOCATIONS.md # 可视化输出位置
│   ├── REGULARIZATION_OPTIMIZATION.md         # 正则化优化
│   ├── README_ORGANIZATION.md                  # 文件组织说明
│   └── PROJECT_STRUCTURE.md                   # 项目结构
│
├── results/                          # 实验结果
│   └── enhanced_causal_clip_results/
│       ├── run_bs24_e100_noamp_fullOCT_noclc_causal001_ls001_kl0001/  # 当前最优配置
│       │   ├── best_model.pth       # 最佳模型
│       │   ├── metrics.json         # 评估指标
│       │   ├── train.log            # 训练日志
│       │   └── training_history.json # 训练历史
│       └── ...                       # 其他实验配置
│
└── visualization/                    # 可视化结果
    ├── causal_analysis_detailed/     # 详细因果分析
    │   ├── causal_adj_epoch_*.npy   # 因果邻接矩阵
    │   ├── causal_graph_epoch_*.png # 因果图可视化
    │   └── causal_paths_analysis_epoch_*.png  # 因果路径分析
    ├── adaptive_causal_intervention_results/  # 自适应因果干预结果
    │   ├── model_architecture.png   # 模型架构图
    │   ├── innovation*.png          # 创新点可视化
    │   └── ...                      # 其他可视化
    └── visualization/               # 可视化工具
        └── ...                      # 可视化脚本
```

---

## 📊 实验结果

### 当前最优配置

**训练配置**：`run_bs24_e100_noamp_fullOCT_noclc_causal001_ls001_kl0001`

**超参数**：
- Batch size: 24
- Learning rate: 3e-4
- Epochs: 100
- Causal loss weight: 0.001
- KL loss weight: 0.001
- Label smoothing: 0.01
- Contrastive weight: 0 (关闭)

**训练状态**：
- Epoch 2: Loss ≈ 0.25-0.30, Accuracy ≈ 60%
- 训练稳定，无NaN/Inf损失

### 模型复杂度

- **总参数量**：约78.53M
  - 特征提取器：55.04M（可训练52.64M）
  - 主模型：23.49M
- **显存占用**：约4-5GB（batch=24，120帧OCT）
- **训练时间**：约2-3分钟/epoch，总时间约3-5小时（100 epochs）

### 性能目标

- **验证集准确率**：> 70%
- **AUC-ROC**：> 0.75
- **F1-Score**：> 0.70

---

## 🔧 使用方法

### 1. 训练模型

#### 基础训练
```bash
python code/train_enhanced_causal_clip.py \
    --data_path /path/to/5centers_multi \
    --output_dir results/my_experiment \
    --batch_size 24 \
    --num_epochs 100 \
    --learning_rate 3e-4 \
    --no_amp
```

#### 自定义超参数
```bash
python code/train_enhanced_causal_clip.py \
    --data_path /path/to/5centers_multi \
    --output_dir results/custom_experiment \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --causal_loss_weight 0.002 \
    --kl_weight 0.003 \
    --label_smoothing 0.05 \
    --contrastive_weight 0.05
```

### 2. 评估模型

```python
# 加载模型
import torch
from code.enhanced_causal_clip import EnhancedCausalBayesianCLIP

model = EnhancedCausalBayesianCLIP(...)
checkpoint = torch.load('results/.../best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 评估
model.eval()
with torch.no_grad():
    outputs = model(oct_images, colpo_images, clinical_features)
    predictions = outputs['logits']
    uncertainties = outputs['uncertainties']
```

### 3. 导出因果图

```bash
python code/export_causal_adj.py \
    --model_path results/.../best_model.pth \
    --data_path /path/to/5centers_multi \
    --output_dir visualization/causal_analysis_detailed/ \
    --num_samples 100
```

### 4. 生成可视化

```bash
# 生成因果图可视化
python code/generate_causal_visualizations.py \
    --model_path results/.../best_model.pth \
    --output_dir visualization/causal_analysis_detailed/

# 生成详细医学因果图
python code/generate_detailed_medical_causal_graph.py \
    --output_dir visualization/causal_analysis_detailed/

# 生成训练过程可视化
python code/generate_training_causal_visualizations.py \
    --log_file results/.../train.log \
    --output_dir visualization/training_plots/
```

---

## 📚 详细文档

### 核心文档

1. **`docs/TECHNICAL_WORK_SUMMARY.md`**
   - 完整的技术工作梳理
   - 核心创新点分析
   - 训练优化历程
   - 下一步工作计划

2. **`docs/EXPERIMENTAL_METHODS.md`**
   - 详细的实验方法说明
   - 数据集描述
   - 模型架构详解
   - 损失函数设计
   - 训练策略

3. **`docs/CAUSAL_CONSTRAINED_CLIP_DEEP_ANALYSIS.md`**
   - 因果约束CLIP深度分析
   - 数学公式体系
   - 实现流程详解
   - 因果图构建机制

4. **`docs/DEEP_TECHNICAL_ANALYSIS.md`**
   - 技术模块深度分析
   - 算法原理
   - 实现细节
   - 设计决策分析

### 可视化文档

- **`docs/CAUSAL_VISUALIZATION_OUTPUT_LOCATIONS.md`**：可视化输出位置说明
- **`visualization/adaptive_causal_intervention_results/DETAILED_FORMULAS_AND_FLOWCHARTS.md`**：详细公式和流程图

---

## 🎯 核心创新点

### 创新点1：可学习因果图发现

**问题**：传统方法使用固定因果图，无法适应数据分布

**解决方案**：
- 数据驱动学习：基于特征学习因果邻接矩阵
- 领域知识融合：硬约束（Clinical → OCT/Colposcopy）
- DAG约束：NOTEARS风格惩罚 + 上三角矩阵
- 稀疏正则化：L1/L0近似，减少虚假因果关系
- 干预反馈：伪干预训练，增强因果敏感性

**技术细节**：见 `code/enhanced_causal_clip.py` 中的 `LearnableCausalGraph` 类

### 创新点2：不确定性分解

**问题**：总不确定性无法区分来源（模型 vs 数据）

**解决方案**：
- 认知不确定性：基于融合特征的MLP估计（模型参数不确定性）
- 偶然不确定性：基于贝叶斯编码器方差（数据固有噪声）
- 总不确定性：epistemic + aleatoric

**技术细节**：见 `code/enhanced_causal_clip.py` 中的 `UncertaintyDecomposition` 类

### 创新点3：贝叶斯CLIP框架

**问题**：标准CLIP无法量化预测不确定性

**解决方案**：
- 变分推断：每个模态输出均值和方差
- 重参数化技巧：训练时采样，推理时使用均值
- KL正则化：防止方差过大，保持数值稳定

**技术细节**：见 `code/enhanced_causal_clip.py` 中的 `BayesianCLIPEncoder` 类

---

## 🔬 实验设置

### 数据集

- **名称**：5centers_multi
- **训练集**：785个患者
- **验证集**：200个患者
- **任务**：二分类（正常 vs. 异常）

### 多模态数据

| 模态 | 数据量 | 维度 | 处理方式 |
|------|--------|------|----------|
| OCT | 120帧/患者 | [B, 120, 3, 224, 224] | Swin-T编码 → 平均池化 → [B, 768] |
| Colposcopy | 3帧/患者 | [B, 3, 3, 224, 224] | Swin-T编码 → 平均池化 → [B, 768] |
| Clinical | 7维/患者 | [B, 7] | Linear投影 → [B, 768] |

### 训练配置

```yaml
优化器: AdamW
学习率: 3e-4
调度器: Cosine Annealing (T_max=100)
批次大小: 24
总epoch: 100
混合精度: 关闭 (FP32)

损失权重:
  - Focal Loss: 1.0 (γ=2.0, α=auto)
  - Label Smoothing: 0.01
  - KL Loss: 0.001
  - Contrastive Loss: 0 (关闭)
  - Causal Loss: 0.001
    - DAG Penalty: 0.02
    - Sparsity Penalty: 0.0002
    - Intervention Penalty: 0.01

特征提取器:
  - Backbone: Swin-T (timm)
  - 微调策略: 解冻最后2层
  - 可训练参数: 52.64M / 55.04M (95.6%)
```

---

## 📈 训练优化历程

### 关键优化点

1. **确保完整数据加载**
   - 设置 `oct_num_frames=120`
   - 关闭 `cache_oct_features=False`
   - 添加调试日志验证

2. **降低损失值**
   - 降低 `causal_loss_weight`: 0.002 → 0.001
   - 降低 `kl_weight`: 0.003 → 0.001
   - 降低 `label_smoothing`: 0.05 → 0.01
   - 关闭 `contrastive_weight`: 0.05 → 0

3. **提高GPU利用率**
   - 增加 `batch_size`: 6 → 24
   - 关闭 `AMP` (使用FP32)
   - 确认所有120帧参与训练

4. **处理NaN/Inf损失**
   - 添加损失检查
   - 添加梯度裁剪（max_norm=1.0）
   - 降低正则化权重

---

## 🎓 评估指标

### 分类性能指标

- **准确率（Accuracy）**
- **AUC-ROC**
- **F1-Score**
- **精确率（Precision）**
- **召回率（Recall / Sensitivity）**
- **特异性（Specificity）**

### 最优阈值选择

使用**Youden指数**：
```python
Youden Index = Sensitivity + Specificity - 1
optimal_threshold = argmax(Youden Index)
```

### 临床指标

- **阳性预测值（PPV）**
- **阴性预测值（NPV）**
- **阳性似然比（LR+）**
- **阴性似然比（LR-）**
- **MCC（Matthews Correlation Coefficient）**

### 不确定性评估

- **总不确定性**：epistemic + aleatoric
- **不确定性校准**：评估预测置信度与实际准确率的一致性

---

## 🚧 下一步工作

### 短期目标（1-2周）

1. **完成训练**：运行100 epochs，达到收敛
2. **性能评估**：验证集AUC > 0.75
3. **超参数调优**：进一步优化损失权重

### 中期目标（2-4周）

1. **消融实验**：
   - 移除可学习因果图
   - 移除不确定性分解
   - 移除干预反馈机制
2. **多中心验证**：5个中心独立评估
3. **可解释性分析**：因果图可视化，注意力热力图

### 长期目标（1-2月）

1. **论文撰写**：方法、实验、讨论
2. **临床验证**：与医生合作，实际应用测试
3. **方法扩展**：扩展到5分类任务

---

## 🔗 相关资源

### 代码仓库
- 主训练脚本：`code/train_enhanced_causal_clip.py`
- 模型定义：`code/enhanced_causal_clip.py`
- 可视化工具：`code/visualize_causal_graph.py`

### 文档资源
- 技术总结：`docs/TECHNICAL_WORK_SUMMARY.md`
- 实验方法：`docs/EXPERIMENTAL_METHODS.md`
- 深度分析：`docs/CAUSAL_CONSTRAINED_CLIP_DEEP_ANALYSIS.md`

### 实验结果
- 训练结果：`results/enhanced_causal_clip_results/`
- 可视化结果：`visualization/causal_analysis_detailed/`

---

## 📝 引用

如果使用本实验的代码或方法，请引用：

```bibtex
@article{causal_bayesian_clip_2024,
  title={Enhanced Causal Bayesian CLIP for Multimodal Medical Image Analysis},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

---

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 项目Issue：GitHub Issues
- 邮箱：your.email@example.com

---

## 📄 许可证

本项目采用 [LICENSE](../LICENSE) 许可证。

---

**最后更新**：2024年1月

**版本**：v1.0

