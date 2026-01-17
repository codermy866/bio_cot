# SOTA方法代码仓库汇总

## 🔍 搜索到的SOTA方法

### 1. 医学多模态Transformer方法

#### ✅ **mmFormer** (2022)
- **论文**: "mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation"
- **GitHub**: https://github.com/YaoZhang93/mmFormer
- **特点**: 处理不完整多模态数据，适用于模态缺失场景
- **适用性**: ⭐⭐⭐⭐⭐ 非常适合（多模态融合）

#### ✅ **HiFuse** (2022)
- **论文**: "HiFuse: Hierarchical Multi-Scale Feature Fusion Network for Medical Image Classification"
- **GitHub**: https://github.com/huoxiangzuo/HiFuse
- **特点**: 层次多尺度特征融合，结合Transformer和CNN
- **适用性**: ⭐⭐⭐⭐⭐ 非常适合（多尺度融合）

#### ✅ **M4oE** (2024)
- **论文**: "M4oE: Medical Multi-modal Mixture-of-Experts Foundation Model"
- **GitHub**: https://github.com/JefferyJiang-YF/M4oE
- **特点**: 专家混合框架，处理模态异质性
- **适用性**: ⭐⭐⭐⭐⭐ 非常适合（多模态专家系统）

### 2. 医学Vision-Language方法

#### ⚠️ **MedCLIP / BioMedCLIP**
- **状态**: 代码仓库需要进一步搜索
- **建议**: 可以基于标准CLIP实现医学版本
- **适用性**: ⭐⭐⭐⭐⭐ 必须包含

#### ⚠️ **ConVIRT**
- **状态**: 代码仓库需要进一步搜索
- **建议**: 可以基于对比学习框架实现
- **适用性**: ⭐⭐⭐⭐⭐ 必须包含

### 3. 其他相关方法

#### ✅ **MedMamba** (2024)
- **GitHub**: https://github.com/YubiaoYue/MedMamba
- **特点**: Vision Mamba用于医学图像分类
- **适用性**: ⭐⭐⭐⭐ 适合（不同backbone）

#### ✅ **Pathomic Fusion**
- **GitHub**: https://github.com/mahmoodlab/PathomicFusion
- **特点**: 融合组织病理学和基因组特征
- **适用性**: ⭐⭐⭐ 部分相关

---

## 📋 推荐的SOTA Baseline实现优先级

### 优先级1: 必须实现 ⭐⭐⭐⭐⭐

1. **mmFormer** - 多模态Transformer（有公开代码）
2. **HiFuse** - 层次多尺度融合（有公开代码）
3. **M4oE** - 专家混合模型（有公开代码）
4. **MedCLIP变体** - 基于标准CLIP实现医学版本

### 优先级2: 强烈建议 ⭐⭐⭐⭐

5. **ConVIRT变体** - 基于对比学习实现
6. **MedMamba** - 不同backbone对比

### 优先级3: 可选 ⭐⭐⭐

7. **Pathomic Fusion** - 如果适用

---

## 🚀 实施策略

### 策略1: 直接使用公开代码（推荐）

对于有公开代码的方法（mmFormer, HiFuse, M4oE）：
1. 克隆GitHub仓库
2. 适配到我们的数据集格式
3. 使用相同的训练/验证集
4. 运行实验

### 策略2: 基于框架实现

对于没有公开代码的方法（MedCLIP, ConVIRT）：
1. 基于标准CLIP实现MedCLIP变体
2. 基于对比学习实现ConVIRT变体
3. 确保实现符合原论文描述

### 策略3: 混合方案

- 直接使用：mmFormer, HiFuse, M4oE
- 自己实现：MedCLIP, ConVIRT
- 保留简单baseline：Simple Fusion, Standard CLIP

---

## 📝 下一步行动

1. **克隆代码仓库**
   ```bash
   # mmFormer
   git clone https://github.com/YaoZhang93/mmFormer.git
   
   # HiFuse
   git clone https://github.com/huoxiangzuo/HiFuse.git
   
   # M4oE
   git clone https://github.com/JefferyJiang-YF/M4oE.git
   ```

2. **评估适配难度**
   - 检查数据格式要求
   - 评估需要多少修改

3. **创建适配框架**
   - 统一的数据接口
   - 统一的评估指标
   - 统一的训练流程

