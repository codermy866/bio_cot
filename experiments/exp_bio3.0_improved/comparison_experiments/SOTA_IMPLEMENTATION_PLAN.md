# SOTA Baseline实现计划

## ✅ 已完成

1. ✅ **创建了SOTA Baseline框架**
   - 统一训练基类：`baselines/sota_baselines/common/trainer_base.py`
   - MedCLIP实现：`baselines/sota_baselines/medclip/train_medclip.py`
   - 更新了实验脚本：`run_all_baselines.py`

2. ✅ **搜索了SOTA方法代码仓库**
   - mmFormer: https://github.com/YaoZhang93/mmFormer
   - HiFuse: https://github.com/huoxiangzuo/HiFuse
   - M4oE: https://github.com/JefferyJiang-YF/M4oE
   - MedMamba: https://github.com/YubiaoYue/MedMamba

## 📋 待实现

### 优先级1: 必须实现 ⭐⭐⭐⭐⭐

#### 1. MedCLIP ✅ (已完成框架)
- **状态**: 框架已创建，可以运行
- **下一步**: 测试运行，优化实现

#### 2. mmFormer ⏳ (待实现)
- **GitHub**: https://github.com/YaoZhang93/mmFormer
- **步骤**:
  1. 克隆代码仓库
  2. 分析代码结构
  3. 创建适配器（`model_adapter.py`）
  4. 创建训练脚本（`train_mmformer.py`）
  5. 测试运行

#### 3. HiFuse ⏳ (待实现)
- **GitHub**: https://github.com/huoxiangzuo/HiFuse
- **步骤**: 同上

#### 4. M4oE ⏳ (待实现)
- **GitHub**: https://github.com/JefferyJiang-YF/M4oE
- **步骤**: 同上

### 优先级2: 强烈建议 ⭐⭐⭐⭐

#### 5. ConVIRT ⏳ (待实现)
- **状态**: 需要基于对比学习框架实现
- **步骤**:
  1. 研究ConVIRT论文
  2. 基于对比学习实现
  3. 创建训练脚本

## 🚀 实施步骤

### 步骤1: 克隆代码仓库

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0_improved/comparison_experiments/baselines/sota_baselines

# mmFormer
git clone https://github.com/YaoZhang93/mmFormer.git mmformer_repo

# HiFuse
git clone https://github.com/huoxiangzuo/HiFuse.git hifuse_repo

# M4oE
git clone https://github.com/JefferyJiang-YF/M4oE.git m4oe_repo
```

### 步骤2: 分析代码结构

对于每个方法：
1. 阅读README和文档
2. 理解数据格式要求
3. 理解模型架构
4. 评估适配难度

### 步骤3: 创建适配器

为每个方法创建：
1. `model_adapter.py` - 模型适配器，统一接口
2. `train_*.py` - 训练脚本，继承`SOTABaselineTrainer`

### 步骤4: 测试运行

1. 使用小数据集测试
2. 确保可以正常运行
3. 检查结果格式是否正确

### 步骤5: 正式运行

1. 使用完整数据集
2. 运行多次（5次）用于统计检验
3. 保存结果

## 📊 当前状态总结

| 方法 | 代码仓库 | 框架状态 | 实现状态 | 优先级 |
|------|---------|---------|---------|--------|
| MedCLIP | 自实现 | ✅ 完成 | ✅ 可运行 | ⭐⭐⭐⭐⭐ |
| mmFormer | ✅ 有 | ⏳ 待创建 | ⏳ 待实现 | ⭐⭐⭐⭐⭐ |
| HiFuse | ✅ 有 | ⏳ 待创建 | ⏳ 待实现 | ⭐⭐⭐⭐⭐ |
| M4oE | ✅ 有 | ⏳ 待创建 | ⏳ 待实现 | ⭐⭐⭐⭐⭐ |
| ConVIRT | ⚠️ 需实现 | ⏳ 待创建 | ⏳ 待实现 | ⭐⭐⭐⭐ |

## 🎯 下一步行动

1. **立即**: 测试MedCLIP实现
2. **本周**: 实现mmFormer适配器
3. **下周**: 实现HiFuse和M4oE适配器
4. **后续**: 实现ConVIRT

## 📝 注意事项

1. **数据格式统一**: 所有方法必须使用相同的数据格式
2. **评估指标统一**: 所有方法使用相同的评估指标
3. **训练设置统一**: 相同的epochs, batch size, learning rate等
4. **随机种子固定**: 确保可复现性

