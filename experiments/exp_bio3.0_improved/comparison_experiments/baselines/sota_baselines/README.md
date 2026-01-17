# SOTA Baseline实现框架

## 📋 目录结构

```
sota_baselines/
├── README.md                    # 本文件
├── mmformer/                    # mmFormer实现
│   ├── train_mmformer.py
│   └── model_adapter.py
├── hifuse/                      # HiFuse实现
│   ├── train_hifuse.py
│   └── model_adapter.py
├── m4oe/                        # M4oE实现
│   ├── train_m4oe.py
│   └── model_adapter.py
├── medclip/                     # MedCLIP实现
│   ├── train_medclip.py
│   └── medclip_model.py
├── convirt/                     # ConVIRT实现
│   ├── train_convirt.py
│   └── convirt_model.py
└── common/                      # 通用工具
    ├── data_adapter.py          # 数据格式适配
    ├── metrics.py               # 统一评估指标
    └── trainer_base.py          # 基础训练框架
```

## 🎯 实现原则

1. **统一接口**: 所有SOTA方法使用相同的数据接口和评估协议
2. **公平对比**: 使用相同的数据集、相同的训练/验证集划分
3. **可复现**: 固定随机种子，多次运行取平均
4. **可扩展**: 易于添加新的SOTA方法

## 📝 实现状态

| 方法 | 状态 | 优先级 | 预计完成时间 |
|------|------|--------|------------|
| mmFormer | ⏳ 待实现 | ⭐⭐⭐⭐⭐ | 1-2天 |
| HiFuse | ⏳ 待实现 | ⭐⭐⭐⭐⭐ | 1-2天 |
| M4oE | ⏳ 待实现 | ⭐⭐⭐⭐⭐ | 1-2天 |
| MedCLIP | ⏳ 待实现 | ⭐⭐⭐⭐⭐ | 2-3天 |
| ConVIRT | ⏳ 待实现 | ⭐⭐⭐⭐ | 2-3天 |

## 🚀 快速开始

### 1. 安装依赖

```bash
# 根据各个方法的要求安装依赖
# 例如mmFormer可能需要特定的环境
```

### 2. 准备数据

```bash
# 使用统一的数据适配器
python common/data_adapter.py --method mmformer --input_dir <data_dir>
```

### 3. 运行实验

```bash
# 运行单个SOTA方法
python mmformer/train_mmformer.py --experiment_name baseline_mmformer

# 或使用统一的运行脚本
python ../run_all_sota_baselines.py
```

