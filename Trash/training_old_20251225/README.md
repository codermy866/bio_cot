# 训练脚本目录

本目录包含所有训练相关的脚本：

- `optimized_vmamba_training.py`: VMamba模型优化训练脚本
- `optimized_2class_training.py`: CNN模型优化训练脚本
- `advanced_training_pipeline.py`: 高级训练流程脚本

## 使用方法

```bash
# VMamba训练
python training/optimized_vmamba_training.py --data_path ../5centers_multi --epochs 20

# CNN训练
python training/optimized_2class_training.py --data_path ../5centers_multi --epochs 20
```
