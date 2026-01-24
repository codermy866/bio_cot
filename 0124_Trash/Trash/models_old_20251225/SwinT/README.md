# Swin‑T 方法

- 结果输出目录：`models/SwinT/_results/`
  - 多模态：`models/SwinT/_results/multimodal`
  - OCT 单模态：`models/SwinT/_results/oct`

## 快速启动（推荐）

使用项目虚拟环境启动（自动使用 `./my_retfound/bin/python`）：

**多模态训练（OCT+Col+Clinical）：**
```bash
bash models/SwinT/scripts/run_swin_multimodal.sh [数据路径] [epochs] [batch_size] [learning_rate]
# 示例（使用默认参数）：
bash models/SwinT/scripts/run_swin_multimodal.sh
# 或指定参数：
bash models/SwinT/scripts/run_swin_multimodal.sh 5centers_multi 20 8 3e-5
```

**OCT 单模态训练：**
```bash
bash models/SwinT/scripts/run_swin_oct.sh [数据路径] [epochs] [batch_size] [learning_rate]
# 示例：
bash models/SwinT/scripts/run_swin_oct.sh
```

## 直接使用 Python 脚本

**多模态（OCT+Col+Clinical）：**
```bash
./my_retfound/bin/python models/SwinT/scripts/start_swin_multimodal.py \
  --data_path 5centers_multi \
  --epochs 20 --batch_size 8 --learning_rate 3e-5
```

**OCT 单模态：**
```bash
./my_retfound/bin/python models/SwinT/scripts/start_swin_oct.py \
  --data_path 5centers_multi \
  --epochs 20 --batch_size 8 --learning_rate 3e-5
```

## 查看训练日志

```bash
# 多模态训练日志
tail -f models/SwinT/_results/multimodal/train.log

# OCT 单模态训练日志
tail -f models/SwinT/_results/oct/train.log
```

## 依赖

- `timm` 已在 `requirements_enhanced.txt` 中列出
- 虚拟环境：`./my_retfound/bin/python`


