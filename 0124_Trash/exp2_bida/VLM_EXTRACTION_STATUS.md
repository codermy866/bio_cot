# VLM特征提取状态

## 步骤1: 提取训练集VLM特征 ✅ 进行中

**命令:**
```bash
source my_retfound/bin/activate
python experiments/exp2_bida/extract_vlm_features.py \
    --data_root /data2/hmy/5Center_datas/5centers_multi_leave_centers_out \
    --split train \
    --output_file /data2/hmy/5Center_datas/5centers_multi_leave_centers_out/vlm_features_cache/train_vlm_features.npy \
    --device cuda:1 \
    --batch_size 4
```

**输出文件:** `/data2/hmy/5Center_datas/5centers_multi_leave_centers_out/vlm_features_cache/train_vlm_features.npy`

**日志文件:** `experiments/exp2_bida/exp_bio_cot/logs/extract_vlm_train.log`

**预计时间:** 2-3小时（669个样本，batch_size=4）

## 步骤2: 提取验证集VLM特征 ⏳ 等待步骤1完成

**命令:**
```bash
source my_retfound/bin/activate
python experiments/exp2_bida/extract_vlm_features.py \
    --data_root /data2/hmy/5Center_datas/5centers_multi_leave_centers_out \
    --split val \
    --output_file /data2/hmy/5Center_datas/5centers_multi_leave_centers_out/vlm_features_cache/val_vlm_features.npy \
    --device cuda:1 \
    --batch_size 4
```

**输出文件:** `/data2/hmy/5Center_datas/5centers_multi_leave_centers_out/vlm_features_cache/val_vlm_features.npy`

**预计时间:** 30-60分钟（168个样本，batch_size=4）

## 步骤3: Student Prior预训练 ⏳ 等待步骤1和2完成

在训练脚本中自动执行，使用提取的VLM特征进行预训练。

## 步骤4: Bio-COT训练 ⏳ 等待步骤3完成

在Student Prior预训练完成后自动开始。

## 监控命令

```bash
# 检查VLM特征提取进度
tail -f experiments/exp2_bida/exp_bio_cot/logs/extract_vlm_train.log

# 检查进程状态
ps aux | grep extract_vlm_features

# 检查输出文件大小
ls -lh /data2/hmy/5Center_datas/5centers_multi_leave_centers_out/vlm_features_cache/

# 检查GPU使用
nvidia-smi
```

