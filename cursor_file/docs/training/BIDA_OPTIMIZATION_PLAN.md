# BIDA训练优化执行计划

## 🎯 立即执行步骤

### 步骤1：预计算VLM特征（最关键）🔥

**执行命令**：
```bash
# 预计算训练集VLM特征
cd /data2/hmy/VLM_Caus_Rm_Mics
python experiments/exp2_bida/precompute_vlm_features.py \
    --data_root /data2/hmy/5Center_datas/5centers_multi_leave_centers_out \
    --split train \
    --output_dir /data2/hmy/5Center_datas/5centers_multi_leave_centers_out/vlm_features_cache/train \
    --device cuda:1 \
    --batch_size 8

# 预计算验证集VLM特征
python experiments/exp2_bida/precompute_vlm_features.py \
    --data_root /data2/hmy/5Center_datas/5centers_multi_leave_centers_out \
    --split val \
    --output_dir /data2/hmy/5Center_datas/5centers_multi_leave_centers_out/vlm_features_cache/val \
    --device cuda:1 \
    --batch_size 8
```

**预期时间**：2-3小时（取决于数据量）
**预期收益**：训练速度提升10-50倍

### 步骤2：修改训练脚本使用VLM缓存

需要修改`DistributionalAnchor`以支持加载缓存的VLM特征。

### 步骤3：调整损失权重

在训练脚本中修改：
```python
lambda_kl = 0.1      # 从0.005增加到0.1
lambda_orth = 0.5    # 从0.01增加到0.5
lambda_adv = 0.3     # 从0.05增加到0.3
```

### 步骤4：重新训练

使用优化后的参数重新训练。

---

## 📊 预期效果

- **训练速度**：提升10-50倍
- **Val Acc**：从67%提升到75-80%
- **Val AUC**：从0.61提升到0.75-0.85

---

**创建时间**：2025-12-29



