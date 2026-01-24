# ✅ 数据路径更新完成

## 📊 更新统计

### 已更新的文件

1. **训练脚本**
   - ✅ `train_vlm_causal_clip.py` - 默认路径已更新
   - ✅ `train_enhanced_causal_clip.py` - 默认路径和函数参数已更新

2. **数据工具脚本**
   - ✅ `split_dataset_by_centers.py` - 默认路径已更新
   - ✅ `check_data_split.py` - 默认路径已更新
   - ✅ `visualize_dataset_split.py` - 默认路径已更新

3. **文档文件**
   - ✅ `docs/DATASET_SIZE_ANALYSIS.md`
   - ✅ `docs/DATASET_SPLIT_COMPARISON.md`
   - ✅ `exp1_Causal_Bayesian_clip/docs/VLM_QUICK_START.md`

4. **软链接**
   - ✅ `/data2/hmy/VLM_Caus_Rm/data/5centers_multi` → `/data2/hmy/5Center_datas/5centers_multi`
   - ✅ `/data2/hmy/VLM_Caus_Rm/data/5centers_multi_internal_external_final` → `/data2/hmy/5Center_datas/5centers_multi_internal_external_final`

---

## 🎯 新数据路径

### 推荐使用（最终划分版本）

```python
DATA_PATH = "/data2/hmy/5Center_datas/5centers_multi_internal_external_final"
```

### 原始数据集

```python
DATA_PATH = "/data2/hmy/5Center_datas/5centers_multi"
```

---

## 🚀 使用示例

### 训练VLM增强模型

```bash
cd /data2/hmy/VLM_Caus_Rm/exp1_Causal_Bayesian_clip/code

# 使用默认路径（已更新）
python train_vlm_causal_clip.py \
    --batch_size 10 \
    --num_epochs 100 \
    --use_amp

# 或指定路径
python train_vlm_causal_clip.py \
    --data_path /data2/hmy/5Center_datas/5centers_multi_internal_external_final \
    --batch_size 10
```

### 训练基础模型

```bash
python train_enhanced_causal_clip.py \
    --batch_size 24 \
    --num_epochs 100
```

---

## ⚠️ 注意事项

1. **数据迁移状态**: 请检查迁移是否完成
   ```bash
   tail -f /data2/hmy/5Center_datas/migration.log
   ls -lh /data2/hmy/5Center_datas/
   ```

2. **验证数据**: 迁移完成后验证数据完整性
   ```python
   import pandas as pd
   df = pd.read_csv('/data2/hmy/5Center_datas/5centers_multi_internal_external_final/train_labels.csv')
   print(f"训练集: {len(df)} 样本")
   ```

3. **相对路径**: 如果使用相对路径，确保工作目录正确

---

## 📝 后续操作

- [x] 更新Python代码路径
- [x] 更新文档路径
- [x] 更新软链接
- [ ] 验证数据迁移完成
- [ ] 测试数据加载
- [ ] 检查shell脚本（如需要）

---

**更新完成时间**: 2025-12-17

