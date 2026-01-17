# 数据路径更新总结

## ✅ 已完成的更新

### 1. 数据集迁移
- **源路径**: `/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/5centers_multi*`
- **新路径**: `/data2/hmy/5Center_datas/5centers_multi*`
- **状态**: 迁移进行中（后台运行）

### 2. 代码路径更新

#### 已更新的文件（11处替换）

1. **VLM_Caus_Rm/docs/DATASET_SIZE_ANALYSIS.md** (2处)
   - 更新数据路径示例

2. **VLM_Caus_Rm/docs/DATASET_SPLIT_COMPARISON.md** (1处)
   - 更新数据路径示例

3. **VLM_Caus_Rm/exp1_Causal_Bayesian_clip/docs/VLM_QUICK_START.md** (1处)
   - 更新训练命令中的数据路径

4. **VLM_Caus_Rm/exp1_Causal_Bayesian_clip/code/check_data_split.py** (1处)
   - 更新默认数据路径

5. **exp1_Causal_Bayesian_clip/README_exp1.md** (1处)
   - 更新数据路径说明

6. **exp1_Causal_Bayesian_clip/code/split_dataset_by_centers.py** (2处)
   - 更新默认数据路径参数

7. **exp1_Causal_Bayesian_clip/code/visualize_dataset_split.py** (1处)
   - 更新默认数据路径

8. **exp1_Causal_Bayesian_clip/code/train_vlm_causal_clip.py** (1处)
   - 更新默认数据路径为: `/data2/hmy/5Center_datas/5centers_multi_internal_external_final`

9. **exp1_Causal_Bayesian_clip/code/train_enhanced_causal_clip.py** (2处)
   - 更新默认数据路径
   - 更新`prepare_data_loaders`函数默认参数

### 3. 软链接更新

**新项目中的软链接** (`/data2/hmy/VLM_Caus_Rm/data/`):
- ✅ `5centers_multi` → `/data2/hmy/5Center_datas/5centers_multi`
- ✅ `5centers_multi_internal_external_final` → `/data2/hmy/5Center_datas/5centers_multi_internal_external_final`

---

## 📋 路径映射表

| 旧路径 | 新路径 |
|--------|--------|
| `/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/5centers_multi` | `/data2/hmy/5Center_datas/5centers_multi` |
| `/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/5centers_multi_internal_external` | `/data2/hmy/5Center_datas/5centers_multi_internal_external` |
| `/data2/hmy/RET_exp/HPV-TCT3multi-RETFound_MAE-main-new-good-multi-0713/5centers_multi_internal_external_final` | `/data2/hmy/5Center_datas/5centers_multi_internal_external_final` |

---

## 🔧 使用新路径

### 在训练脚本中

```python
# 推荐使用最终划分版本
python train_vlm_causal_clip.py \
    --data_path /data2/hmy/5Center_datas/5centers_multi_internal_external_final \
    --batch_size 10
```

### 在代码中

```python
# 使用绝对路径（推荐）
DATA_PATH = "/data2/hmy/5Center_datas/5centers_multi_internal_external_final"

# 或使用相对路径（在新项目中）
DATA_PATH = "../../data/5centers_multi_internal_external_final"
```

---

## ⚠️ 注意事项

1. **迁移状态**: 数据集迁移可能仍在进行中，请检查迁移日志
   ```bash
   tail -f /data2/hmy/5Center_datas/migration.log
   ```

2. **验证数据**: 迁移完成后，请验证数据完整性
   ```bash
   ls -lh /data2/hmy/5Center_datas/
   du -sh /data2/hmy/5Center_datas/*
   ```

3. **测试加载**: 更新路径后，请测试数据加载功能
   ```python
   import pandas as pd
   df = pd.read_csv('/data2/hmy/5Center_datas/5centers_multi_internal_external_final/train_labels.csv')
   print(f"训练集样本数: {len(df)}")
   ```

4. **相对路径**: 如果使用相对路径，确保工作目录正确

---

## 📝 待检查的文件

以下文件可能仍包含旧路径，需要手动检查：

1. **训练脚本** (`.sh`文件)
   - `run_optimized_training.sh`
   - `start_optimized_clip_training.sh`
   - 其他shell脚本

2. **配置文件** (`.json`, `.yaml`)
   - 检查configs目录下的配置文件

3. **文档文件** (`.md`)
   - 部分文档可能仍包含旧路径示例

---

## ✅ 验证清单

- [x] 更新Python代码中的绝对路径
- [x] 更新文档中的路径示例
- [x] 更新新项目中的软链接
- [x] 更新训练脚本默认参数
- [ ] 检查shell脚本中的路径（需要手动检查）
- [ ] 验证数据迁移完成
- [ ] 测试数据加载功能

---

**最后更新**: 2025-12-17

