# Bio-COT 3.0 快速启动训练指南

## 📋 当前状态

✅ **已完成**：
- 知识库已构建：`knowledge_base/medical_guidelines.json`
- 训练脚本已配置为30个epoch
- 可视化功能已集成

⏳ **进行中**：
- Knowledge Note Embeddings正在后台生成（可能需要10-30分钟）

---

## 🚀 启动训练的两种方式

### 方式1：自动检查并启动（推荐）

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate
./check_and_train.sh
```

这个脚本会：
1. 检查embeddings是否已生成
2. 如果已生成，自动启动训练
3. 如果未生成，显示生成进度

### 方式2：手动启动

#### Step 1: 检查embeddings是否生成完成

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0
ls -lh data/knowledge_embeddings.pt
```

如果文件存在且大小合理（约3-5MB），说明已生成完成。

#### Step 2: 如果未生成，等待或手动生成

```bash
# 查看生成进度
tail -f generate_embeddings.log

# 或者手动生成（如果后台进程失败）
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate
python knowledge_base/generate_knowledge_notes.py \
    --csv_paths \
        /data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal/internal_train/labels.csv \
        /data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal/internal_val/labels.csv \
    --guidelines knowledge_base/medical_guidelines.json \
    --output data/knowledge_embeddings.pt \
    --device cuda:1 \
    --batch_size 32
```

#### Step 3: 启动训练

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate
python training/train_bio_cot_v3.py 2>&1 | tee training_output.log
```

---

## 📊 训练监控

### 实时查看训练日志

```bash
# 方式1：查看输出日志
tail -f training_output.log

# 方式2：使用监控脚本
python monitor_training.py
```

### 查看训练进度

训练过程中会自动：
- ✅ 保存最佳模型到 `checkpoints/best_model_v3_<timestamp>.pth`
- ✅ 保存训练历史到 `logs/training_history_<timestamp>.json`
- ✅ 生成可视化图表到 `logs/training_curves_<timestamp>.png`

---

## 📈 训练完成后查看结果

### 1. 查看训练历史

```bash
python monitor_training.py
```

### 2. 生成详细可视化

```bash
python visualize_results.py
```

这会生成一个综合的可视化图表，包含：
- Loss曲线
- Accuracy曲线
- AUC曲线
- F1-Score曲线
- Loss组件分解
- 综合性能指标
- 性能趋势分析
- 训练总结

---

## ⚙️ 训练配置

当前配置（30个epoch）：
- **Batch Size**: 16
- **Learning Rate**: 0.00024
- **Warm-up Epochs**: 5（前5个epoch beta=1.0）
- **Beta衰减**: Epoch 5-20线性递减
- **最终Beta**: 0.1（强过滤模式）

---

## 🔍 预期训练时间

- **Knowledge Note Embeddings生成**: 10-30分钟（取决于数据集大小）
- **30个Epoch训练**: 约2-4小时（取决于GPU和数据集大小）

---

## 📝 注意事项

1. **虚拟环境**：确保使用 `/data2/hmy/VLM_Caus_Rm_Mics/my_retfound`
2. **GPU设备**：训练脚本默认使用 `cuda:0`，可在脚本中修改
3. **显存**：如果显存不足，可以减小 `batch_size`（在 `config.py` 中修改）
4. **中断恢复**：训练会自动保存最佳模型，可以手动加载继续训练

---

## 🐛 常见问题

### Q1: embeddings文件不存在
**A**: 等待生成完成，或手动运行生成脚本

### Q2: 显存不足
**A**: 减小 `batch_size`（在 `config.py` 中修改为8或更小）

### Q3: 训练中断
**A**: 检查 `checkpoints/` 目录中的最佳模型，可以加载继续训练

---

**最后更新**: 2025-01-08

