# 🚀 启动训练 - 完整指南

## 当前状态

✅ **已完成**：
- 代码修复（3个关键漏洞）
- 知识库构建
- 训练脚本配置（30个epoch）
- 可视化功能集成

⏳ **需要完成**：
- 生成Knowledge Note Embeddings（遇到torch版本问题，已修复）

---

## 📋 快速启动步骤

### Step 1: 生成Knowledge Note Embeddings

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0
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

**注意**：如果遇到torch版本问题，代码会自动fallback到bert-base-uncased。

**预期时间**：10-30分钟

### Step 2: 检查生成结果

```bash
ls -lh data/knowledge_embeddings.pt
```

如果文件存在且大小合理（约3-5MB），说明生成成功。

### Step 3: 启动训练

```bash
cd /data2/hmy/VLM_Caus_Rm_Mics/experiments/exp_bio3.0
source /data2/hmy/VLM_Caus_Rm_Mics/my_retfound/bin/activate

python training/train_bio_cot_v3.py 2>&1 | tee training_output.log
```

**预期时间**：2-4小时（30个epoch）

---

## 📊 训练监控

### 实时查看日志

```bash
tail -f training_output.log
```

### 使用监控脚本

```bash
python monitor_training.py
```

---

## 📈 训练完成后

### 1. 查看训练历史

```bash
python monitor_training.py
```

### 2. 生成详细可视化

```bash
python visualize_results.py
```

---

## ⚙️ 训练配置

- **Epochs**: 30
- **Batch Size**: 16
- **Learning Rate**: 0.00024
- **Warm-up**: 5 epochs
- **Beta策略**: 动态衰减（1.0 → 0.1）

---

## 🔍 输出文件

训练完成后会生成：
- `logs/train_bio_cot_v3_<timestamp>.log` - 训练日志
- `logs/training_history_<timestamp>.json` - 训练历史
- `checkpoints/best_model_v3_<timestamp>.pth` - 最佳模型
- `logs/training_curves_<timestamp>.png` - 可视化图表

---

**准备好了！按照上述步骤操作即可开始训练。**

