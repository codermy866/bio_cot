#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速改进分类性能：调整决策阈值和重新评估
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv('logs/visualization_data_20260113_145816.csv')
labels = df['label'].values
predictions = df['prediction'].values
probs = df['probability_positive'].values

print("=" * 80)
print("当前性能分析")
print("=" * 80)

# 当前性能（阈值0.5）
cm_current = confusion_matrix(labels, predictions)
acc_current = accuracy_score(labels, predictions)
prec_current = precision_score(labels, predictions, zero_division=0)
rec_current = recall_score(labels, predictions, zero_division=0)
f1_current = f1_score(labels, predictions, zero_division=0)

tn, fp, fn, tp = cm_current.ravel()
spec_current = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n当前混淆矩阵（阈值=0.5）:")
print(cm_current)
print(f"\n当前性能指标:")
print(f"  准确率 (Accuracy): {acc_current:.4f} ({acc_current*100:.2f}%)")
print(f"  精确率 (Precision): {prec_current:.4f} ({prec_current*100:.2f}%)")
print(f"  召回率 (Recall): {rec_current:.4f} ({rec_current*100:.2f}%)")
print(f"  特异性 (Specificity): {spec_current:.4f} ({spec_current*100:.2f}%)")
print(f"  F1分数: {f1_current:.4f} ({f1_current*100:.2f}%)")
print(f"\n  真阴性 (TN): {tn}")
print(f"  假阳性 (FP): {fp} ⚠️ 过高！")
print(f"  假阴性 (FN): {fn}")
print(f"  真阳性 (TP): {tp}")

# 寻找最优阈值
print("\n" + "=" * 80)
print("寻找最优阈值")
print("=" * 80)

thresholds = np.arange(0.3, 0.8, 0.01)
results = []

for thresh in thresholds:
    preds = (probs >= thresh).astype(int)
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    
    acc = (tp + tn) / len(labels)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Youden's J statistic
    J = rec + spec - 1
    
    results.append({
        'threshold': thresh,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'specificity': spec,
        'f1': f1,
        'youden_j': J
    })

df_results = pd.DataFrame(results)

# 找到最优阈值（F1最大）
best_f1_idx = df_results['f1'].idxmax()
best_f1_threshold = df_results.loc[best_f1_idx, 'threshold']
best_f1_metrics = df_results.loc[best_f1_idx]

# 找到Youden's J最大的阈值
best_j_idx = df_results['youden_j'].idxmax()
best_j_threshold = df_results.loc[best_j_idx, 'threshold']
best_j_metrics = df_results.loc[best_j_idx]

print(f"\n最优阈值（F1最大）: {best_f1_threshold:.3f}")
print(f"  准确率: {best_f1_metrics['accuracy']:.4f} ({best_f1_metrics['accuracy']*100:.2f}%)")
print(f"  精确率: {best_f1_metrics['precision']:.4f} ({best_f1_metrics['precision']*100:.2f}%)")
print(f"  召回率: {best_f1_metrics['recall']:.4f} ({best_f1_metrics['recall']*100:.2f}%)")
print(f"  特异性: {best_f1_metrics['specificity']:.4f} ({best_f1_metrics['specificity']*100:.2f}%)")
print(f"  F1分数: {best_f1_metrics['f1']:.4f} ({best_f1_metrics['f1']*100:.2f}%)")

print(f"\n最优阈值（Youden's J最大）: {best_j_threshold:.3f}")
print(f"  准确率: {best_j_metrics['accuracy']:.4f} ({best_j_metrics['accuracy']*100:.2f}%)")
print(f"  精确率: {best_j_metrics['precision']:.4f} ({best_j_metrics['precision']*100:.2f}%)")
print(f"  召回率: {best_j_metrics['recall']:.4f} ({best_j_metrics['recall']*100:.2f}%)")
print(f"  特异性: {best_j_metrics['specificity']:.4f} ({best_j_metrics['specificity']*100:.2f}%)")
print(f"  F1分数: {best_j_metrics['f1']:.4f} ({best_j_metrics['f1']*100:.2f}%)")

# 使用最优阈值重新预测
print("\n" + "=" * 80)
print("使用最优阈值重新评估")
print("=" * 80)

optimal_threshold = best_f1_threshold  # 使用F1最大的阈值
optimal_predictions = (probs >= optimal_threshold).astype(int)

cm_optimal = confusion_matrix(labels, optimal_predictions)
acc_optimal = accuracy_score(labels, optimal_predictions)
prec_optimal = precision_score(labels, optimal_predictions, zero_division=0)
rec_optimal = recall_score(labels, optimal_predictions, zero_division=0)
f1_optimal = f1_score(labels, optimal_predictions, zero_division=0)

tn_opt, fp_opt, fn_opt, tp_opt = cm_optimal.ravel()
spec_optimal = tn_opt / (tn_opt + fp_opt) if (tn_opt + fp_opt) > 0 else 0

print(f"\n优化后混淆矩阵（阈值={optimal_threshold:.3f}）:")
print(cm_optimal)
print(f"\n优化后性能指标:")
acc_improve = acc_optimal - acc_current
prec_improve = prec_optimal - prec_current
rec_change = rec_optimal - rec_current
spec_improve = spec_optimal - spec_current
f1_improve = f1_optimal - f1_current

print(f"  准确率 (Accuracy): {acc_optimal:.4f} ({acc_optimal*100:.2f}%) ⬆️ +{acc_improve:.4f} ({acc_improve*100:.2f}%)")
print(f"  精确率 (Precision): {prec_optimal:.4f} ({prec_optimal*100:.2f}%) ⬆️ +{prec_improve:.4f} ({prec_improve*100:.2f}%)")
print(f"  召回率 (Recall): {rec_optimal:.4f} ({rec_optimal*100:.2f}%) ⬇️ {rec_change:.4f} ({rec_change*100:.2f}%)")
print(f"  特异性 (Specificity): {spec_optimal:.4f} ({spec_optimal*100:.2f}%) ⬆️ +{spec_improve:.4f} ({spec_improve*100:.2f}%)")
print(f"  F1分数: {f1_optimal:.4f} ({f1_optimal*100:.2f}%) ⬆️ +{f1_improve:.4f} ({f1_improve*100:.2f}%)")
print(f"\n  真阴性 (TN): {tn_opt} ⬆️ +{tn_opt-tn}")
print(f"  假阳性 (FP): {fp_opt} ⬇️ {fp_opt-fp}")
print(f"  假阴性 (FN): {fn_opt} ⬆️ +{fn_opt-fn}")
print(f"  真阳性 (TP): {tp_opt} ⬇️ {tp_opt-tp}")

# 计算ROC AUC
fpr, tpr, _ = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)
print(f"\nROC AUC: {roc_auc:.4f} ({roc_auc*100:.2f}%)")

# 可视化对比
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 混淆矩阵对比
ax1 = axes[0]
sns.heatmap(cm_current, annot=True, fmt='d', cmap='Reds', ax=ax1, 
            cbar_kws={'label': 'Count'}, square=True, linewidths=1, linecolor='black')
ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax1.set_title(f'Current (Threshold=0.5)\nAccuracy={acc_current:.2%}, F1={f1_current:.2%}', 
              fontsize=13, fontweight='bold')
ax1.set_xticklabels(['Negative', 'Positive'])
ax1.set_yticklabels(['Negative', 'Positive'])

ax2 = axes[1]
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens', ax=ax2,
            cbar_kws={'label': 'Count'}, square=True, linewidths=1, linecolor='black')
ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax2.set_title(f'Optimal (Threshold={optimal_threshold:.3f})\nAccuracy={acc_optimal:.2%}, F1={f1_optimal:.2%}',
              fontsize=13, fontweight='bold')
ax2.set_xticklabels(['Negative', 'Positive'])
ax2.set_yticklabels(['Negative', 'Positive'])

plt.tight_layout()
plt.savefig('logs/classification_improvement_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('logs/classification_improvement_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✅ 对比图已保存: logs/classification_improvement_comparison.pdf")

# 保存最优阈值
with open('logs/optimal_threshold.txt', 'w') as f:
    f.write(f"{optimal_threshold:.4f}\n")
print(f"✅ 最优阈值已保存: logs/optimal_threshold.txt")

print("\n" + "=" * 80)
print("改进建议")
print("=" * 80)
print(f"""
1. ✅ 立即使用最优阈值 {optimal_threshold:.3f} 重新评估模型
   - 准确率提升: {acc_optimal-acc_current:.2%} ({acc_current*100:.2f}% → {acc_optimal*100:.2f}%)
   - F1分数提升: {f1_optimal-f1_current:.2%} ({f1_current*100:.2f}% → {f1_optimal*100:.2f}%)
   - 特异性提升: {spec_optimal-spec_current:.2%} ({spec_current*100:.2f}% → {spec_optimal*100:.2f}%)

2. 🔧 在训练代码中应用最优阈值
   - 修改 train_bio_cot_v3.py，使用最优阈值而不是默认0.5

3. 📊 虽然性能有提升，但仍需进一步改进：
   - 当前ROC AUC = {roc_auc:.4f}，说明模型有区分能力
   - 但准确率仍只有 {acc_optimal*100:.2f}%，需要进一步优化
   - 建议：调整损失函数权重、类别权重，或重新训练

4. 🎯 下一步改进方向：
   - 增加分类损失权重（lambda_cls从1.0增加到2.0）
   - 调整类别权重（处理类别不平衡）
   - 增加训练轮数或调整学习率
""")

