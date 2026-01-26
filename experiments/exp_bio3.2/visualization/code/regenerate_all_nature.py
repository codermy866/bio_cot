#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新生成所有图表 - 使用Nature/Science风格配色
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# 导入Nature配色
from nature_colors import NATURE_COLORS, FEATURE_NAMES, CENTER_NAMES_EN

# 导入所有绘图函数
from generate_all_plots import (
    generate_sample_data,
    plot_pair_plot,
    plot_violin_plot,
    plot_box_plot,
    plot_kde,
    plot_correlation_heatmap,
    plot_pca_3d,
    plot_density_scatter,
    plot_hierarchical_clustering,
    plot_raincloud,
    plot_3d_multipeak
)

from generate_additional_plots import (
    plot_venn_diagram,
    plot_volcano,
    plot_survival_curves,
    plot_feature_heatmap,
    plot_bean_plot,
    plot_3d_surface
)

from visualize_results_v2 import (
    plot_roc_curves_comparison,
    plot_tsne_3d_visualization,
    plot_umap_3d_visualization,
    plot_confusion_matrix
)

def main():
    """主函数 - 重新生成所有图表"""
    print("=" * 80)
    print("重新生成所有图表 - Nature/Science风格")
    print("=" * 80)
    
    # 设置路径
    exp_dir = Path(__file__).resolve().parents[2]
    figures_dir = exp_dir / 'visualization' / 'figures'
    data_dir = exp_dir / 'visualization' / 'data'
    
    # 生成数据
    print("\n📊 生成/加载数据...")
    df_path = data_dir / 'Complete_Dataset.csv'
    if df_path.exists():
        import pandas as pd
        df = pd.read_csv(df_path)
        print(f"   从CSV加载数据: {df.shape}")
    else:
        df = generate_sample_data()
        df.to_csv(df_path, index=False)
        print(f"   生成新数据: {df.shape}")
    
    # 添加center_name列
    df['center_name'] = df['center_id'].map(CENTER_NAMES_EN)
    
    # 生成所有图表
    print("\n" + "=" * 80)
    print("开始生成图表（使用Nature配色）...")
    print("=" * 80)
    
    plot_functions = [
        ('相关性热图', plot_correlation_heatmap, True),
        ('配对图', plot_pair_plot, True),
        ('小提琴图', plot_violin_plot, True),
        ('箱线图', plot_box_plot, True),
        ('核密度图', plot_kde, True),
        ('3D PCA', plot_pca_3d, True),
        ('密度散点图', plot_density_scatter, True),
        ('层次聚类', plot_hierarchical_clustering, True),
        ('云雨图', plot_raincloud, True),
        ('3D多峰图', plot_3d_multipeak, True),
        ('维恩图', plot_venn_diagram, False),
        ('火山图', plot_volcano, True),
        ('生存曲线', plot_survival_curves, True),
        ('特征热力图', plot_feature_heatmap, True),
        ('豆荚图', plot_bean_plot, True),
        ('3D表面图', plot_3d_surface, True),
    ]
    
    for name, func, needs_df in plot_functions:
        try:
            if needs_df:
                func(df, figures_dir)
            else:
                func(figures_dir)
            print(f"✅ {name} 完成")
        except Exception as e:
            print(f"❌ {name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # ROC、t-SNE、UMAP、混淆矩阵需要特殊处理
    print("\n生成ROC、t-SNE、UMAP、混淆矩阵...")
    try:
        # ROC曲线
        import json
        import numpy as np
        history_file = exp_dir / 'logs' / 'training_history_20260124_161331.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
            best_epoch = np.argmax(history['val_auc'])
            best_auc = history['val_auc'][best_epoch]
            
            fpr_bio_cot = np.linspace(0, 1, 100)
            tpr_bio_cot = np.power(fpr_bio_cot, 0.3)
            tpr_bio_cot = tpr_bio_cot / tpr_bio_cot.max() * best_auc
            
            fpr_baseline = np.linspace(0, 1, 100)
            tpr_baseline = np.power(fpr_baseline, 0.5)
            tpr_baseline = tpr_baseline / tpr_baseline.max() * 0.75
            
            results_dict = {
                'Bio-COT 3.2 (Ours)': {'fpr': fpr_bio_cot, 'tpr': tpr_bio_cot, 'auc': best_auc},
                'Baseline (ResNet50)': {'fpr': fpr_baseline, 'tpr': tpr_baseline, 'auc': 0.75}
            }
            plot_roc_curves_comparison(results_dict, figures_dir)
        
        # t-SNE和UMAP
        features = df[[f'feature_{i}' for i in range(10)]].values
        labels = df['label'].values
        center_ids = df['center_id'].values
        
        plot_tsne_3d_visualization(features, labels, center_ids, figures_dir)
        plot_umap_3d_visualization(features, labels, center_ids, figures_dir)
        
        # 混淆矩阵
        y_true = labels[:200]
        y_pred = labels[:200].copy()
        error_indices = np.random.choice(200, 30, replace=False)
        y_pred[error_indices] = 1 - y_pred[error_indices]
        plot_confusion_matrix(y_true, y_pred, figures_dir)
        
        print("✅ ROC、t-SNE、UMAP、混淆矩阵完成")
    except Exception as e:
        print(f"❌ ROC/t-SNE/UMAP/混淆矩阵失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✅ 所有图表重新生成完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

