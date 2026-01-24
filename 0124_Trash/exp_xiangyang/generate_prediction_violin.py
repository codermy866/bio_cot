#!/usr/bin/env python3
"""
加载模型并生成预测概率，然后生成小提琴图
"""
import torch
import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train_bio_cot_multimodal_balanced_xiangyang import (
    XiangyangMultimodalDatasetFromCSV,
    extract_features_with_resnet50,
    BioCOTModel,
    plot_advanced_violin_analysis,
    plot_prediction_distribution
)
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

def collate_fn(batch):
    """自定义collate函数"""
    oct_images = torch.stack([item['oct_images'] for item in batch])
    colposcopy_images = torch.stack([item['colposcopy_images'] for item in batch])
    clinical_features = torch.stack([item['clinical_features'] for item in batch])
    clinical_data = [item['clinical_data'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    oct_ids = [item['oct_id'] for item in batch]
    
    return {
        'oct_images': oct_images,
        'colposcopy_images': colposcopy_images,
        'clinical_features': clinical_features,
        'clinical_data': clinical_data,
        'label': labels,
        'oct_id': oct_ids
    }

def generate_predictions_and_violin():
    """生成预测并绘制小提琴图"""
    timestamp = '20260107_155330'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集
    val_csv = Path(f'experiments/exp_xiangyang/logs/val_labels_balanced_{timestamp}.csv')
    if not val_csv.exists():
        print(f"❌ 验证集CSV文件不存在: {val_csv}")
        return
    
    val_dataset = XiangyangMultimodalDatasetFromCSV(
        csv_path=str(val_csv),
        oct_num_frames=60,
        max_col_images=3
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # 加载模型
    checkpoint_path = Path(f'experiments/exp_xiangyang/checkpoints_multimodal_balanced/best_model_balanced_{timestamp}.pth')
    if not checkpoint_path.exists():
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        return
    
    print(f"📦 加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 创建模型
    model = BioCOTModel(
        embed_dim=768,
        num_classes=2,
        num_centers=1,
        input_dim=2048
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # 生成预测
    print("🔮 生成预测概率...")
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in val_loader:
            oct_images = batch['oct_images'].to(device)
            colposcopy_images = batch['colposcopy_images'].to(device)
            clinical_features = batch['clinical_features'].to(device)
            clinical_data = batch['clinical_data']
            labels = batch['label'].to(device)
            
            # 提取特征
            oct_features = extract_features_with_resnet50(oct_images, device)
            colpo_features = extract_features_with_resnet50(colposcopy_images, device)
            
            # 准备clinical_data（转换为字典格式）
            batch_clinical_dict = {
                'hpv': [item.get('hpv', 0) for item in clinical_data],
                'tct': [item.get('tct', 0) for item in clinical_data],
                'age': [item.get('age', 0) for item in clinical_data]
            }
            
            # 模型推理
            output = model(
                oct_features=oct_features,
                colpo_features=colpo_features,
                clinical_features=clinical_features,
                clinical_data=batch_clinical_dict
            )
            
            probs = torch.softmax(output['logits'], dim=1)[:, 1].cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels_np)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    print(f"✅ 生成了 {len(all_probs)} 个预测")
    print(f"   阴性样本: {np.sum(all_labels == 0)}, 阳性样本: {np.sum(all_labels == 1)}")
    
    # 生成小提琴图
    print("\n📊 生成高级小提琴图分析...")
    output_dir = Path('experiments/exp_xiangyang/results_multimodal_balanced')
    plot_advanced_violin_analysis(all_labels, all_probs, output_dir, timestamp)
    
    # 重新生成预测分布图（包含小提琴图）
    print("\n📊 重新生成预测分布分析图（包含小提琴图）...")
    plot_prediction_distribution(all_labels, all_probs, output_dir, timestamp)
    
    # 保存预测结果到JSON
    results_file = Path(f'experiments/exp_xiangyang/results_multimodal_balanced/results_bio_cot_multimodal_balanced_{timestamp}.json')
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        results['final_val_labels'] = all_labels.tolist()
        results['final_val_probs'] = all_probs.tolist()
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ 预测结果已保存到: {results_file}")
    
    print("\n✅ 小提琴图生成完成！")

if __name__ == '__main__':
    generate_predictions_and_violin()

