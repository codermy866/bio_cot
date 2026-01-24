#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预计算VLM特征脚本
将所有训练/验证样本的VLM特征预先提取并保存，避免训练时重复计算
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig

def precompute_vlm_features(dataset, output_dir, device='cuda:1', batch_size=8):
    """
    预计算VLM特征
    
    Args:
        dataset: EnhancedMultimodalCervicalDataset
        output_dir: 输出目录，保存特征文件
        device: 设备
        batch_size: 批处理大小（VLM处理较慢，使用小batch）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载VLM模型
    print("🔄 加载VLM模型...")
    vlm_model = "Qwen/Qwen2-VL-2B-Instruct"
    config = AutoConfig.from_pretrained(vlm_model)
    config.output_hidden_states = True
    
    vlm = Qwen2VLForConditionalGeneration.from_pretrained(
        vlm_model,
        config=config,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    vlm.eval()
    
    processor = AutoProcessor.from_pretrained(vlm_model, use_fast=False)
    
    print(f"✅ VLM模型加载完成，使用设备: {device}")
    
    # 创建数据加载器
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 存储特征
    all_features = []
    all_patient_ids = []
    
    print(f"🔄 开始提取VLM特征，共 {len(dataset)} 个样本...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting VLM features")):
            # 解析batch
            if isinstance(batch, dict):
                oct_images = batch.get('oct_images')
                clinical_data = batch.get('clinical_data', {})
                patient_ids = batch.get('patient_id', [f'batch_{batch_idx}_{i}' for i in range(len(oct_images))])
            else:
                continue
            
            if oct_images is None:
                print(f"⚠️ Batch {batch_idx}: 没有OCT图像，跳过")
                continue
            
            B = oct_images.size(0)
            
            # 准备图像和文本
            text_prompts = []
            image_list = []
            
            for i in range(B):
                # 构建临床文本
                if clinical_data:
                    hpv_val = clinical_data.get('hpv', [0] * B)[i] if isinstance(clinical_data.get('hpv'), list) else 0
                    tct_val = clinical_data.get('tct', ['NILM'] * B)[i] if isinstance(clinical_data.get('tct'), list) else 'NILM'
                    age_val = clinical_data.get('age', [45] * B)[i] if isinstance(clinical_data.get('age'), list) else 45
                else:
                    hpv_val = 0
                    tct_val = 'NILM'
                    age_val = 45
                
                hpv_status = "阳性" if int(hpv_val) == 1 else "阴性"
                text = f"患者年龄{int(age_val)}岁，HPV检测结果{hpv_status}，TCT检查结果为{tct_val}。"
                text_prompts.append(text)
                
                # 处理图像
                img_tensor = oct_images[i]
                if img_tensor.dim() == 4 and img_tensor.size(0) > 1:
                    img_tensor = img_tensor[0]  # 取第一帧
                elif img_tensor.dim() == 4 and img_tensor.size(0) == 1:
                    img_tensor = img_tensor[0]
                
                if img_tensor.max() > 1.0:
                    img_tensor = img_tensor / 255.0
                img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
                
                to_pil = transforms.ToPILImage()
                img_pil = to_pil(img_tensor.cpu())
                image_list.append(img_pil)
            
            # 使用VLM处理
            try:
                processed_texts = []
                processed_images = []
                
                for text, img in zip(text_prompts, image_list):
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": text}
                        ]
                    }]
                    processed_text = processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    processed_texts.append(processed_text)
                    processed_images.append(img)
                
                inputs = processor(
                    text=processed_texts,
                    images=processed_images,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
                # VLM forward
                outputs = vlm(**inputs, output_hidden_states=True)
                
                # 提取特征
                if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                    hidden_states = outputs.last_hidden_state
                    text_features = hidden_states.mean(dim=1)  # [B, hidden_size]
                elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    hidden_states = outputs.hidden_states[-1]
                    text_features = hidden_states.mean(dim=1)
                else:
                    print(f"⚠️ Batch {batch_idx}: 无法提取VLM特征，使用零向量")
                    text_features = torch.zeros(B, 1536).to(device)
                
                # 保存特征
                for i in range(B):
                    patient_id = patient_ids[i] if i < len(patient_ids) else f'batch_{batch_idx}_{i}'
                    feature = text_features[i].cpu().numpy()  # [1536]
                    
                    # 保存为.npy文件
                    feature_file = output_dir / f"{patient_id}_vlm_feat.npy"
                    np.save(feature_file, feature)
                    
                    all_features.append(feature)
                    all_patient_ids.append(patient_id)
                
            except Exception as e:
                print(f"❌ Batch {batch_idx}: VLM处理失败: {e}")
                # 使用零向量作为fallback
                for i in range(B):
                    patient_id = patient_ids[i] if i < len(patient_ids) else f'batch_{batch_idx}_{i}'
                    feature = np.zeros(1536)
                    feature_file = output_dir / f"{patient_id}_vlm_feat.npy"
                    np.save(feature_file, feature)
                    all_features.append(feature)
                    all_patient_ids.append(patient_id)
    
    # 保存索引文件
    index_file = output_dir / "vlm_features_index.txt"
    with open(index_file, 'w') as f:
        for pid in all_patient_ids:
            f.write(f"{pid}\n")
    
    print(f"✅ VLM特征提取完成！")
    print(f"   - 总样本数: {len(all_features)}")
    print(f"   - 特征维度: {all_features[0].shape}")
    print(f"   - 特征文件保存在: {output_dir}")
    print(f"   - 索引文件: {index_file}")
    
    return all_features, all_patient_ids


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='数据根目录')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True, help='数据集划分')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--device', type=str, default='cuda:1', help='设备')
    parser.add_argument('--batch_size', type=int, default=8, help='批处理大小')
    args = parser.parse_args()
    
    # 加载数据集
    print(f"📂 加载数据集: {args.split}")
    if args.split == 'train':
        dataset = EnhancedMultimodalCervicalDataset(
            root=Path(args.data_root) / 'internal_train' / 'train',
            labels_file=Path(args.data_root) / 'train_labels.csv',
            use_pretrained_backbones=True
        )
    elif args.split == 'val':
        dataset = EnhancedMultimodalCervicalDataset(
            root=Path(args.data_root) / 'internal_train' / 'val',
            labels_file=Path(args.data_root) / 'val_labels.csv',
            use_pretrained_backbones=True
        )
    else:
        dataset = EnhancedMultimodalCervicalDataset(
            root=Path(args.data_root) / 'external_test',
            labels_file=Path(args.data_root) / 'external_test_labels.csv',
            use_pretrained_backbones=True
        )
    
    print(f"✅ 数据集加载完成: {len(dataset)} 个样本")
    
    # 预计算特征
    precompute_vlm_features(
        dataset=dataset,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()



