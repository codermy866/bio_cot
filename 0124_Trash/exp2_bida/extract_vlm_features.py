#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1: 离线VLM特征提取脚本
目标：遍历所有数据，提取Qwen2-VL特征并保存为.npy文件
彻底移除训练循环中的VLM推理，将训练速度提升50倍
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import json
from collections import defaultdict

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 注意：需要根据实际的数据集类调整导入
try:
    from src.data.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset, build_enhanced_dataset
except ImportError:
    from src.data.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig


def extract_vlm_features(dataset, output_file, device='cuda:1', batch_size=8):
    """
    提取VLM特征并保存
    
    Args:
        dataset: EnhancedMultimodalCervicalDataset
        output_file: 输出文件路径 (.npy)
        device: 设备
        batch_size: 批处理大小（VLM处理较慢，使用小batch）
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 加载VLM模型
    print("🔄 加载Qwen2-VL模型...")
    vlm_model = "Qwen/Qwen2-VL-2B-Instruct"
    config = AutoConfig.from_pretrained(vlm_model)
    config.output_hidden_states = True
    
    vlm = Qwen2VLForConditionalGeneration.from_pretrained(
        vlm_model,
        config=config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    vlm.eval()
    
    processor = AutoProcessor.from_pretrained(vlm_model, use_fast=False)
    
    print(f"✅ VLM模型加载完成，使用设备: {device}")
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 存储特征字典 {sample_id: feature}
    feature_dict = {}
    failed_samples = []
    
    print(f"🔄 开始提取VLM特征，共 {len(dataset)} 个样本...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting VLM features")):
            # 解析batch
            if isinstance(batch, dict):
                oct_images = batch.get('oct_images')
                clinical_feat = batch.get('clinical_features')
                
                # 从clinical_feat构建clinical_data
                if clinical_feat is not None:
                    batch_size_actual = clinical_feat.size(0)
                    clinical_data = {
                        'hpv': [1 if clinical_feat[i, 1].item() > 0.5 else 0 for i in range(batch_size_actual)],
                        'tct': [],
                        'age': [int(clinical_feat[i, 0].item() * 100) for i in range(batch_size_actual)]
                    }
                    tct_categories = ['ASC-US', 'LSIL', 'HSIL', 'NILM', 'OTHER']
                    for i in range(batch_size_actual):
                        tct_onehot = clinical_feat[i, 2:7]
                        tct_idx = tct_onehot.argmax().item()
                        clinical_data['tct'].append(tct_categories[tct_idx])
                else:
                    clinical_data = {}
                
                # 生成patient_id（使用batch索引）
                start_idx = batch_idx * batch_size
                if oct_images is not None:
                    B = oct_images.size(0)
                    patient_ids = [f'sample_{start_idx + i}' for i in range(B)]
                else:
                    patient_ids = []
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
                
                # 提取特征：使用last_hidden_state的平均池化
                if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                    hidden_states = outputs.last_hidden_state  # [B, seq_len, hidden_size]
                    # 取[EOS] token或平均池化
                    text_features = hidden_states.mean(dim=1)  # [B, hidden_size]
                elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    hidden_states = outputs.hidden_states[-1]  # 最后一层
                    text_features = hidden_states.mean(dim=1)
                else:
                    # Fallback: 使用logits
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                        text_features = logits.mean(dim=1)  # [B, vocab_size]
                        # 如果维度不对，需要投影
                        if text_features.size(-1) != 1536:
                            # 简单处理：取前1536维或填充
                            if text_features.size(-1) > 1536:
                                text_features = text_features[:, :1536]
                            else:
                                padding = torch.zeros(B, 1536 - text_features.size(-1)).to(device)
                                text_features = torch.cat([text_features, padding], dim=-1)
                    else:
                        print(f"⚠️ Batch {batch_idx}: 无法提取VLM特征，使用零向量")
                        text_features = torch.zeros(B, 1536).to(device)
                
                # 保存特征
                for i in range(B):
                    patient_id = patient_ids[i] if i < len(patient_ids) else f'batch_{batch_idx}_{i}'
                    feature = text_features[i].cpu().numpy().astype(np.float32)  # [1536]
                    feature_dict[patient_id] = feature
                
            except Exception as e:
                print(f"❌ Batch {batch_idx}: VLM处理失败: {e}")
                # 使用零向量作为fallback
                for i in range(B):
                    patient_id = patient_ids[i] if i < len(patient_ids) else f'batch_{batch_idx}_{i}'
                    feature = np.zeros(1536, dtype=np.float32)
                    feature_dict[patient_id] = feature
                    failed_samples.append(patient_id)
    
    # 保存特征字典
    print(f"💾 保存特征到: {output_file}")
    np.save(str(output_file), feature_dict)
    
    # 保存元数据
    metadata = {
        'total_samples': len(feature_dict),
        'failed_samples': failed_samples,
        'feature_dim': 1536,
        'model': vlm_model
    }
    metadata_file = output_file.with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ VLM特征提取完成！")
    print(f"   - 总样本数: {len(feature_dict)}")
    print(f"   - 特征维度: 1536")
    print(f"   - 失败样本: {len(failed_samples)}")
    print(f"   - 特征文件: {output_file}")
    print(f"   - 元数据文件: {metadata_file}")
    
    return feature_dict


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='提取VLM特征')
    parser.add_argument('--data_root', type=str, 
                       default='/data2/hmy/5Center_datas/5centers_multi_leave_centers_out',
                       help='数据根目录')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], 
                       default='train', help='数据集划分')
    parser.add_argument('--output_file', type=str, 
                       default='/data2/hmy/5Center_datas/5centers_multi_leave_centers_out/vlm_features_cache/train_vlm_features.npy',
                       help='输出文件路径')
    parser.add_argument('--device', type=str, default='cuda:1', help='设备')
    parser.add_argument('--batch_size', type=int, default=8, help='批处理大小')
    args = parser.parse_args()
    
    # 加载数据集
    print(f"📂 加载数据集: {args.split}")
    try:
        # 尝试使用build_enhanced_dataset
        if args.split == 'train':
            dataset = build_enhanced_dataset(
                root=Path(args.data_root) / 'internal_train' / 'train',
                labels_file=Path(args.data_root) / 'train_labels.csv',
                use_pretrained_backbones=True
            )
        elif args.split == 'val':
            dataset = build_enhanced_dataset(
                root=Path(args.data_root) / 'internal_train' / 'val',
                labels_file=Path(args.data_root) / 'val_labels.csv',
                use_pretrained_backbones=True
            )
        else:
            dataset = build_enhanced_dataset(
                root=Path(args.data_root) / 'external_test',
                labels_file=Path(args.data_root) / 'external_test_labels.csv',
                use_pretrained_backbones=True
            )
    except:
        # Fallback: 使用类（需要args参数）
        class DummyArgs:
            data_path = args.data_root
            input_size = 224
            oct_num_frames = 48
            oct_points = 12
            oct_frames_per_point = 10
            oct_cache_dir = None
            use_pretrained_backbones = True
        
        dummy_args = DummyArgs()
        if args.split == 'train':
            dataset = EnhancedMultimodalCervicalDataset(
                root=Path(args.data_root) / 'internal_train' / 'train',
                is_train='train',
                args=dummy_args,
                use_enhanced_oct=True
            )
        elif args.split == 'val':
            dataset = EnhancedMultimodalCervicalDataset(
                root=Path(args.data_root) / 'internal_train' / 'val',
                is_train='val',
                args=dummy_args,
                use_enhanced_oct=True
            )
        else:
            dataset = EnhancedMultimodalCervicalDataset(
                root=Path(args.data_root) / 'external_test',
                is_train='test',
                args=dummy_args,
                use_enhanced_oct=True
            )
    
    print(f"✅ 数据集加载完成: {len(dataset)} 个样本")
    
    # 提取特征
    extract_vlm_features(
        dataset=dataset,
        output_file=args.output_file,
        device=args.device,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()

