#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型加载和推理模块
支持加载多个训练好的模型并进行集成预测
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import DataLoader

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# Import models
from paper1_hierarchical_multimodal.models import HierarchicalMultimodalModel
from paper1_hierarchical_multimodal.models.hierarchical_multimodal_model_vit import HierarchicalMultimodalModelViT
from types import SimpleNamespace

# Try to import dataset builder, fallback to util/datasets if needed
try:
    from src.data.enhanced_multimodal_dataset import build_enhanced_dataset
except ImportError:
    try:
        # Fallback: use util/datasets
        from util.datasets import build_dataset as build_enhanced_dataset
    except ImportError:
        print("Warning: Could not import dataset builder. Will need to provide data loader manually.")
        build_enhanced_dataset = None


class ModelEnsemble(nn.Module):
    """
    模型集成类
    支持多个模型的加权集成或投票集成
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        ensemble_method: str = 'weighted_average'  # 'weighted_average', 'voting', 'stacking'
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        
        if weights is None:
            # 默认等权重
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # 归一化权重
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        print(f"Ensemble initialized with {len(models)} models")
        print(f"  Method: {ensemble_method}")
        print(f"  Weights: {self.weights}")
    
    def forward(self, oct_images, col_images, clinical_features, labels=None):
        """
        集成前向传播
        
        Returns:
            outputs: 包含logits和概率的字典
        """
        all_logits = []
        all_probs = []
        
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                outputs = model(oct_images, col_images, clinical_features, labels)
                logits = outputs['logits']
                probs = torch.softmax(logits, dim=-1)
                
                all_logits.append(logits * weight)
                all_probs.append(probs * weight)
        
        # 加权平均
        ensemble_logits = sum(all_logits)
        ensemble_probs = sum(all_probs)
        
        return {
            'logits': ensemble_logits,
            'probs': ensemble_probs,
            'ensemble_method': self.ensemble_method
        }


class MultimodalModelLoader:
    """
    多模态模型加载器
    支持加载单个模型或集成多个模型
    """
    
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = torch.device(device)
        self.models = {}
        self.ensemble = None
        
        print(f"Model loader initialized on device: {self.device}")
    
    def load_hierarchical_model(
        self,
        model_path: str,
        use_vit_backbone: bool = False,
        backbone_type: str = 'vit',
        vit_model_name: str = 'vit_base_patch16_224',
        embed_dim: int = 768,
        num_classes: int = 2,
        clinical_dim: int = 7,
        contrastive_weight: float = 0.1,
        input_size: int = 224
    ) -> nn.Module:
        """
        加载层次化多模态模型
        
        Args:
            model_path: 模型权重路径
            use_vit_backbone: 是否使用ViT backbone
            ...其他模型参数
        
        Returns:
            加载的模型
        """
        print(f"\nLoading model from: {model_path}")
        
        if use_vit_backbone:
            model = HierarchicalMultimodalModelViT(
                embed_dim=embed_dim,
                num_classes=num_classes,
                clinical_dim=clinical_dim,
                contrastive_weight=contrastive_weight,
                backbone_type=backbone_type,
                model_name=vit_model_name,
                use_pretrained=True,
                input_size=input_size
            )
        else:
            model = HierarchicalMultimodalModel(
                embed_dim=embed_dim,
                num_classes=num_classes,
                clinical_dim=clinical_dim,
                contrastive_weight=contrastive_weight
            )
        
        # 加载权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 处理不同的checkpoint格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # 加载权重（允许部分匹配）
            model.load_state_dict(state_dict, strict=False)
            print(f"  ✅ Model loaded successfully")
            
            # 打印模型信息
            if isinstance(checkpoint, dict) and 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                print(f"  Model metrics: {metrics}")
        else:
            print(f"  ⚠️  Model file not found, using random initialization")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def load_best_models(
        self,
        model_configs: List[Dict],
        create_ensemble: bool = True,
        ensemble_weights: Optional[List[float]] = None
    ) -> Union[nn.Module, ModelEnsemble]:
        """
        加载多个最佳模型并创建集成
        
        Args:
            model_configs: 模型配置列表，每个配置包含模型路径和参数
            create_ensemble: 是否创建集成模型
            ensemble_weights: 集成权重（如果为None，则从config中获取或等权重）
        
        Returns:
            单个模型或集成模型
        """
        models = []
        
        # 如果没有提供权重，尝试从config中获取
        if ensemble_weights is None and len(model_configs) > 0:
            ensemble_weights = [config.get('weight', 1.0 / len(model_configs)) for config in model_configs]
            # 归一化
            total = sum(ensemble_weights)
            ensemble_weights = [w / total for w in ensemble_weights]
        
        for i, config in enumerate(model_configs):
            print(f"\n[{i+1}/{len(model_configs)}] Loading {config.get('name', 'Model')}...")
            model_params = {k: v for k, v in config.items() if k not in ['name', 'weight', 'auc']}
            try:
                model = self.load_hierarchical_model(**model_params)
                models.append(model)
            except RuntimeError as exc:
                print(f"⚠️  Failed to load {config.get('name', 'Model')}: {exc}")
                continue
        
        if create_ensemble and len(models) > 1:
            print(f"\n创建集成模型 (共{len(models)}个模型)...")
            self.ensemble = ModelEnsemble(
                models=models,
                weights=ensemble_weights,
                ensemble_method='weighted_average'
            )
            return self.ensemble
        elif len(models) == 1:
            return models[0]
        else:
            return models
    
    def predict_batch(
        self,
        model: nn.Module,
        oct_images: torch.Tensor,
        col_images: torch.Tensor,
        clinical_features: torch.Tensor,
        return_probs: bool = True
    ) -> Dict:
        """
        批量预测
        
        Args:
            model: 模型
            oct_images: OCT图像 [B, T, C, H, W] 或 [B, C, H, W]
            col_images: Colposcopy图像 [B, C, H, W]
            clinical_features: 临床特征 [B, clinical_dim]
            return_probs: 是否返回概率
        
        Returns:
            包含预测结果的字典
        """
        model.eval()
        
        with torch.no_grad():
            # 移动到设备
            oct_images = oct_images.to(self.device)
            col_images = col_images.to(self.device)
            clinical_features = clinical_features.to(self.device)
            
            # 前向传播
            outputs = model(oct_images, col_images, clinical_features)
            
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            result = {
                'logits': logits.cpu().numpy(),
                'predictions': preds.cpu().numpy(),
            }
            
            if return_probs:
                result['probabilities'] = probs.cpu().numpy()
                result['probabilities_class1'] = probs[:, 1].cpu().numpy()  # 病变概率
            
            return result
    
    def predict_dataset(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        return_labels: bool = True
    ) -> Dict:
        """
        对整个数据集进行预测
        
        Args:
            model: 模型
            data_loader: 数据加载器
            return_labels: 是否返回真实标签
        
        Returns:
            包含所有预测结果的字典
        """
        all_predictions = []
        all_probabilities = []
        all_logits = []
        all_labels = []
        
        model.eval()
        
        print(f"Predicting on {len(data_loader)} batches...")
        
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(data_loader)}")
            
            oct_images = batch['oct_images']
            col_images = batch['col_images']
            clinical_features = batch['clinical_features']
            labels = batch['labels'] if 'labels' in batch else None
            
            # 预测
            result = self.predict_batch(
                model, oct_images, col_images, clinical_features
            )
            
            all_predictions.extend(result['predictions'])
            all_probabilities.extend(result['probabilities_class1'])
            all_logits.extend(result['logits'])
            
            if return_labels and labels is not None:
                all_labels.extend(labels.cpu().numpy())
        
        output = {
            'predictions': np.array(all_predictions),
            'probabilities': np.array(all_probabilities),
            'logits': np.array(all_logits)
        }
        
        if return_labels and len(all_labels) > 0:
            output['labels'] = np.array(all_labels)
        
        print(f"✅ Prediction complete: {len(all_predictions)} samples")
        
        return output


def create_data_loader(
    data_path: str,
    split: str = 'test',  # 'train' or 'test'
    batch_size: int = 8,
    num_workers: int = 4,
    input_size: int = 224,
    oct_num_frames: int = 32,
    oct_frames_per_point: int = 5
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        data_path: 数据路径
        split: 数据集分割（'train' or 'test'）
        ...其他参数
    
    Returns:
        数据加载器
    """
    from torch.utils.data import DataLoader
    
    if build_enhanced_dataset is None:
        # Fallback: use util/datasets
        from util.datasets import build_dataset
        from types import SimpleNamespace as NS
        
        args = NS(
            data_path=data_path,
            input_size=input_size,
            oct_num_frames=oct_num_frames,
            oct_points=12,
            oct_frames_per_point=oct_frames_per_point,
            col_num_frames=1,
            oct_cache_dir=None,
            num_classes=2
        )
        
        is_train = (split == 'train')
        dataset_result = build_dataset(is_train, args)
        # build_dataset returns (dataset, class_weights) tuple
        if isinstance(dataset_result, tuple):
            dataset, _ = dataset_result
        else:
            dataset = dataset_result
        
        def collate_fn(batch):
            oct_images = torch.stack([sample[0] for sample in batch]).float()
            col_images = torch.stack([sample[1] for sample in batch]).float()
            clinical = torch.stack([sample[2] for sample in batch]).float()
            labels = torch.stack([sample[3] for sample in batch]).long()
            return {
                'oct_images': oct_images,
                'col_images': col_images,
                'clinical_features': clinical,
                'labels': labels
            }
    else:
        # 构建数据集参数
        dataset_args = SimpleNamespace(
            data_path=data_path,
            input_size=input_size,
            oct_num_frames=oct_num_frames,
            oct_points=12,
            oct_frames_per_point=oct_frames_per_point,
            col_num_frames=1,
            oct_cache_dir=None,
            use_pretrained_backbones=True,
            use_text_contrastive=False,
            num_classes=2
        )
        
        # 构建数据集
        dataset = build_enhanced_dataset(split, dataset_args)
        
        # 定义collate函数
        def collate_fn(batch):
            if isinstance(batch[0], dict):
                def to_tensor(x):
                    return x if torch.is_tensor(x) else torch.tensor(x)
                oct_images = torch.stack([to_tensor(sample['oct_images']) for sample in batch]).float()
                col_images = torch.stack([to_tensor(sample['col_images']) for sample in batch]).float()
                clinical = torch.stack([to_tensor(sample['clinical_features']) for sample in batch]).float()
                labels = torch.stack([to_tensor(sample['label']) for sample in batch]).long()
            else:
                oct_images = torch.stack([sample[0] for sample in batch]).float()
                col_images = torch.stack([sample[1] for sample in batch]).float()
                clinical = torch.stack([sample[2] for sample in batch]).float()
                labels = torch.stack([sample[3] for sample in batch]).long()
            return {
                'oct_images': oct_images,
                'col_images': col_images,
                'clinical_features': clinical,
                'labels': labels
            }
    
    # 创建数据加载器
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    return loader


def get_best_model_configs() -> List[Dict]:
    """
    获取最佳模型配置列表
    基于训练结果选择性能最好的模型
    使用模型集成提升性能和鲁棒性
    """
    base_path = ROOT_DIR / "paper1_hierarchical_multimodal" / "results"
    
    configs = [
        {
            'model_path': str(base_path / "cuda1_vit_backbone" / "best_model.pth"),
            'use_vit_backbone': True,
            'backbone_type': 'vit',
            'vit_model_name': 'vit_base_patch16_224',
            'embed_dim': 768,
            'num_classes': 2,
            'clinical_dim': 7,
            'contrastive_weight': 0.3,
            'input_size': 224,
            'name': 'ViT_Backbone',
            'weight': 0.4,  # 基于AUC=0.67，性能最好
            'auc': 0.672
        },
        {
            'model_path': str(base_path / "cuda0_optimized_v2" / "best_model.pth"),
            'use_vit_backbone': False,
            'embed_dim': 768,
            'num_classes': 2,
            'clinical_dim': 7,
            'contrastive_weight': 0.3,
            'name': 'ResNet_Optimized',
            'weight': 0.35,  # 基于AUC=0.64
            'auc': 0.640
        },
        {
            'model_path': str(base_path / "cuda1_run_e10_supcont" / "best_model.pth"),
            'use_vit_backbone': False,
            'embed_dim': 768,
            'num_classes': 2,
            'clinical_dim': 7,
            'contrastive_weight': 0.5,
            'name': 'ResNet_SupervisedContrastive',
            'weight': 0.25,  # 监督对比学习，提供多样性
            'auc': 0.64  # 估计值
        }
    ]
    
    # 检查模型文件是否存在
    valid_configs = []
    weights = []
    for config in configs:
        if os.path.exists(config['model_path']):
            valid_configs.append(config)
            weights.append(config.get('weight', 1.0 / len(configs)))
        else:
            print(f"⚠️  Model not found: {config['model_path']}")
    
    # 归一化权重
    if len(weights) > 0:
        total_weight = sum(weights)
        for i, config in enumerate(valid_configs):
            config['weight'] = weights[i] / total_weight
    
    print(f"\n模型配置 (共{len(valid_configs)}个):")
    for config in valid_configs:
        print(f"  - {config['name']}: {config['model_path']}")
        print(f"    AUC: {config.get('auc', 'N/A')}, Weight: {config.get('weight', 0):.3f}")
    
    return valid_configs


if __name__ == '__main__':
    # 测试模型加载
    loader = MultimodalModelLoader(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取模型配置
    configs = get_best_model_configs()
    print(f"\nFound {len(configs)} valid model configurations")
    
    # 加载模型
    if len(configs) > 0:
        model = loader.load_best_models(
            configs,
            create_ensemble=True,
            ensemble_weights=[0.4, 0.35, 0.25]  # 根据性能调整权重
        )
        print(f"\n✅ Model(s) loaded successfully")
    else:
        print("❌ No valid models found")

