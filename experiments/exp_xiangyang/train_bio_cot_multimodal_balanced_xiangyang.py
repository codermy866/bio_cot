#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT模型在襄阳多模态数据集上的训练脚本（平衡样本版本）
使用OCT + Colposcopy + Clinical数据（多模态）
重新分配训练集和验证集，使类别分布更均衡
"""

import sys
from pathlib import Path
import warnings
# 在导入torchvision之前抑制所有UserWarning
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
    print("⚠️ seaborn未安装，混淆矩阵将使用matplotlib绘制")
from datetime import datetime
import json
from PIL import Image
from typing import Dict, List, Tuple

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from src.models.bida.bio_cot_model import BioCOTModel
from src.models.bida.prior_net import build_clinical_vector
from src.utils.anti_overfitting import FocalLoss


class XiangyangMultimodalDatasetFromCSV(Dataset):
    """从CSV文件加载襄阳多模态数据集（与原版相同）"""
    
    def __init__(
        self,
        csv_path: str,
        transform=None,
        oct_num_frames: int = 60,
        max_col_images: int = 3
    ):
        """
        Args:
            csv_path: CSV文件路径（train_labels.csv, val_labels.csv, test_labels.csv）
            transform: 图像变换
            oct_num_frames: 每个样本使用的OCT帧数
            max_col_images: 每个样本使用的Colposcopy图像数（最多3张）
        """
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.oct_num_frames = oct_num_frames
        self.max_col_images = max_col_images
        
        # 读取CSV文件
        try:
            self.df = pd.read_csv(self.csv_path, encoding='utf-8')
        except:
            try:
                self.df = pd.read_csv(self.csv_path, encoding='gbk')
            except Exception as e:
                raise ValueError(f"无法读取CSV文件: {e}")
        
        print(f"✅ 从 {csv_path} 加载了 {len(self.df)} 个样本")
        print(f"   标签分布: 阴性={sum(self.df['label']==0)}, 阳性={sum(self.df['label']==1)}")
    
    def __len__(self):
        return len(self.df)
    
    def _load_oct_frames(self, oct_path_str: str) -> torch.Tensor:
        """加载OCT帧序列"""
        if pd.isna(oct_path_str) or oct_path_str == '':
            # 如果没有OCT图像，返回零张量
            return torch.zeros(self.oct_num_frames, 3, 224, 224)
        
        # 解析路径（分号分隔）
        oct_paths_raw = [p.strip() for p in str(oct_path_str).split(';') if p.strip()]
        oct_paths = []
        for p in oct_paths_raw:
            path = Path(p)
            # 如果路径不存在，尝试替换为_binary_multimodal目录
            if not path.exists():
                new_path_str = str(p).replace('襄阳按点图片分类_multimodal', '襄阳按点图片分类_binary_multimodal')
                if Path(new_path_str).exists():
                    oct_paths.append(Path(new_path_str))
                else:
                    # 尝试从文件名重建路径
                    filename = path.name
                    # 查找可能的目录
                    data_root = Path('/data2/hmy/5Center_datas/襄阳按点图片分类_binary_multimodal')
                    for category in ['低级别', '炎', '高级别', '癌']:
                        candidate = data_root / category / 'oct' / filename
                        if candidate.exists():
                            oct_paths.append(candidate)
                            break
            else:
                oct_paths.append(path)
        
        if not oct_paths:
            return torch.zeros(self.oct_num_frames, 3, 224, 224)
        
        # 均匀采样到指定帧数
        if len(oct_paths) >= self.oct_num_frames:
            indices = np.linspace(0, len(oct_paths) - 1, self.oct_num_frames, dtype=int)
            oct_paths = [oct_paths[i] for i in indices]
        else:
            # 循环填充
            while len(oct_paths) < self.oct_num_frames:
                oct_paths.extend(oct_paths)
            oct_paths = oct_paths[:self.oct_num_frames]
        
        # 加载图像
        frames = []
        for oct_path in oct_paths:
            try:
                img = Image.open(oct_path).convert('RGB')
                img = img.resize((224, 224))
                if self.transform:
                    img = self.transform(img)
                else:
                    from torchvision import transforms
                    img = transforms.ToTensor()(img)
                frames.append(img)
            except Exception as e:
                frames.append(torch.zeros(3, 224, 224))
        
        return torch.stack(frames) if frames else torch.zeros(self.oct_num_frames, 3, 224, 224)
    
    def _load_colposcopy_images(self, col_path_str: str) -> torch.Tensor:
        """加载Colposcopy图像"""
        if pd.isna(col_path_str) or col_path_str == '':
            return torch.zeros(self.max_col_images, 3, 224, 224)
        
        col_paths_raw = [p.strip() for p in str(col_path_str).split(';') if p.strip()]
        col_paths = []
        for p in col_paths_raw:
            path = Path(p)
            if not path.exists():
                new_path_str = str(p).replace('襄阳按点图片分类_multimodal', '襄阳按点图片分类_binary_multimodal')
                if Path(new_path_str).exists():
                    col_paths.append(Path(new_path_str))
            else:
                col_paths.append(path)
        
        # 限制最多max_col_images张
        col_paths = col_paths[:self.max_col_images]
        
        images = []
        for col_path in col_paths:
            try:
                img = Image.open(col_path).convert('RGB')
                img = img.resize((224, 224))
                if self.transform:
                    img = self.transform(img)
                else:
                    from torchvision import transforms
                    img = transforms.ToTensor()(img)
                images.append(img)
            except Exception as e:
                images.append(torch.zeros(3, 224, 224))
        
        # 填充到max_col_images张
        while len(images) < self.max_col_images:
            images.append(torch.zeros(3, 224, 224))
        
        return torch.stack(images[:self.max_col_images])
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 加载OCT帧序列
        oct_images = self._load_oct_frames(row['oct_path'])
        
        # 加载Colposcopy图像
        colposcopy_images = self._load_colposcopy_images(row['col_path'])
        
        # 解析临床数据
        age = float(row['age']) if pd.notna(row['age']) else 50.0
        hpv_str = str(row['hpv']).lower() if pd.notna(row['hpv']) else ''
        hpv = 1.0 if any(k in hpv_str for k in ['16', '18', 'positive', '阳性', '高危']) else 0.0
        
        tct_str = str(row['tct']).upper() if pd.notna(row['tct']) else ''
        tct_onehot = [0.0, 0.0, 0.0, 0.0, 0.0]
        if 'ASC-US' in tct_str:
            tct_onehot[0] = 1.0
        elif 'ASC-H' in tct_str:
            tct_onehot[1] = 1.0
        elif 'LSIL' in tct_str:
            tct_onehot[2] = 1.0
        elif 'HSIL' in tct_str:
            tct_onehot[3] = 1.0
        elif 'SCC' in tct_str or '癌' in tct_str:
            tct_onehot[4] = 1.0
        
        clinical_features = torch.tensor([
            age / 100.0,  # 归一化年龄
            hpv,
            *tct_onehot
        ], dtype=torch.float32)
        
        clinical_data = {
            'hpv': int(hpv),
            'tct': tct_str,
            'age': age
        }
        
        label = int(row['label'])
        oct_id = str(row['oct_id']) if 'oct_id' in row else f'sample_{idx}'
        
        return {
            'oct_images': oct_images,
            'colposcopy_images': colposcopy_images,
            'clinical_features': clinical_features,
            'clinical_data': clinical_data,
            'label': torch.tensor(label, dtype=torch.long),
            'oct_id': oct_id
        }


def create_balanced_split(data_root: str, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, random_state: int = 42):
    """
    创建平衡的训练/验证/测试集划分
    合并原有的train和val CSV，然后重新分层划分
    """
    data_root = Path(data_root)
    
    # 读取原有的CSV文件
    train_csv = data_root / 'train_labels.csv'
    val_csv = data_root / 'val_labels.csv'
    
    df_train = pd.read_csv(train_csv, encoding='utf-8')
    df_val = pd.read_csv(val_csv, encoding='utf-8')
    
    # 合并所有数据
    df_all = pd.concat([df_train, df_val], ignore_index=True)
    
    print(f"📊 总样本数: {len(df_all)}")
    print(f"   阴性样本: {sum(df_all['label']==0)} ({sum(df_all['label']==0)/len(df_all)*100:.1f}%)")
    print(f"   阳性样本: {sum(df_all['label']==1)} ({sum(df_all['label']==1)/len(df_all)*100:.1f}%)")
    
    # 分层划分
    labels = df_all['label'].values
    
    # 先分出训练集
    train_df, temp_df = train_test_split(
        df_all, 
        test_size=(1 - train_ratio), 
        stratify=labels, 
        random_state=random_state
    )
    
    # 再从剩余数据中分出验证集和测试集
    temp_labels = temp_df['label'].values
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_test_ratio),
        stratify=temp_labels,
        random_state=random_state
    )
    
    print(f"\n✅ 重新划分完成:")
    print(f"   训练集: {len(train_df)} 个样本 (阴性={sum(train_df['label']==0)}, 阳性={sum(train_df['label']==1)})")
    print(f"   验证集: {len(val_df)} 个样本 (阴性={sum(val_df['label']==0)}, 阳性={sum(val_df['label']==1)})")
    print(f"   测试集: {len(test_df)} 个样本 (阴性={sum(test_df['label']==0)}, 阳性={sum(test_df['label']==1)})")
    
    return train_df, val_df, test_df


def get_data_transforms(args, is_train: bool):
    """获取数据增强变换"""
    if is_train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def extract_features_with_resnet50(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    使用ResNet50提取图像特征
    Args:
        images: [B, F, C, H, W] 或 [B, C, H, W] 或 [B, N, C, H, W]
    Returns:
        features: [B, 2048] (ResNet50的feature维度)
    """
    from torchvision import models
    import warnings
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except:
            resnet = models.resnet50(pretrained=True)
    
    resnet = resnet.to(device)
    resnet.eval()
    
    # 移除最后的分类层和平均池化层，保留到avgpool之前
    resnet = nn.Sequential(*list(resnet.children())[:-2])  # 保留到layer4之后，输出[B, 2048, H', W']
    
    with torch.no_grad():
        if len(images.shape) == 5:  # [B, F, C, H, W] 或 [B, N, C, H, W]
            B, num_frames, C, H, W = images.shape
            images = images.view(B * num_frames, C, H, W)
            features = resnet(images)  # [B*num_frames, 2048, H', W']
            # 全局平均池化
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))  # [B*num_frames, 2048, 1, 1]
            features = features.view(B * num_frames, -1)  # [B*num_frames, 2048]
            features = features.view(B, num_frames, -1)
            features = features.mean(dim=1)  # [B, 2048]
        elif len(images.shape) == 4:  # [B, C, H, W]
            features = resnet(images)  # [B, 2048, H', W']
            # 全局平均池化
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))  # [B, 2048, 1, 1]
            features = features.view(images.size(0), -1)  # [B, 2048]
        else:
            raise ValueError(f"Unexpected image shape: {images.shape}")
    
    return features


class BioCOTMultimodalArgs:
    """训练参数配置"""
    def __init__(self):
        self.data_root = '/data2/hmy/5Center_datas/襄阳按点图片分类_binary_multimodal'
        # 自动选择GPU（优先使用显存较少的）
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    gpu_idx = int(parts[0])
                    mem_used = int(parts[1].split()[0])
                    mem_total = int(parts[2].split()[0])
                    mem_free = mem_total - mem_used
                    gpu_info.append((gpu_idx, mem_free, mem_total))
            # 选择显存最多的GPU
            gpu_info.sort(key=lambda x: x[1], reverse=True)
            self.device = f'cuda:{gpu_info[0][0]}'
            print(f"✅ 自动选择GPU: {self.device} (可用显存: {gpu_info[0][1]}MB / {gpu_info[0][2]}MB)")
        except:
            # 如果nvidia-smi不可用，默认使用cuda:0
            self.device = 'cuda:0'
            print(f"⚠️ 无法检测GPU，使用默认设备: {self.device}")
        self.batch_size = 8
        self.num_workers = 4
        self.pin_memory = True
        self.learning_rate = 1.2e-4
        self.num_epochs = 50
        self.oct_num_frames = 60
        self.max_col_images = 3
        
        # 损失权重
        self.lambda_cls = 1.0
        self.lambda_ot = 1.0
        self.lambda_consist = 0.5
        self.lambda_adv = 0.1
        
        # 输出目录
        self.output_dir = Path(__file__).parent / 'results_multimodal_balanced'
        self.checkpoint_dir = Path(__file__).parent / 'checkpoints_multimodal_balanced'
        self.log_dir = Path(__file__).parent / 'logs'
        
        # 创建目录
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # 数据划分比例
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        self.random_state = 42


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, args):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_ot_loss = 0.0
    total_consist_loss = 0.0
    total_adv_loss = 0.0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.num_epochs} [Train]')
    
    for batch_idx, batch in enumerate(pbar):
        oct_images = batch['oct_images'].to(device)  # [B, F, C, H, W]
        colposcopy_images = batch['colposcopy_images'].to(device)  # [B, N, C, H, W]
        clinical_features = batch['clinical_features'].to(device)  # [B, 7]
        clinical_data = batch['clinical_data']  # list of dict
        labels = batch['label'].to(device)  # [B]
        
        # 提取特征
        oct_features = extract_features_with_resnet50(oct_images, device)  # [B, 2048]
        colpo_features = extract_features_with_resnet50(colposcopy_images, device)  # [B, 2048]
        
        # 前向传播（启用损失组件计算）
        center_labels = torch.zeros(len(labels), dtype=torch.long, device=device)  # 假设所有样本来自同一中心
        
        # 将clinical_data列表转换为字典格式（batch级别）
        # clinical_data是list of dict，需要转换为batch级别的dict
        batch_clinical_data = {
            'hpv': [cd.get('hpv', 0) for cd in clinical_data],
            'tct': [cd.get('tct', 'NILM') for cd in clinical_data],
            'age': [cd.get('age', 50.0) for cd in clinical_data]
        }
        
        outputs = model(
            oct_features=oct_features,
            colpo_features=colpo_features,
            clinical_features=clinical_features,
            clinical_data=batch_clinical_data,  # 传递batch级别的字典
            center_labels=center_labels,  # 使用center_labels而不是center_ids
            return_loss_components=True,  # 启用损失组件计算
            use_counterfactual=True  # 启用反事实干预
        )
        
        logits = outputs['logits']
        
        # 计算分类损失
        cls_loss = criterion(logits, labels)
        
        # 获取损失组件（如果存在）
        loss_components = outputs.get('loss_components', {})
        ot_loss = loss_components.get('L_ot', torch.tensor(0.0, device=device))
        consist_loss = loss_components.get('L_consist', torch.tensor(0.0, device=device))
        adv_loss = loss_components.get('L_adv', torch.tensor(0.0, device=device))
        
        # 确保损失是标量
        if ot_loss.dim() > 0:
            ot_loss = ot_loss.mean() if ot_loss.numel() > 0 else torch.tensor(0.0, device=device)
        if consist_loss.dim() > 0:
            consist_loss = consist_loss.mean() if consist_loss.numel() > 0 else torch.tensor(0.0, device=device)
        if adv_loss.dim() > 0:
            adv_loss = adv_loss.mean() if adv_loss.numel() > 0 else torch.tensor(0.0, device=device)
        
        # 计算总损失
        loss = (
            args.lambda_cls * cls_loss +
            args.lambda_ot * ot_loss +
            args.lambda_consist * consist_loss +
            args.lambda_adv * adv_loss
        )
        
        # 构建损失字典用于统计
        loss_dict = {
            'cls_loss': cls_loss,
            'ot_loss': ot_loss,
            'consist_loss': consist_loss,
            'adv_loss': adv_loss
        }
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        total_cls_loss += loss_dict['cls_loss'].item()
        total_ot_loss += loss_dict['ot_loss'].item()
        total_consist_loss += loss_dict['consist_loss'].item()
        total_adv_loss += loss_dict['adv_loss'].item()
        
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())
        
        # 更新进度条
        acc = accuracy_score(all_labels, all_preds)
        pos_pred = sum(all_preds)
        pos_prob = np.mean(all_probs) if all_probs else 0.0
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{loss_dict["cls_loss"].item():.4f}',
            'ot': f'{loss_dict["ot_loss"].item():.4f}',
            'acc': f'{acc:.4f}',
            'pos_pred': f'{pos_pred}/{len(all_labels)}',
            'pos_prob': f'{pos_prob:.3f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_ot_loss = total_ot_loss / len(dataloader)
    avg_consist_loss = total_consist_loss / len(dataloader)
    avg_adv_loss = total_adv_loss / len(dataloader)
    
    acc = accuracy_score(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'cls_loss': avg_cls_loss,
        'ot_loss': avg_ot_loss,
        'consist_loss': avg_consist_loss,
        'adv_loss': avg_adv_loss,
        'acc': acc,
        'preds': all_preds,
        'labels': all_labels,
        'probs': all_probs
    }


def validate(model, dataloader, criterion, device, epoch, args):
    """验证"""
    model.eval()
    total_loss = 0.0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.num_epochs} [Val]')
        
        for batch in pbar:
            oct_images = batch['oct_images'].to(device)
            colposcopy_images = batch['colposcopy_images'].to(device)
            clinical_features = batch['clinical_features'].to(device)
            clinical_data = batch['clinical_data']
            labels = batch['label'].to(device)
            
            # 提取特征
            oct_features = extract_features_with_resnet50(oct_images, device)  # [B, 2048]
            colpo_features = extract_features_with_resnet50(colposcopy_images, device)  # [B, 2048]
            
            # 前向传播（验证时不需要计算损失组件）
            center_labels = torch.zeros(len(labels), dtype=torch.long, device=device)
            
            # 将clinical_data列表转换为字典格式（batch级别）
            batch_clinical_data = {
                'hpv': [cd.get('hpv', 0) for cd in clinical_data],
                'tct': [cd.get('tct', 'NILM') for cd in clinical_data],
                'age': [cd.get('age', 50.0) for cd in clinical_data]
            }
            
            outputs = model(
                oct_features=oct_features,
                colpo_features=colpo_features,
                clinical_features=clinical_features,
                clinical_data=batch_clinical_data,
                center_labels=center_labels,
                return_loss_components=False,  # 验证时不需要
                use_counterfactual=False  # 验证时不需要
            )
            
            logits = outputs['logits']
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
            
            acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    
    # 计算AUC和F1
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    f1 = f1_score(all_labels, all_preds)
    
    # 确保返回的是numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    return {
        'loss': avg_loss,
        'acc': acc,
        'auc': auc,
        'f1': f1,
        'preds': all_preds,
        'labels': all_labels,
        'probs': all_probs
    }


def plot_training_curves(history, output_dir, timestamp):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 训练和验证损失
    axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 训练和验证准确率
    axes[0, 1].plot(epochs, history['train_acc'], label='Train Acc', color='blue')
    axes[0, 1].plot(epochs, history['val_acc'], label='Val Acc', color='red')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # AUC
    axes[0, 2].plot(epochs, history['val_auc'], label='Val AUC', color='green')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('AUC')
    axes[0, 2].set_title('Validation AUC')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # 损失组件
    axes[1, 0].plot(history['train_ot_loss'], label='OT Loss')
    axes[1, 0].plot(history['train_consist_loss'], label='Consist Loss')
    axes[1, 0].plot(history['train_adv_loss'], label='Adv Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Loss Components')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1 Score
    axes[1, 1].plot(history['val_f1'], label='Val F1', color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Validation F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 清空最后一个子图
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_bio_cot_multimodal_balanced_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 训练曲线已保存")


def plot_confusion_matrix(cm, output_dir, timestamp):
    """绘制混淆矩阵（带热图）"""
    plt.figure(figsize=(10, 8))
    
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    if sns is not None:
        # 使用seaborn绘制热图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'],
                    cbar_kws={'label': 'Count'})
        # 添加百分比标注
        for i in range(2):
            for j in range(2):
                if cm[i, j] > 0:
                    plt.text(j+0.5, i+0.7, f'({cm_percent[i, j]:.1f}%)', 
                            ha='center', va='center', fontsize=10, color='red', weight='bold')
    else:
        # 使用matplotlib绘制混淆矩阵
        im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, label='Count')
        plt.xticks([0, 1], ['Negative', 'Positive'])
        plt.yticks([0, 1], ['Negative', 'Positive'])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)', 
                        ha='center', va='center', color='black', fontsize=14, weight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix (with Percentages)', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_bio_cot_multimodal_balanced_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存")


def plot_feature_heatmap(model, dataloader, device, output_dir, timestamp, num_samples=20):
    """绘制特征热图（z_causal, z_noise, z_sem的相似度热图）"""
    model.eval()
    
    all_z_causal = []
    all_z_noise = []
    all_z_sem = []
    all_labels = []
    
    with torch.no_grad():
        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            
            oct_images = batch['oct_images'].to(device)
            colposcopy_images = batch['colposcopy_images'].to(device)
            clinical_features = batch['clinical_features'].to(device)
            clinical_data = batch['clinical_data']
            labels = batch['label'].to(device)
            
            # 提取特征
            oct_features = extract_features_with_resnet50(oct_images, device)
            colpo_features = extract_features_with_resnet50(colposcopy_images, device)
            
            # 构建临床向量
            batch_clinical_data = {
                'hpv': [cd.get('hpv', 0) for cd in clinical_data],
                'tct': [cd.get('tct', 'NILM') for cd in clinical_data],
                'age': [cd.get('age', 50.0) for cd in clinical_data]
            }
            
            center_labels = torch.zeros(len(labels), dtype=torch.long, device=device)
            outputs = model(
                oct_features=oct_features,
                colpo_features=colpo_features,
                clinical_features=clinical_features,
                clinical_data=batch_clinical_data,
                center_labels=center_labels,
                return_loss_components=False,
                use_counterfactual=False
            )
            
            all_z_causal.append(outputs['z_causal'].cpu())
            all_z_noise.append(outputs['z_noise'].cpu())
            all_z_sem.append(outputs['z_sem'].cpu())
            all_labels.append(labels.cpu())
            
            count += len(labels)
            if count >= num_samples:
                break
    
    # 拼接所有特征
    z_causal = torch.cat(all_z_causal, dim=0)[:num_samples]  # [N, 768]
    z_noise = torch.cat(all_z_noise, dim=0)[:num_samples]  # [N, 768]
    z_sem = torch.cat(all_z_sem, dim=0)[:num_samples]  # [N, 768]
    labels = torch.cat(all_labels, dim=0)[:num_samples]  # [N]
    
    # 计算相似度矩阵（余弦相似度）
    def cosine_similarity_matrix(x):
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        return torch.mm(x_norm, x_norm.t())
    
    sim_causal = cosine_similarity_matrix(z_causal).numpy()
    sim_noise = cosine_similarity_matrix(z_noise).numpy()
    sim_sem = cosine_similarity_matrix(z_sem).numpy()
    
    # 绘制热图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # z_causal相似度热图
    im1 = axes[0].imshow(sim_causal, cmap='viridis', aspect='auto', vmin=-1, vmax=1)
    axes[0].set_title('z_causal Similarity Heatmap', fontsize=12, weight='bold')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Sample Index')
    plt.colorbar(im1, ax=axes[0], label='Cosine Similarity')
    
    # z_noise相似度热图
    im2 = axes[1].imshow(sim_noise, cmap='plasma', aspect='auto', vmin=-1, vmax=1)
    axes[1].set_title('z_noise Similarity Heatmap', fontsize=12, weight='bold')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Sample Index')
    plt.colorbar(im2, ax=axes[1], label='Cosine Similarity')
    
    # z_sem相似度热图
    im3 = axes[2].imshow(sim_sem, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[2].set_title('z_sem Similarity Heatmap', fontsize=12, weight='bold')
    axes[2].set_xlabel('Sample Index')
    axes[2].set_ylabel('Sample Index')
    plt.colorbar(im3, ax=axes[2], label='Cosine Similarity')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'feature_heatmap_bio_cot_multimodal_balanced_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 特征热图已保存")


def plot_loss_heatmap(history, output_dir, timestamp):
    """绘制损失组件热图（按epoch）"""
    # 准备数据
    epochs = range(1, len(history['train_loss']) + 1)
    loss_data = np.array([
        history['train_cls_loss'],
        history['train_ot_loss'],
        history['train_consist_loss'],
        history['train_adv_loss']
    ])
    
    # 归一化（按行归一化，便于比较）
    loss_data_norm = (loss_data - loss_data.min(axis=1, keepdims=True)) / (loss_data.max(axis=1, keepdims=True) - loss_data.min(axis=1, keepdims=True) + 1e-10)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 原始损失热图
    im1 = axes[0].imshow(loss_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    axes[0].set_yticks(range(4))
    axes[0].set_yticklabels(['CLS Loss', 'OT Loss', 'Consist Loss', 'Adv Loss'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss Component')
    axes[0].set_title('Loss Components Heatmap (Raw)', fontsize=12, weight='bold')
    plt.colorbar(im1, ax=axes[0], label='Loss Value')
    
    # 归一化损失热图
    im2 = axes[1].imshow(loss_data_norm, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    axes[1].set_yticks(range(4))
    axes[1].set_yticklabels(['CLS Loss', 'OT Loss', 'Consist Loss', 'Adv Loss'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss Component')
    axes[1].set_title('Loss Components Heatmap (Normalized)', fontsize=12, weight='bold')
    plt.colorbar(im2, ax=axes[1], label='Normalized Loss')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'loss_heatmap_bio_cot_multimodal_balanced_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 损失热图已保存")


def plot_roc_curve(y_true, y_probs, output_dir, timestamp):
    """绘制ROC曲线"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, weight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'roc_curve_bio_cot_multimodal_balanced_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC曲线已保存")


def plot_prediction_distribution(y_true, y_probs, output_dir, timestamp):
    """绘制预测概率分布（直方图、箱线图、小提琴图、热图等专业可视化）"""
    # 确保输入是numpy数组
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_probs, np.ndarray):
        y_probs = np.array(y_probs)
    
    # 确保是一维数组
    y_true = y_true.flatten()
    y_probs = y_probs.flatten()
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 按真实标签分组
    neg_probs = y_probs[y_true == 0]
    pos_probs = y_probs[y_true == 1]
    
    # 确保数据不为空且是数组
    if len(neg_probs) == 0:
        neg_probs = np.array([0.0])
    else:
        neg_probs = np.array(neg_probs).flatten()
    
    if len(pos_probs) == 0:
        pos_probs = np.array([0.0])
    else:
        pos_probs = np.array(pos_probs).flatten()
    
    # 1. 直方图（重叠）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(neg_probs, bins=20, alpha=0.7, label='Negative', color='#3498db', edgecolor='black', linewidth=0.5)
    ax1.hist(pos_probs, bins=20, alpha=0.7, label='Positive', color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, weight='bold')
    ax1.set_title('(A) Histogram Overlay', fontsize=12, weight='bold', pad=10)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0, 1])
    
    # 2. 箱线图（修复维度问题）
    ax2 = fig.add_subplot(gs[0, 1])
    box_data = []
    labels = []
    if len(neg_probs) > 0:
        box_data.append(neg_probs)
        labels.append('Negative')
    if len(pos_probs) > 0:
        box_data.append(pos_probs)
        labels.append('Positive')
    
    if len(box_data) > 0:
        bp = ax2.boxplot(box_data, labels=labels, patch_artist=True, 
                        widths=0.6, showmeans=True, meanline=True)
        colors = ['#3498db', '#e74c3c']
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)
    ax2.set_ylabel('Predicted Probability', fontsize=11, weight='bold')
    ax2.set_title('(B) Box Plot', fontsize=12, weight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_ylim([0, 1])
    
    # 3. 小提琴图（修复数据格式）
    ax3 = fig.add_subplot(gs[0, 2])
    try:
        import seaborn as sns
        import pandas as pd
        # 准备DataFrame格式的数据
        df_data = []
        for prob in neg_probs:
            df_data.append({'Probability': prob, 'Class': 'Negative'})
        for prob in pos_probs:
            df_data.append({'Probability': prob, 'Class': 'Positive'})
        df = pd.DataFrame(df_data)
        
        sns.violinplot(data=df, x='Class', y='Probability', ax=ax3, 
                      palette=['#3498db', '#e74c3c'], inner='box', width=0.8)
        ax3.set_xlabel('True Class', fontsize=11, weight='bold')
        ax3.set_ylabel('Predicted Probability', fontsize=11, weight='bold')
        ax3.set_title('(C) Violin Plot', fontsize=12, weight='bold', pad=10)
        ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax3.set_ylim([0, 1])
    except Exception as e:
        # 如果seaborn不可用，使用KDE密度图
        from scipy import stats
        try:
            if len(neg_probs) > 1:
                kde_neg = stats.gaussian_kde(neg_probs)
                x_neg = np.linspace(0, 1, 100)
                ax3.plot(x_neg, kde_neg(x_neg), label='Negative', color='#3498db', linewidth=2)
            if len(pos_probs) > 1:
                kde_pos = stats.gaussian_kde(pos_probs)
                x_pos = np.linspace(0, 1, 100)
                ax3.plot(x_pos, kde_pos(x_pos), label='Positive', color='#e74c3c', linewidth=2)
            ax3.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
            ax3.set_ylabel('Density', fontsize=11, weight='bold')
            ax3.set_title('(C) Kernel Density Estimation', fontsize=12, weight='bold', pad=10)
            ax3.legend(frameon=True, fancybox=True, shadow=True)
        except:
            ax3.hist(neg_probs, bins=20, alpha=0.5, label='Negative', color='#3498db', density=True)
            ax3.hist(pos_probs, bins=20, alpha=0.5, label='Positive', color='#e74c3c', density=True)
            ax3.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
            ax3.set_ylabel('Density', fontsize=11, weight='bold')
            ax3.set_title('(C) Density Histogram', fontsize=12, weight='bold', pad=10)
            ax3.legend(frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 4. 2D热图（真实标签 vs 预测概率）
    ax4 = fig.add_subplot(gs[1, 0])
    prob_bins = np.linspace(0, 1, 21)
    hist_2d, xedges, yedges = np.histogram2d(y_probs, y_true, bins=[prob_bins, [0, 0.5, 1]])
    im = ax4.imshow(hist_2d.T, cmap='YlOrRd', aspect='auto', origin='lower', 
                    extent=[0, 1, 0, 1], interpolation='bilinear')
    ax4.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
    ax4.set_ylabel('True Label', fontsize=11, weight='bold')
    ax4.set_yticks([0.25, 0.75])
    ax4.set_yticklabels(['Negative (0)', 'Positive (1)'])
    ax4.set_title('(D) Probability vs Label Heatmap', fontsize=12, weight='bold', pad=10)
    cbar = plt.colorbar(im, ax=ax4, label='Count')
    cbar.ax.tick_params(labelsize=9)
    
    # 5. 累积分布函数（CDF）
    ax5 = fig.add_subplot(gs[1, 1])
    if len(neg_probs) > 0:
        sorted_neg = np.sort(neg_probs)
        ax5.plot(sorted_neg, np.arange(len(sorted_neg)) / max(len(sorted_neg), 1), 
                 label='Negative', color='#3498db', linewidth=2.5, linestyle='-', marker='o', markersize=3, alpha=0.8)
    if len(pos_probs) > 0:
        sorted_pos = np.sort(pos_probs)
        ax5.plot(sorted_pos, np.arange(len(sorted_pos)) / max(len(sorted_pos), 1), 
                 label='Positive', color='#e74c3c', linewidth=2.5, linestyle='-', marker='s', markersize=3, alpha=0.8)
    ax5.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
    ax5.set_ylabel('Cumulative Probability', fontsize=11, weight='bold')
    ax5.set_title('(E) Cumulative Distribution Function', fontsize=12, weight='bold', pad=10)
    ax5.legend(frameon=True, fancybox=True, shadow=True)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1])
    
    # 6. 分位数-分位数图（Q-Q Plot）
    ax6 = fig.add_subplot(gs[1, 2])
    try:
        from scipy import stats
        if len(neg_probs) > 1 and len(pos_probs) > 1:
            stats.probplot(neg_probs, dist="norm", plot=ax6)
            stats.probplot(pos_probs, dist="norm", plot=ax6)
            ax6.get_lines()[0].set_color('#3498db')
            ax6.get_lines()[0].set_label('Negative')
            ax6.get_lines()[1].set_color('#e74c3c')
            ax6.get_lines()[1].set_label('Positive')
            ax6.set_title('(F) Q-Q Plot', fontsize=12, weight='bold', pad=10)
            ax6.legend(frameon=True, fancybox=True, shadow=True)
        else:
            ax6.text(0.5, 0.5, 'Insufficient data\nfor Q-Q plot', 
                    ha='center', va='center', fontsize=12, transform=ax6.transAxes)
            ax6.set_title('(F) Q-Q Plot', fontsize=12, weight='bold', pad=10)
    except:
        ax6.text(0.5, 0.5, 'Q-Q plot\nnot available', 
                ha='center', va='center', fontsize=12, transform=ax6.transAxes)
        ax6.set_title('(F) Q-Q Plot', fontsize=12, weight='bold', pad=10)
    ax6.grid(True, alpha=0.3, linestyle='--')
    
    # 7. 统计信息表格（增强版）
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.axis('off')
    from scipy import stats
    try:
        neg_mean, neg_std = np.mean(neg_probs), np.std(neg_probs)
        pos_mean, pos_std = np.mean(pos_probs), np.std(pos_probs)
        neg_median = np.median(neg_probs)
        pos_median = np.median(pos_probs)
        neg_q25, neg_q75 = np.percentile(neg_probs, [25, 75])
        pos_q25, pos_q75 = np.percentile(pos_probs, [25, 75])
        
        stats_text = f"""
        Statistical Summary:
        
        Negative Class (n={len(neg_probs)}):
          Mean ± SD: {neg_mean:.4f} ± {neg_std:.4f}
          Median [IQR]: {neg_median:.4f} [{neg_q25:.4f}, {neg_q75:.4f}]
          Range: [{np.min(neg_probs):.4f}, {np.max(neg_probs):.4f}]
        
        Positive Class (n={len(pos_probs)}):
          Mean ± SD: {pos_mean:.4f} ± {pos_std:.4f}
          Median [IQR]: {pos_median:.4f} [{pos_q25:.4f}, {pos_q75:.4f}]
          Range: [{np.min(pos_probs):.4f}, {np.max(pos_probs):.4f}]
        """
    except:
        stats_text = "Statistics calculation error"
    
    ax7.text(0.1, 0.5, stats_text, fontsize=9.5, family='monospace', 
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='#f8f9fa', 
             alpha=0.9, edgecolor='#dee2e6', linewidth=1.5))
    ax7.set_title('(G) Statistical Summary', fontsize=12, weight='bold', pad=10)
    
    # 8. 密度对比图（使用seaborn）
    ax8 = fig.add_subplot(gs[2, 1])
    try:
        import seaborn as sns
        import pandas as pd
        df_data = []
        for prob in neg_probs:
            df_data.append({'Probability': prob, 'Class': 'Negative'})
        for prob in pos_probs:
            df_data.append({'Probability': prob, 'Class': 'Positive'})
        df = pd.DataFrame(df_data)
        
        sns.kdeplot(data=df, x='Probability', hue='Class', ax=ax8, 
                   palette=['#3498db', '#e74c3c'], fill=True, alpha=0.6, linewidth=2)
        ax8.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
        ax8.set_ylabel('Density', fontsize=11, weight='bold')
        ax8.set_title('(H) Density Comparison', fontsize=12, weight='bold', pad=10)
        ax8.legend(frameon=True, fancybox=True, shadow=True)
        ax8.set_xlim([0, 1])
    except:
        ax8.hist(neg_probs, bins=20, alpha=0.5, label='Negative', color='#3498db', density=True)
        ax8.hist(pos_probs, bins=20, alpha=0.5, label='Positive', color='#e74c3c', density=True)
        ax8.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
        ax8.set_ylabel('Density', fontsize=11, weight='bold')
        ax8.set_title('(H) Density Histogram', fontsize=12, weight='bold', pad=10)
        ax8.legend(frameon=True, fancybox=True, shadow=True)
    ax8.grid(True, alpha=0.3, linestyle='--')
    
    # 9. 分面直方图（Side-by-side）
    ax9 = fig.add_subplot(gs[2, 2])
    try:
        import seaborn as sns
        import pandas as pd
        df_data = []
        for prob in neg_probs:
            df_data.append({'Probability': prob, 'Class': 'Negative'})
        for prob in pos_probs:
            df_data.append({'Probability': prob, 'Class': 'Positive'})
        df = pd.DataFrame(df_data)
        
        sns.histplot(data=df, x='Probability', hue='Class', bins=20, 
                    ax=ax9, palette=['#3498db', '#e74c3c'], alpha=0.7, 
                    stat='density', kde=True, line_kws={'linewidth': 2})
        ax9.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
        ax9.set_ylabel('Density', fontsize=11, weight='bold')
        ax9.set_title('(I) Histogram with KDE', fontsize=12, weight='bold', pad=10)
        ax9.legend(frameon=True, fancybox=True, shadow=True)
        ax9.set_xlim([0, 1])
    except:
        ax9.hist([neg_probs, pos_probs], bins=20, alpha=0.7, 
                label=['Negative', 'Positive'], color=['#3498db', '#e74c3c'], 
                edgecolor='black', linewidth=0.5)
        ax9.set_xlabel('Predicted Probability', fontsize=11, weight='bold')
        ax9.set_ylabel('Frequency', fontsize=11, weight='bold')
        ax9.set_title('(I) Side-by-side Histogram', fontsize=12, weight='bold', pad=10)
        ax9.legend(frameon=True, fancybox=True, shadow=True)
    ax9.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Comprehensive Prediction Distribution Analysis', fontsize=16, weight='bold', y=0.995)
    plt.savefig(output_dir / f'prediction_distribution_bio_cot_multimodal_balanced_{timestamp}.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 预测分布分析图已保存: prediction_distribution_bio_cot_multimodal_balanced_{timestamp}.png")


def plot_loss_boxplot(history, output_dir, timestamp):
    """绘制损失组件的箱线图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准备数据
    loss_data = {
        'CLS Loss': history['train_cls_loss'],
        'OT Loss': history['train_ot_loss'],
        'Consist Loss': history['train_consist_loss'],
        'Adv Loss': history['train_adv_loss']
    }
    
    # 箱线图
    ax1 = axes[0]
    box_data = [loss_data[key] for key in loss_data.keys()]
    bp = ax1.boxplot(box_data, labels=list(loss_data.keys()), patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax1.set_ylabel('Loss Value', fontsize=11)
    ax1.set_title('Loss Components Distribution (Boxplot)', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 小提琴图（如果seaborn可用）
    ax2 = axes[1]
    try:
        import seaborn as sns
        data_list = []
        labels_list = []
        for key, values in loss_data.items():
            data_list.extend(values)
            labels_list.extend([key] * len(values))
        df = pd.DataFrame({'Loss': data_list, 'Component': labels_list})
        sns.violinplot(data=df, x='Component', y='Loss', ax=ax2, palette=colors)
        ax2.set_title('Loss Components Distribution (Violin Plot)', fontsize=12, weight='bold')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    except:
        # 如果seaborn不可用，使用分组箱线图
        positions = [1, 2, 3, 4]
        bp2 = ax2.boxplot(box_data, positions=positions, labels=list(loss_data.keys()), patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
        ax2.set_ylabel('Loss Value', fontsize=11)
        ax2.set_title('Loss Components Distribution (Grouped Boxplot)', fontsize=12, weight='bold')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'loss_boxplot_bio_cot_multimodal_balanced_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 损失箱线图已保存")


def plot_advanced_violin_analysis(y_true, y_probs, output_dir, timestamp):
    """绘制高级小提琴图分析（多子图展示）"""
    # 确保输入是numpy数组
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_probs, np.ndarray):
        y_probs = np.array(y_probs)
    
    # 确保是一维数组
    y_true = y_true.flatten()
    y_probs = y_probs.flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Advanced Violin Plot Analysis', fontsize=16, weight='bold', y=0.995)
    
    neg_probs = y_probs[y_true == 0]
    pos_probs = y_probs[y_true == 1]
    
    # 确保数据不为空且是数组
    if len(neg_probs) == 0:
        neg_probs = np.array([0.0])
    else:
        neg_probs = np.array(neg_probs).flatten()
    
    if len(pos_probs) == 0:
        pos_probs = np.array([0.0])
    else:
        pos_probs = np.array(pos_probs).flatten()
    
    try:
        import seaborn as sns
        import pandas as pd
        
        # 准备数据
        df_data = []
        for prob in neg_probs:
            df_data.append({'Probability': prob, 'Class': 'Negative'})
        for prob in pos_probs:
            df_data.append({'Probability': prob, 'Class': 'Positive'})
        df = pd.DataFrame(df_data)
        
        # 1. 标准小提琴图
        ax1 = axes[0, 0]
        sns.violinplot(data=df, x='Class', y='Probability', ax=ax1, 
                      palette=['#3498db', '#e74c3c'], inner='box', width=0.8)
        ax1.set_title('(A) Standard Violin Plot', fontsize=12, weight='bold', pad=10)
        ax1.set_ylabel('Predicted Probability', fontsize=11, weight='bold')
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.set_ylim([0, 1])
        
        # 2. 小提琴图 + 散点图
        ax2 = axes[0, 1]
        sns.violinplot(data=df, x='Class', y='Probability', ax=ax2, 
                      palette=['#3498db', '#e74c3c'], inner=None, width=0.8)
        sns.stripplot(data=df, x='Class', y='Probability', ax=ax2, 
                     color='black', size=3, alpha=0.3, jitter=True)
        ax2.set_title('(B) Violin + Strip Plot', fontsize=12, weight='bold', pad=10)
        ax2.set_ylabel('Predicted Probability', fontsize=11, weight='bold')
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.set_ylim([0, 1])
        
        # 3. 小提琴图 + 箱线图
        ax3 = axes[1, 0]
        sns.violinplot(data=df, x='Class', y='Probability', ax=ax3, 
                      palette=['#3498db', '#e74c3c'], inner='box', width=0.8)
        ax3.set_title('(C) Violin + Box Plot', fontsize=12, weight='bold', pad=10)
        ax3.set_ylabel('Predicted Probability', fontsize=11, weight='bold')
        ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax3.set_ylim([0, 1])
        
        # 4. 小提琴图 + 均值点
        ax4 = axes[1, 1]
        sns.violinplot(data=df, x='Class', y='Probability', ax=ax4, 
                      palette=['#3498db', '#e74c3c'], inner='quart', width=0.8)
        # 添加均值点
        neg_mean = np.mean(neg_probs)
        pos_mean = np.mean(pos_probs)
        ax4.scatter([0], [neg_mean], color='yellow', s=100, marker='D', 
                   edgecolors='black', linewidths=1.5, zorder=10, label='Mean')
        ax4.scatter([1], [pos_mean], color='yellow', s=100, marker='D', 
                   edgecolors='black', linewidths=1.5, zorder=10)
        ax4.set_title('(D) Violin + Quartiles + Mean', fontsize=12, weight='bold', pad=10)
        ax4.set_ylabel('Predicted Probability', fontsize=11, weight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax4.set_ylim([0, 1])
        
    except ImportError:
        # 如果没有seaborn，使用matplotlib绘制简化版本
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'Seaborn required\nfor violin plots', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'violin_analysis_bio_cot_multimodal_balanced_{timestamp}.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 小提琴图分析已保存: violin_analysis_bio_cot_multimodal_balanced_{timestamp}.png")


def plot_loss_component_analysis(history, output_dir, timestamp):
    """绘制损失组件的详细分析（多维度可视化）"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    epochs = np.arange(1, len(history['train_loss']) + 1)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    # 1. 损失组件趋势（对数尺度）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(epochs, history['train_cls_loss'], label='CLS Loss', color=colors[0], linewidth=2)
    ax1.semilogy(epochs, history['train_ot_loss'], label='OT Loss', color=colors[1], linewidth=2)
    ax1.semilogy(epochs, history['train_consist_loss'], label='Consist Loss', color=colors[2], linewidth=2)
    ax1.semilogy(epochs, history['train_adv_loss'], label='Adv Loss', color=colors[3], linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11, weight='bold')
    ax1.set_ylabel('Loss (Log Scale)', fontsize=11, weight='bold')
    ax1.set_title('(A) Loss Components (Log Scale)', fontsize=12, weight='bold', pad=10)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. 损失组件箱线图
    ax2 = fig.add_subplot(gs[0, 1])
    loss_data = [
        history['train_cls_loss'],
        history['train_ot_loss'],
        history['train_consist_loss'],
        history['train_adv_loss']
    ]
    bp = ax2.boxplot(loss_data, labels=['CLS', 'OT', 'Consist', 'Adv'], patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.7)
    ax2.set_ylabel('Loss Value', fontsize=11, weight='bold')
    ax2.set_title('(B) Loss Components Distribution', fontsize=12, weight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 3. 损失组件小提琴图
    ax3 = fig.add_subplot(gs[0, 2])
    try:
        import seaborn as sns
        import pandas as pd
        loss_df = pd.DataFrame({
            'CLS': history['train_cls_loss'],
            'OT': history['train_ot_loss'],
            'Consist': history['train_consist_loss'],
            'Adv': history['train_adv_loss']
        })
        loss_df_melted = loss_df.melt(var_name='Loss Type', value_name='Loss Value')
        sns.violinplot(data=loss_df_melted, x='Loss Type', y='Loss Value', ax=ax3, 
                      palette=colors, inner='box')
        ax3.set_title('(C) Loss Components Violin Plot', fontsize=12, weight='bold', pad=10)
        ax3.set_ylabel('Loss Value', fontsize=11, weight='bold')
        ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    except:
        ax3.text(0.5, 0.5, 'Seaborn required\nfor violin plot', 
                ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax3.set_title('(C) Loss Components Violin Plot', fontsize=12, weight='bold', pad=10)
    
    # 4. 损失组件相关性热图
    ax4 = fig.add_subplot(gs[1, 0])
    try:
        import seaborn as sns
        import pandas as pd
        loss_df = pd.DataFrame({
            'CLS': history['train_cls_loss'],
            'OT': history['train_ot_loss'],
            'Consist': history['train_consist_loss'],
            'Adv': history['train_adv_loss']
        })
        corr = loss_df.corr()
        sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
                   ax=ax4, square=True, linewidths=1, cbar_kws={'label': 'Correlation'})
        ax4.set_title('(D) Loss Components Correlation', fontsize=12, weight='bold', pad=10)
    except:
        ax4.text(0.5, 0.5, 'Seaborn required\nfor heatmap', 
                ha='center', va='center', fontsize=12, transform=ax4.transAxes)
        ax4.set_title('(D) Loss Components Correlation', fontsize=12, weight='bold', pad=10)
    
    # 5. 损失组件占比（饼图）
    ax5 = fig.add_subplot(gs[1, 1])
    total_loss = (np.mean(history['train_cls_loss']) + 
                  np.mean(history['train_ot_loss']) + 
                  np.mean(history['train_consist_loss']) + 
                  np.mean(history['train_adv_loss']))
    if total_loss > 0:
        sizes = [
            np.mean(history['train_cls_loss']) / total_loss,
            np.mean(history['train_ot_loss']) / total_loss,
            np.mean(history['train_consist_loss']) / total_loss,
            np.mean(history['train_adv_loss']) / total_loss
        ]
        labels = ['CLS', 'OT', 'Consist', 'Adv']
        ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
               textprops={'fontsize': 11, 'weight': 'bold'})
    ax5.set_title('(E) Average Loss Components Ratio', fontsize=12, weight='bold', pad=10)
    
    # 6. 损失组件累积贡献
    ax6 = fig.add_subplot(gs[1, 2])
    cumulative_cls = np.cumsum(history['train_cls_loss'])
    cumulative_ot = np.cumsum(history['train_ot_loss'])
    cumulative_consist = np.cumsum(history['train_consist_loss'])
    cumulative_adv = np.cumsum(history['train_adv_loss'])
    ax6.fill_between(epochs, 0, cumulative_cls, alpha=0.6, color=colors[0], label='CLS')
    ax6.fill_between(epochs, cumulative_cls, cumulative_cls + cumulative_ot, alpha=0.6, color=colors[1], label='OT')
    ax6.fill_between(epochs, cumulative_cls + cumulative_ot, 
                     cumulative_cls + cumulative_ot + cumulative_consist, 
                     alpha=0.6, color=colors[2], label='Consist')
    ax6.fill_between(epochs, cumulative_cls + cumulative_ot + cumulative_consist,
                     cumulative_cls + cumulative_ot + cumulative_consist + cumulative_adv,
                     alpha=0.6, color=colors[3], label='Adv')
    ax6.set_xlabel('Epoch', fontsize=11, weight='bold')
    ax6.set_ylabel('Cumulative Loss', fontsize=11, weight='bold')
    ax6.set_title('(F) Cumulative Loss Components', fontsize=12, weight='bold', pad=10)
    ax6.legend(frameon=True, fancybox=True, shadow=True)
    ax6.grid(True, alpha=0.3, linestyle='--')
    
    # 7. 损失组件与性能指标的关系
    ax7 = fig.add_subplot(gs[2, 0])
    ax7_twin = ax7.twinx()
    line1 = ax7.plot(epochs, history['train_cls_loss'], 'o-', color=colors[0], label='CLS Loss', linewidth=2, markersize=4)
    line2 = ax7_twin.plot(epochs, history['val_acc'], 's-', color='#9b59b6', label='Val Acc', linewidth=2, markersize=4)
    ax7.set_xlabel('Epoch', fontsize=11, weight='bold')
    ax7.set_ylabel('CLS Loss', fontsize=11, weight='bold', color=colors[0])
    ax7_twin.set_ylabel('Validation Accuracy', fontsize=11, weight='bold', color='#9b59b6')
    ax7.tick_params(axis='y', labelcolor=colors[0])
    ax7_twin.tick_params(axis='y', labelcolor='#9b59b6')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax7.legend(lines, labels, loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax7.set_title('(G) CLS Loss vs Validation Accuracy', fontsize=12, weight='bold', pad=10)
    ax7.grid(True, alpha=0.3, linestyle='--')
    
    # 8. 损失组件散点矩阵
    ax8 = fig.add_subplot(gs[2, 1])
    scatter = ax8.scatter(history['train_cls_loss'], history['train_ot_loss'], 
               alpha=0.6, s=50, c=epochs, cmap='viridis', edgecolors='black', linewidths=0.5)
    ax8.set_xlabel('CLS Loss', fontsize=11, weight='bold')
    ax8.set_ylabel('OT Loss', fontsize=11, weight='bold')
    ax8.set_title('(H) CLS vs OT Loss Scatter', fontsize=12, weight='bold', pad=10)
    cbar = plt.colorbar(scatter, ax=ax8)
    cbar.set_label('Epoch', fontsize=9)
    ax8.grid(True, alpha=0.3, linestyle='--')
    
    # 9. 损失组件统计摘要
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    stats_text = f"""
    Loss Components Statistics:
    
    CLS Loss:
      Mean: {np.mean(history['train_cls_loss']):.6f}
      Std: {np.std(history['train_cls_loss']):.6f}
      Min: {np.min(history['train_cls_loss']):.6f}
      Max: {np.max(history['train_cls_loss']):.6f}
    
    OT Loss:
      Mean: {np.mean(history['train_ot_loss']):.6f}
      Std: {np.std(history['train_ot_loss']):.6f}
      Min: {np.min(history['train_ot_loss']):.6f}
      Max: {np.max(history['train_ot_loss']):.6f}
    
    Consist Loss:
      Mean: {np.mean(history['train_consist_loss']):.6f}
      Std: {np.std(history['train_consist_loss']):.6f}
      Min: {np.min(history['train_consist_loss']):.6f}
      Max: {np.max(history['train_consist_loss']):.6f}
    
    Adv Loss:
      Mean: {np.mean(history['train_adv_loss']):.6f}
      Std: {np.std(history['train_adv_loss']):.6f}
      Min: {np.min(history['train_adv_loss']):.6f}
      Max: {np.max(history['train_adv_loss']):.6f}
    """
    ax9.text(0.1, 0.5, stats_text, fontsize=9, family='monospace', 
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='#f8f9fa', 
             alpha=0.9, edgecolor='#dee2e6', linewidth=1.5))
    ax9.set_title('(I) Statistical Summary', fontsize=12, weight='bold', pad=10)
    
    plt.suptitle('Comprehensive Loss Component Analysis', fontsize=16, weight='bold', y=0.995)
    plt.savefig(output_dir / f'loss_component_analysis_bio_cot_multimodal_balanced_{timestamp}.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 损失组件分析图已保存: loss_component_analysis_bio_cot_multimodal_balanced_{timestamp}.png")


def plot_metrics_comparison(history, output_dir, timestamp):
    """绘制指标对比图（箱线图、热图）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. 训练指标箱线图
    ax1 = axes[0, 0]
    train_metrics = {
        'Train Loss': history['train_loss'],
        'Train Acc': history['train_acc']
    }
    box_data = [train_metrics[key] for key in train_metrics.keys()]
    bp1 = ax1.boxplot(box_data, labels=list(train_metrics.keys()), patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightblue')
    bp1['boxes'][1].set_facecolor('lightgreen')
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_title('Training Metrics Distribution', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 验证指标箱线图
    ax2 = axes[0, 1]
    val_metrics = {
        'Val Loss': history['val_loss'],
        'Val Acc': history['val_acc'],
        'Val AUC': history['val_auc'],
        'Val F1': history['val_f1']
    }
    box_data2 = [val_metrics[key] for key in val_metrics.keys()]
    bp2 = ax2.boxplot(box_data2, labels=list(val_metrics.keys()), patch_artist=True)
    colors2 = ['lightcoral', 'lightgreen', 'lightyellow', 'lightblue']
    for patch, color in zip(bp2['boxes'], colors2):
        patch.set_facecolor(color)
    ax2.set_ylabel('Value', fontsize=11)
    ax2.set_title('Validation Metrics Distribution', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. 指标相关性热图
    ax3 = axes[1, 0]
    metrics_matrix = np.array([
        history['train_loss'],
        history['train_acc'],
        history['val_loss'],
        history['val_acc'],
        history['val_auc'],
        history['val_f1']
    ])
    # 计算相关性矩阵
    corr_matrix = np.corrcoef(metrics_matrix)
    metric_names = ['Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Val AUC', 'Val F1']
    im = ax3.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax3.set_xticks(range(len(metric_names)))
    ax3.set_yticks(range(len(metric_names)))
    ax3.set_xticklabels(metric_names, rotation=45, ha='right')
    ax3.set_yticklabels(metric_names)
    ax3.set_title('Metrics Correlation Heatmap', fontsize=12, weight='bold')
    # 添加数值标注
    for i in range(len(metric_names)):
        for j in range(len(metric_names)):
            text = ax3.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    plt.colorbar(im, ax=ax3, label='Correlation')
    
    # 4. 指标变化趋势热图（按epoch）
    ax4 = axes[1, 1]
    metrics_trend = np.array([
        history['train_loss'],
        history['train_acc'],
        history['val_loss'],
        history['val_acc'],
        history['val_auc'],
        history['val_f1']
    ])
    # 归一化到[0, 1]
    metrics_trend_norm = (metrics_trend - metrics_trend.min(axis=1, keepdims=True)) / \
                         (metrics_trend.max(axis=1, keepdims=True) - metrics_trend.min(axis=1, keepdims=True) + 1e-10)
    im2 = ax4.imshow(metrics_trend_norm, cmap='viridis', aspect='auto', interpolation='nearest')
    ax4.set_yticks(range(len(metric_names)))
    ax4.set_yticklabels(metric_names)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_title('Normalized Metrics Trend Heatmap', fontsize=12, weight='bold')
    plt.colorbar(im2, ax=ax4, label='Normalized Value')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'metrics_comparison_bio_cot_multimodal_balanced_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 指标对比图已保存（包含箱线图、相关性热图、趋势热图）")


def main():
    """主训练函数"""
    args = BioCOTMultimodalArgs()
    
    # 设置日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.log_dir / f'train_bio_cot_multimodal_balanced_{timestamp}.log'
    
    # 创建日志文件并同时输出到控制台和文件
    import sys
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    log_f = open(log_file, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)
    
    print(f"📝 日志文件: {log_file}")
    print("=" * 80)
    
    # 设置随机种子
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    
    print("=" * 80)
    print("Bio-COT 多模态训练（襄阳数据集 - 平衡样本版本）")
    print("=" * 80)
    print(f"数据路径: {args.data_root}")
    print(f"设备: {args.device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"OCT帧数: {args.oct_num_frames}")
    print(f"Colposcopy图像数: {args.max_col_images}")
    print("=" * 80)
    
    # 创建平衡的数据划分
    train_df, val_df, test_df = create_balanced_split(
        args.data_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state
    )
    
    # 保存重新划分的数据集到临时CSV文件
    temp_train_csv = args.log_dir / f'train_labels_balanced_{timestamp}.csv'
    temp_val_csv = args.log_dir / f'val_labels_balanced_{timestamp}.csv'
    temp_test_csv = args.log_dir / f'test_labels_balanced_{timestamp}.csv'
    
    train_df.to_csv(temp_train_csv, index=False, encoding='utf-8')
    val_df.to_csv(temp_val_csv, index=False, encoding='utf-8')
    test_df.to_csv(temp_test_csv, index=False, encoding='utf-8')
    
    print(f"\n✅ 平衡数据集已保存到临时CSV文件")
    
    # 创建数据集
    train_transform = get_data_transforms(args, is_train=True)
    val_transform = get_data_transforms(args, is_train=False)
    
    train_dataset = XiangyangMultimodalDatasetFromCSV(
        csv_path=str(temp_train_csv),
        transform=train_transform,
        oct_num_frames=args.oct_num_frames,
        max_col_images=args.max_col_images
    )
    
    val_dataset = XiangyangMultimodalDatasetFromCSV(
        csv_path=str(temp_val_csv),
        transform=val_transform,
        oct_num_frames=args.oct_num_frames,
        max_col_images=args.max_col_images
    )
    
    # 自定义collate函数
    def collate_fn(batch):
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
    
    # 创建数据加载器（使用加权采样平衡类别）
    train_labels = [train_dataset[i]['label'].item() for i in range(len(train_dataset))]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,  # 使用加权采样
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn
    )
    
    # 创建模型
    device = torch.device(args.device)
    # 先测试一下特征维度
    test_batch = next(iter(train_loader))
    test_oct_features = extract_features_with_resnet50(test_batch['oct_images'].to(device), device)
    test_colpo_features = extract_features_with_resnet50(test_batch['colposcopy_images'].to(device), device)
    actual_input_dim = test_oct_features.size(-1)  # 应该是2048
    print(f"✅ 检测到实际特征维度: {actual_input_dim}")
    
    model = BioCOTModel(
        embed_dim=768,
        num_classes=2,
        num_centers=1,  # 单中心数据
        input_dim=actual_input_dim,  # 使用实际检测到的维度
        use_vlm_encoder=False
    ).to(device)
    
    print(f"\n📊 使用传统MLP图像编码器")
    print(f"\n✅ 模型创建完成")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数量: {trainable_params:,}")
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [], 'train_cls_loss': [], 'train_ot_loss': [],
        'train_consist_loss': [], 'train_adv_loss': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_f1': []
    }
    
    best_auc = 0.0
    best_epoch = 0
    
    print(f"\n🚀 开始训练...")
    print("=" * 80)
    
    for epoch in range(1, args.num_epochs + 1):
        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args)
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device, epoch, args)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        history['train_cls_loss'].append(train_metrics['cls_loss'])
        history['train_ot_loss'].append(train_metrics['ot_loss'])
        history['train_consist_loss'].append(train_metrics['consist_loss'])
        history['train_adv_loss'].append(train_metrics['adv_loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_f1'].append(val_metrics['f1'])
        
        # 打印epoch结果
        print(f"\nEpoch {epoch}/{args.num_epochs}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"  Loss Components - CLS: {train_metrics['cls_loss']:.4f}, OT: {train_metrics['ot_loss']:.4f}, Consist: {train_metrics['consist_loss']:.4f}, Adv: {train_metrics['adv_loss']:.4f}")
        
        # 保存最佳模型
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_epoch = epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_auc,
                'history': history
            }
            torch.save(checkpoint, args.checkpoint_dir / f'best_model_balanced_{timestamp}.pth')
            print(f"  ✅ 保存最佳模型 (AUC: {best_auc:.4f})")
    
    # 最终评估
    print(f"\n{'='*80}")
    print(f"训练完成！最佳AUC: {best_auc:.4f} (Epoch {best_epoch})")
    print(f"{'='*80}")
    
    # 绘制训练曲线
    plot_training_curves(history, args.output_dir, timestamp)
    
    # 绘制损失热图
    plot_loss_heatmap(history, args.output_dir, timestamp)
    
    # 绘制损失箱线图
    plot_loss_boxplot(history, args.output_dir, timestamp)
    
    # 绘制指标对比图
    plot_metrics_comparison(history, args.output_dir, timestamp)
    
    # 绘制混淆矩阵（使用最佳模型的验证集结果）
    best_checkpoint = torch.load(args.checkpoint_dir / f'best_model_balanced_{timestamp}.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    final_val_metrics = validate(model, val_loader, criterion, device, args.num_epochs, args)
    cm = confusion_matrix(final_val_metrics['labels'], final_val_metrics['preds'])
    plot_confusion_matrix(cm, args.output_dir, timestamp)
    
    # 绘制ROC曲线
    plot_roc_curve(final_val_metrics['labels'], final_val_metrics['probs'], args.output_dir, timestamp)
    
    # 绘制预测概率分布（包含多种图表）
    plot_prediction_distribution(final_val_metrics['labels'], final_val_metrics['probs'], args.output_dir, timestamp)
    
    # 绘制高级小提琴图分析
    print("\n📊 生成高级小提琴图分析...")
    plot_advanced_violin_analysis(final_val_metrics['labels'], final_val_metrics['probs'], args.output_dir, timestamp)
    
    # 绘制损失组件详细分析
    print("\n📊 生成损失组件详细分析...")
    plot_loss_component_analysis(history, args.output_dir, timestamp)
    
    # 绘制特征热图（使用验证集样本）
    print("\n📊 生成特征热图...")
    plot_feature_heatmap(model, val_loader, device, args.output_dir, timestamp, num_samples=min(20, len(val_dataset)))
    
    # 保存结果
    results = {
        'best_epoch': best_epoch,
        'best_auc': best_auc,
        'final_val_acc': final_val_metrics['acc'],
        'final_val_f1': final_val_metrics['f1'],
        'history': history,
        'confusion_matrix': cm.tolist()
    }
    
    with open(args.output_dir / f'results_bio_cot_multimodal_balanced_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存到: {args.output_dir}")
    
    # 恢复stdout/stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_f.close()


if __name__ == '__main__':
    main()

