#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT模型在襄阳多模态数据集上的训练脚本
使用OCT + Colposcopy + Clinical数据（多模态）
"""

import sys
from pathlib import Path
import warnings
# 在导入torchvision之前抑制所有UserWarning
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, f1_score
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
    """从CSV文件加载襄阳多模态数据集"""
    
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
        # 注意：CSV中的路径可能指向_multimodal目录，需要检查文件是否存在
        oct_paths_raw = [p.strip() for p in str(oct_path_str).split(';') if p.strip()]
        oct_paths = []
        for p in oct_paths_raw:
            path = Path(p)
            # 如果路径不存在，尝试替换为_binary_multimodal目录
            if not path.exists():
                # 替换路径中的_multimodal为_binary_multimodal
                new_path_str = str(p).replace('襄阳按点图片分类_multimodal', '襄阳按点图片分类_multimodal')
                # 如果还是不存在，尝试从文件名重建路径
                if not Path(new_path_str).exists():
                    # 从完整路径提取文件名，然后在_multimodal目录中查找
                    filename = path.name
                    # 尝试在_multimodal目录中查找
                    base_dir = Path('/data2/hmy/5Center_datas/襄阳按点图片分类_multimodal')
                    found = list(base_dir.rglob(filename))
                    if found:
                        oct_paths.append(found[0])
                    else:
                        # 如果找不到，跳过这个文件
                        continue
                else:
                    oct_paths.append(Path(new_path_str))
            else:
                oct_paths.append(path)
        
        if len(oct_paths) == 0:
            return torch.zeros(self.oct_num_frames, 3, 224, 224)
        
        # 加载图像
        oct_images = []
        for oct_path in oct_paths[:self.oct_num_frames]:
            try:
                img = Image.open(oct_path).convert('RGB')
                img = img.resize((224, 224))
                if self.transform:
                    img = self.transform(img)
                else:
                    img = transforms.ToTensor()(img)
                oct_images.append(img)
            except Exception as e:
                # 如果加载失败，使用零张量
                oct_images.append(torch.zeros(3, 224, 224))
        
        # 如果帧数不足，用零张量填充
        while len(oct_images) < self.oct_num_frames:
            oct_images.append(torch.zeros(3, 224, 224))
        
        # 堆叠为 [F, C, H, W]
        return torch.stack(oct_images[:self.oct_num_frames])
    
    def _load_colposcopy_images(self, col_path_str: str) -> torch.Tensor:
        """加载Colposcopy图像"""
        if pd.isna(col_path_str) or col_path_str == '':
            # 如果没有Colposcopy图像，返回零张量
            return torch.zeros(self.max_col_images, 3, 224, 224)
        
        # 解析路径（分号分隔）
        col_paths_raw = [p.strip() for p in str(col_path_str).split(';') if p.strip()]
        col_paths = []
        for p in col_paths_raw:
            path = Path(p)
            # 如果路径不存在，尝试在_multimodal目录中查找
            if not path.exists():
                filename = path.name
                base_dir = Path('/data2/hmy/5Center_datas/襄阳按点图片分类_multimodal')
                found = list(base_dir.rglob(filename))
                if found:
                    col_paths.append(found[0])
                else:
                    continue
            else:
                col_paths.append(path)
        
        if len(col_paths) == 0:
            return torch.zeros(self.max_col_images, 3, 224, 224)
        
        # 加载图像
        col_images = []
        for col_path in col_paths[:self.max_col_images]:
            try:
                img = Image.open(col_path).convert('RGB')
                img = img.resize((224, 224))
                if self.transform:
                    img = self.transform(img)
                else:
                    img = transforms.ToTensor()(img)
                col_images.append(img)
            except Exception as e:
                # 如果加载失败，使用零张量
                col_images.append(torch.zeros(3, 224, 224))
        
        # 如果图像数不足，用零张量填充
        while len(col_images) < self.max_col_images:
            col_images.append(torch.zeros(3, 224, 224))
        
        # 堆叠为 [K, C, H, W]，K <= max_col_images
        return torch.stack(col_images[:self.max_col_images])
    
    def _parse_tct(self, tct_str) -> List[float]:
        """解析TCT结果，转换为5维one-hot编码"""
        # TCT类别映射: NILM=0, ASC-US=1, LSIL=2, HSIL=3, 其他=4
        tct_mapping = {
            'NILM': 0,
            'ASC-US': 1,
            'LSIL': 2,
            'HSIL': 3,
            'ASC-H': 3,  # ASC-H归类为HSIL
            '恶性肿瘤(癌)': 4,
            '癌': 4
        }
        
        if pd.isna(tct_str) or tct_str == '' or tct_str == ' ':
            tct_idx = 0  # 默认NILM
        else:
            tct_str = str(tct_str).strip().upper()
            tct_idx = tct_mapping.get(tct_str, 0)  # 未知类别默认为NILM
        
        # 转换为one-hot编码
        tct_onehot = [0.0] * 5
        tct_onehot[tct_idx] = 1.0
        return tct_onehot
    
    def _parse_hpv(self, hpv_str) -> float:
        """解析HPV结果，转换为二值（0或1）"""
        if pd.isna(hpv_str) or hpv_str == '' or hpv_str == '-':
            return 0.0
        
        hpv_str = str(hpv_str).strip()
        
        # 如果包含数字，判断是否阳性
        try:
            # 尝试提取数字
            import re
            numbers = re.findall(r'\d+', hpv_str)
            if numbers:
                # 如果有数字，认为是阳性
                return 1.0
            else:
                return 0.0
        except:
            # 如果无法解析，默认为阴性
            return 0.0
    
    def __getitem__(self, idx: int) -> Dict:
        """获取一个样本"""
        row = self.df.iloc[idx]
        
        # 加载OCT帧序列 [F, C, H, W]
        oct_frames = self._load_oct_frames(row['oct_path'])
        
        # 加载Colposcopy图像 [K, C, H, W]
        col_images = self._load_colposcopy_images(row['col_path'])
        
        # 构建临床特征向量 [7] = [HPV(1) + TCT(5) + Age(1)]
        age = float(row['age']) / 100.0 if not pd.isna(row['age']) else 0.5  # 归一化到[0,1]
        hpv = self._parse_hpv(row['hpv'])
        tct_onehot = self._parse_tct(row['tct'])
        
        clinical_features = torch.tensor([hpv] + tct_onehot + [age], dtype=torch.float32)
        
        # 构建临床数据字典（用于Student Prior）- 单个样本格式
        clinical_data = {
            'hpv': int(hpv),
            'tct': row['tct'] if not pd.isna(row['tct']) and str(row['tct']).strip() != '' else 'NILM',
            'age': float(row['age']) if not pd.isna(row['age']) else 50.0
        }
        
        # 标签
        label = torch.tensor(int(row['label']), dtype=torch.long)
        
        return {
            'oct_images': oct_frames,  # [F, C, H, W]
            'colposcopy_images': col_images,  # [K, C, H, W]
            'clinical_features': clinical_features,  # [7]
            'clinical_data': clinical_data,  # dict for Student Prior
            'label': label,
            'oct_id': row['oct_id']
        }


class BioCOTMultimodalArgs:
    """Bio-COT多模态训练参数"""
    def __init__(self):
        # 数据路径
        self.data_root = '/data2/hmy/5Center_datas/襄阳按点图片分类_binary_multimodal'
        self.output_dir = Path(__file__).parent / 'results_multimodal'
        self.checkpoint_dir = Path(__file__).parent / 'checkpoints_multimodal'
        self.log_dir = Path(__file__).parent / 'logs'
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练参数
        self.device = 'cuda:1'
        self.num_epochs = 50
        self.batch_size = 8  # 多模态数据batch size稍小
        self.num_workers = 4
        self.pin_memory = True
        
        # 优化器参数
        self.learning_rate = 1.2e-4
        self.weight_decay = 1e-4
        self.momentum = 0.9
        
        # 学习率调度
        self.warmup_epochs = 3
        self.use_warmup = True
        
        # 梯度裁剪
        self.max_grad_norm = 1.0
        
        # 数据增强
        self.input_size = 224
        self.use_augmentation = True
        
        # OCT参数
        self.oct_num_frames = 60  # 使用60帧OCT
        self.max_col_images = 3  # 最多3张Colposcopy图像
        
        # Early Stopping
        self.patience = 10
        self.min_delta = 0.001
        
        # Bio-COT模型参数
        self.embed_dim = 768
        self.num_classes = 2
        self.num_centers = 1  # 单中心数据集
        self.input_dim = 512
        self.use_vlm_encoder = False  # 不使用VLM，使用传统MLP编码器
        
        # 损失函数权重
        self.lambda_cls = 1.0
        self.lambda_ot = 1.0
        self.lambda_consist = 0.5
        self.lambda_adv = 0.1


def get_data_transforms(args, is_train=True):
    """获取数据增强变换"""
    if is_train and args.use_augmentation:
        return transforms.Compose([
            transforms.Resize((args.input_size + 32, args.input_size + 32)),
            transforms.RandomCrop(args.input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])


def extract_features_with_resnet50(images, device):
    """使用ResNet50提取图像特征"""
    import os
    # 设置环境变量抑制警告
    os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
    
    from torchvision import models
    import warnings
    # 使用新版本API，避免警告
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        try:
            # 新版本torchvision使用weights参数
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
        except (AttributeError, TypeError):
            # 兼容旧版本
            resnet = models.resnet50(pretrained=True).to(device)
    resnet.eval()
    resnet.fc = nn.Identity()
    
    with torch.no_grad():
        # 处理多帧OCT图像
        if images.dim() == 5:  # [B, F, C, H, W]
            B, F, C, H, W = images.shape
            images_flat = images.view(B * F, C, H, W)
            features_flat = resnet(images_flat)  # [B*F, 2048]
            features = features_flat.view(B, F, -1).mean(dim=1)  # [B, 2048] - 平均池化
        else:  # [B, C, H, W]
            features = resnet(images)  # [B, 2048]
    
    # 投影到512维
    proj = nn.Linear(2048, 512).to(device)
    features_512 = proj(features)  # [B, 512]
    
    return features_512


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, args):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_ot_loss = 0.0
    running_consist_loss = 0.0
    running_adv_loss = 0.0
    
    all_preds = []
    all_labels = []
    
    ema_loss = None
    ema_decay = 0.9
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Train]')
    for batch in pbar:
        oct_images = batch['oct_images'].to(device)  # [B, F, C, H, W]
        colposcopy_images = batch['colposcopy_images'].to(device)  # [B, K, C, H, W]
        clinical_features = batch['clinical_features'].to(device)  # [B, 7]
        clinical_data_list = batch['clinical_data']  # list of dict
        labels = batch['label'].to(device)
        
        # 提取图像特征（使用ResNet50）
        # OCT特征：平均池化多帧
        oct_feat = extract_features_with_resnet50(oct_images, device)  # [B, 512]
        
        # Colposcopy特征：平均池化多张图像
        B, K, C, H, W = colposcopy_images.shape
        colpo_images_flat = colposcopy_images.view(B * K, C, H, W)
        colpo_feat_flat = extract_features_with_resnet50(colpo_images_flat, device)  # [B*K, 512]
        colpo_feat = colpo_feat_flat.view(B, K, -1).mean(dim=1)  # [B, 512]
        
        # 检查是否有Colposcopy图像（全零表示没有）
        has_colpo = colpo_feat.abs().sum(dim=1) > 1e-6  # [B]
        
        # 前向传播
        optimizer.zero_grad()
        
        # 构建batch的clinical_data字典（batch格式）
        # clinical_data_list是一个list，每个元素是一个dict
        B = oct_feat.size(0)  # 获取batch size
        
        # 确保clinical_data_list的长度等于batch size
        if len(clinical_data_list) != B:
            print(f"⚠️ Warning: clinical_data_list length ({len(clinical_data_list)}) != B ({B})")
            # 使用clinical_features作为fallback
            batch_clinical_data = None
        else:
            batch_clinical_data = {
                'hpv': [],
                'tct': [],
                'age': []
            }
            for d in clinical_data_list:
                if isinstance(d, dict):
                    batch_clinical_data['hpv'].append(int(d.get('hpv', 0)))
                    batch_clinical_data['tct'].append(str(d.get('tct', 'NILM')))
                    batch_clinical_data['age'].append(float(d.get('age', 50.0)))
                else:
                    # 如果格式不对，使用默认值
                    batch_clinical_data['hpv'].append(0)
                    batch_clinical_data['tct'].append('NILM')
                    batch_clinical_data['age'].append(50.0)
        
        output = model(
            oct_features=oct_feat,
            colpo_features=colpo_feat,
            clinical_features=clinical_features,
            clinical_data=batch_clinical_data,
            center_labels=None,  # 单中心数据集
            return_loss_components=True,
            use_counterfactual=False  # 单中心数据集，不使用反事实干预
        )
        
        logits = output['logits']
        loss_components = output.get('loss_components', {})
        
        # 计算总损失
        cls_loss = criterion(logits, labels)
        ot_loss = loss_components.get('L_ot', torch.tensor(0.0, device=device))
        consist_loss = loss_components.get('L_consist', torch.tensor(0.0, device=device))
        adv_loss = loss_components.get('L_adv', torch.tensor(0.0, device=device))
        
        total_loss = (
            args.lambda_cls * cls_loss +
            args.lambda_ot * ot_loss +
            args.lambda_consist * consist_loss +
            args.lambda_adv * adv_loss
        )
        
        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()
        
        # 统计
        batch_loss = total_loss.item()
        running_loss += batch_loss
        running_cls_loss += cls_loss.item()
        running_ot_loss += ot_loss.item() if isinstance(ot_loss, torch.Tensor) else ot_loss
        running_consist_loss += consist_loss.item() if isinstance(consist_loss, torch.Tensor) else consist_loss
        running_adv_loss += adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss
        
        if ema_loss is None:
            ema_loss = batch_loss
        else:
            ema_loss = ema_decay * ema_loss + (1 - ema_decay) * batch_loss
        
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pos_preds = (preds == 1).sum().item()
        pos_labels = (labels == 1).sum().item()
        pos_probs = torch.softmax(logits, dim=1)[:, 1].mean().item()
        
        acc = accuracy_score(all_labels, all_preds)
        pbar.set_postfix({
            'loss': f'{ema_loss:.4f}',
            'cls': f'{cls_loss.item():.4f}',
            'ot': f'{ot_loss.item() if isinstance(ot_loss, torch.Tensor) else ot_loss:.4f}',
            'acc': f'{acc:.4f}',
            'pos_pred': f'{pos_preds}/{pos_labels}',
            'pos_prob': f'{pos_probs:.3f}'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return {
        'loss': epoch_loss,
        'acc': epoch_acc,
        'cls_loss': running_cls_loss / len(dataloader),
        'ot_loss': running_ot_loss / len(dataloader),
        'consist_loss': running_consist_loss / len(dataloader),
        'adv_loss': running_adv_loss / len(dataloader)
    }


def validate(model, dataloader, criterion, device, epoch, args):
    """验证"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Val]')
        for batch in pbar:
            oct_images = batch['oct_images'].to(device)
            colposcopy_images = batch['colposcopy_images'].to(device)
            clinical_features = batch['clinical_features'].to(device)
            clinical_data_list = batch['clinical_data']
            labels = batch['label'].to(device)
            
            # 提取图像特征
            oct_feat = extract_features_with_resnet50(oct_images, device)
            
            B, K, C, H, W = colposcopy_images.shape
            colpo_images_flat = colposcopy_images.view(B * K, C, H, W)
            colpo_feat_flat = extract_features_with_resnet50(colpo_images_flat, device)
            colpo_feat = colpo_feat_flat.view(B, K, -1).mean(dim=1)
            
            # 构建batch的clinical_data字典
            B = oct_feat.size(0)
            
            if len(clinical_data_list) != B:
                batch_clinical_data = None
            else:
                batch_clinical_data = {
                    'hpv': [],
                    'tct': [],
                    'age': []
                }
                for d in clinical_data_list:
                    if isinstance(d, dict):
                        batch_clinical_data['hpv'].append(int(d.get('hpv', 0)))
                        batch_clinical_data['tct'].append(str(d.get('tct', 'NILM')))
                        batch_clinical_data['age'].append(float(d.get('age', 50.0)))
                    else:
                        batch_clinical_data['hpv'].append(0)
                        batch_clinical_data['tct'].append('NILM')
                        batch_clinical_data['age'].append(50.0)
            
            # 前向传播
            output = model(
                oct_features=oct_feat,
                colpo_features=colpo_feat,
                clinical_features=clinical_features,
                clinical_data=batch_clinical_data,
                center_labels=None,
                return_loss_components=False
            )
            
            logits = output['logits']
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            adjusted_threshold = 0.35
            preds = (probs[:, 1] > adjusted_threshold).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
            acc = accuracy_score(all_labels, all_preds)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except:
        epoch_auc = 0.0
    
    try:
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')
        epoch_f1_pos = f1_score(all_labels, all_preds, pos_label=1)
    except:
        epoch_f1 = 0.0
        epoch_f1_pos = 0.0
    
    return {
        'loss': epoch_loss,
        'acc': epoch_acc,
        'auc': epoch_auc,
        'f1': epoch_f1,
        'f1_pos': epoch_f1_pos,
        'preds': all_preds,
        'labels': all_labels,
        'probs': all_probs
    }


def main():
    """主训练函数"""
    args = BioCOTMultimodalArgs()
    
    # 设置日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.log_dir / f'train_bio_cot_multimodal_{timestamp}.log'
    
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
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=" * 80)
    print("Bio-COT 多模态训练（襄阳数据集）")
    print("=" * 80)
    print(f"数据路径: {args.data_root}")
    print(f"设备: {args.device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"OCT帧数: {args.oct_num_frames}")
    print(f"Colposcopy图像数: {args.max_col_images}")
    print("=" * 80)
    
    # 创建数据集
    train_transform = get_data_transforms(args, is_train=True)
    val_transform = get_data_transforms(args, is_train=False)
    
    train_csv = Path(args.data_root) / 'train_labels.csv'
    val_csv = Path(args.data_root) / 'val_labels.csv'
    
    train_dataset = XiangyangMultimodalDatasetFromCSV(
        csv_path=str(train_csv),
        transform=train_transform,
        oct_num_frames=args.oct_num_frames,
        max_col_images=args.max_col_images
    )
    
    val_dataset = XiangyangMultimodalDatasetFromCSV(
        csv_path=str(val_csv),
        transform=val_transform,
        oct_num_frames=args.oct_num_frames,
        max_col_images=args.max_col_images
    )
    
    # 自定义collate函数，正确处理多模态数据
    def collate_fn(batch):
        """自定义collate函数，正确处理clinical_data"""
        oct_images = torch.stack([item['oct_images'] for item in batch])
        colposcopy_images = torch.stack([item['colposcopy_images'] for item in batch])
        clinical_features = torch.stack([item['clinical_features'] for item in batch])
        clinical_data = [item['clinical_data'] for item in batch]  # 保持为list
        labels = torch.stack([item['label'] for item in batch])
        oct_ids = [item['oct_id'] for item in batch]
        
        return {
            'oct_images': oct_images,
            'colposcopy_images': colposcopy_images,
            'clinical_features': clinical_features,
            'clinical_data': clinical_data,  # list of dict
            'label': labels,
            'oct_id': oct_ids
        }
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
    model = BioCOTModel(
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        num_centers=args.num_centers,
        input_dim=args.input_dim,
        use_vlm_encoder=args.use_vlm_encoder
    ).to(args.device)
    
    print(f"\n✅ 模型创建完成")
    print(f"   总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 损失函数
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    if args.use_warmup:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=args.warmup_epochs * len(train_loader)
        )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs - args.warmup_epochs if args.use_warmup else args.num_epochs,
        eta_min=5e-7
    )
    
    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_f1': [],
        'train_cls_loss': [], 'train_ot_loss': [], 'train_consist_loss': [], 'train_adv_loss': []
    }
    
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确保日志文件已创建（在main函数开始时已创建）
    # timestamp在main函数开始时已生成，这里使用相同的timestamp
    
    print("\n🚀 开始训练...")
    print("=" * 80)
    
    for epoch in range(args.num_epochs):
        # Warmup
        if args.use_warmup and epoch < args.warmup_epochs:
            warmup_scheduler.step()
        
        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, args.device, epoch, args)
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, args.device, epoch, args)
        
        # 更新学习率（Warmup后）
        if not (args.use_warmup and epoch < args.warmup_epochs):
            scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['acc'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_cls_loss'].append(train_metrics['cls_loss'])
        history['train_ot_loss'].append(train_metrics['ot_loss'])
        history['train_consist_loss'].append(train_metrics['consist_loss'])
        history['train_adv_loss'].append(train_metrics['adv_loss'])
        
        # 打印epoch结果
        print(f"\nEpoch {epoch+1}/{args.num_epochs}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"  Loss Components - CLS: {train_metrics['cls_loss']:.4f}, OT: {train_metrics['ot_loss']:.4f}, Consist: {train_metrics['consist_loss']:.4f}, Adv: {train_metrics['adv_loss']:.4f}")
        
        # 保存最佳模型
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint_path = args.checkpoint_dir / f'best_bio_cot_multimodal_{timestamp}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
                'val_metrics': val_metrics
            }, checkpoint_path)
            print(f"  ✅ 保存最佳模型 (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
        
        # Early Stopping
        if patience_counter >= args.patience:
            print(f"\n⏹️  Early Stopping (patience={args.patience})")
            break
    
    print("\n" + "=" * 80)
    print(f"训练完成！最佳Epoch: {best_epoch+1}, 最佳AUC: {best_val_auc:.4f}")
    print("=" * 80)
    
    # 加载最佳模型进行最终评估
    checkpoint = torch.load(args.checkpoint_dir / f'best_bio_cot_multimodal_{timestamp}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_val_metrics = validate(model, val_loader, criterion, args.device, epoch, args)
    
    # 计算混淆矩阵
    cm = confusion_matrix(final_val_metrics['labels'], final_val_metrics['preds'])
    
    # 保存结果
    results = {
        'best_epoch': best_epoch + 1,
        'best_val_auc': best_val_auc,
        'final_val_acc': final_val_metrics['acc'],
        'final_val_auc': final_val_metrics['auc'],
        'final_val_f1': final_val_metrics['f1'],
        'final_val_f1_pos': final_val_metrics['f1_pos'],
        'confusion_matrix': cm.tolist(),
        'history': history
    }
    
    results_path = args.output_dir / f'results_bio_cot_multimodal_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果已保存: {results_path}")
    
    # 绘制训练曲线
    plot_training_curves(history, args.output_dir, timestamp)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(cm, args.output_dir, timestamp)
    
    print("\n🎉 训练完成！")
    print(f"📝 日志文件已保存: {log_file}")
    
    # 恢复标准输出
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_f.close()


def plot_training_curves(history, output_dir, timestamp):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss曲线
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy曲线
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # AUC曲线
    axes[0, 2].plot(history['val_auc'], label='Val AUC', color='green')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('AUC')
    axes[0, 2].set_title('Validation AUC')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Loss Components
    axes[1, 0].plot(history['train_cls_loss'], label='CLS Loss')
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
    plt.savefig(output_dir / f'training_curves_bio_cot_multimodal_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 训练曲线已保存")


def plot_confusion_matrix(cm, output_dir, timestamp):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
    else:
        # 使用matplotlib绘制混淆矩阵
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        plt.xticks([0, 1], ['Negative', 'Positive'])
        plt.yticks([0, 1], ['Negative', 'Positive'])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_bio_cot_multimodal_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存")


if __name__ == '__main__':
    main()

