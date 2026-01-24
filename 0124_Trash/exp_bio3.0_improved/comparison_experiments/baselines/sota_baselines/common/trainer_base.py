#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SOTA Baseline统一训练框架基类
所有SOTA方法都应该继承这个基类，确保统一的训练和评估流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import json
from pathlib import Path
from abc import ABC, abstractmethod
import sys

# 添加项目路径
# trainer_base.py -> common -> sota_baselines -> baselines -> comparison_experiments -> exp_bio3.0_improved
ROOT = Path(__file__).resolve().parents[4]  # 到exp_bio3.0_improved
sys.path.insert(0, str(ROOT))

from data.dataset_v3 import FiveCentersMultimodalDatasetV3
from torchvision import transforms
from utils.experiment_manager import ExperimentConfig, ExperimentResult
from utils.statistics import compute_statistics


class SOTABaselineTrainer(ABC):
    """
    SOTA Baseline训练器基类
    所有SOTA方法都应该继承这个类并实现抽象方法
    """
    
    def __init__(self, config: ExperimentConfig, device: torch.device):
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None
        
    @abstractmethod
    def create_model(self):
        """创建模型 - 子类必须实现"""
        pass
    
    @abstractmethod
    def prepare_data(self, data_root: str):
        """准备数据 - 子类必须实现"""
        pass
    
    @abstractmethod
    def forward_pass(self, batch):
        """前向传播 - 子类必须实现"""
        pass
    
    def set_seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def create_optimizer(self):
        """创建优化器"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def create_criterion(self):
        """创建损失函数"""
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """训练一个epoch（优化显存使用）"""
        self.model.train()
        train_loss = 0.0
        
        # 定期清理显存
        torch.cuda.empty_cache()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}/{self.config.num_epochs} [Train]', leave=False)):
            # 前向传播（由子类实现）
            logits, loss = self.forward_pass(batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            train_loss += loss.item()
            
            # 每10个batch清理一次显存
            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        return train_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader):
        """验证"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='[Val]', leave=False):
                logits, loss = self.forward_pass(batch)
                val_loss += loss.item()
                
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # 计算指标
        acc = accuracy_score(all_labels, all_preds)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        cm = confusion_matrix(all_labels, all_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0
        
        return {
            'loss': val_loss / len(val_loader),
            'accuracy': acc,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1
        }
    
    def train_single_run(self, seed: int, run_id: int, output_dir: Path):
        """训练单次运行"""
        import time
        
        print(f"\n{'='*80}")
        print(f"开始训练运行 {run_id} (seed={seed})")
        print(f"{'='*80}")
        sys.stdout.flush()
        
        # 记录开始时间
        start_time = time.time()
        
        self.set_seed(seed)
        
        # 准备数据
        print("正在准备数据...")
        sys.stdout.flush()
        train_loader, val_loader = self.prepare_data(self.config.data_root)
        print(f"数据准备完成: 训练集{len(train_loader.dataset)}样本, 验证集{len(val_loader.dataset)}样本")
        sys.stdout.flush()
        
        # 创建模型
        print(f"正在创建模型并移动到 {self.device}...")
        sys.stdout.flush()
        self.model = self.create_model().to(self.device)
        print(f"模型创建完成，参数量: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")
        sys.stdout.flush()
        
        # 创建优化器和损失函数
        self.create_optimizer()
        self.create_criterion()
        print("优化器和损失函数已创建")
        sys.stdout.flush()
        
        # 训练循环
        print(f"\n开始训练，共 {self.config.num_epochs} 个epochs...")
        sys.stdout.flush()
        
        best_auc = 0.0
        best_epoch = 0
        best_metrics = None
        
        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.num_epochs}")
            sys.stdout.flush()
            train_loss = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                best_epoch = epoch
                best_metrics = val_metrics.copy()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                      f"Val AUC={val_metrics['auc']:.4f}, Acc={val_metrics['accuracy']:.4f}")
        
        # 计算训练时间
        training_time = time.time() - start_time
        
        return {
            'auc': best_metrics['auc'],
            'accuracy': best_metrics['accuracy'],
            'precision': best_metrics['precision'],
            'recall': best_metrics['recall'],
            'specificity': best_metrics['specificity'],
            'f1_score': best_metrics['f1_score'],
            'best_epoch': best_epoch,
            'train_loss': train_loss,
            'val_loss': best_metrics['loss'],
            'training_time': training_time  # 添加训练时间
        }
    
    def run_experiment(self, num_runs: int = 5, seeds: list = None):
        """运行完整实验（多次运行）"""
        if seeds is None:
            seeds = [42, 123, 456, 789, 2024][:num_runs]
        
        results = []
        for run_id, seed in enumerate(seeds, 1):
            print(f"\n运行 {run_id}/{num_runs} (seed={seed})...")
            result_dict = self.train_single_run(seed, run_id, Path(self.config.output_dir))
            
            result = ExperimentResult(
                experiment_name=self.config.experiment_name,
                run_id=run_id,
                random_seed=seed,
                **result_dict,
                checkpoint_path=""
            )
            results.append(result)
            
            print(f"完成! AUC: {result.auc:.4f}, Acc: {result.accuracy:.4f}")
        
        # 保存结果
        self.save_results(results)
        
        return results
    
    def save_results(self, results: list):
        """保存实验结果"""
        output_path = Path(self.config.output_dir) / self.config.experiment_name
        output_path.mkdir(parents=True, exist_ok=True)
        results_dir = output_path / "results"
        results_dir.mkdir(exist_ok=True)
        
        # 保存JSON
        all_results = [r.to_dict() for r in results]
        with open(results_dir / "all_results.json", 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV
        df = pd.DataFrame(all_results)
        df.to_csv(results_dir / "all_results.csv", index=False)
        
        # 计算统计信息
        stats = {}
        metrics = ['auc', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score']
        for metric in metrics:
            values = [getattr(r, metric) for r in results]
            stats[metric] = compute_statistics(values)
        
        stats_clean = {}
        for key, value in stats.items():
            stats_clean[key] = {k: v for k, v in value.items() if k != 'values'}
        
        with open(results_dir / "statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats_clean, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 结果已保存到: {results_dir}")

