#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验管理器 - 用于管理所有实验的运行和结果收集
确保实验的可复现性和统计严谨性
"""

import os
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_name: str
    method: str
    random_seed: int = 42
    batch_size: int = 48
    num_epochs: int = 100
    learning_rate: float = 0.0002
    weight_decay: float = 1e-5
    
    # 模块开关
    use_knowledge_notes: bool = True
    use_visual_notes: bool = True
    use_ot: bool = True
    use_dual: bool = True
    use_cross_attn: bool = True
    
    # 损失权重
    lambda_cls: float = 2.0
    lambda_ot: float = 0.5
    lambda_sparse: float = 0.01
    lambda_consist: float = 0.2
    lambda_adv: float = 0.5
    
    # 其他配置
    data_root: str = '/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal'
    output_dir: str = 'experiments/results'
    
    def to_dict(self):
        """转换为字典"""
        return asdict(self)
    
    def save(self, path: str):
        """保存配置"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str):
        """加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


@dataclass
class ExperimentResult:
    """实验结果"""
    experiment_name: str
    run_id: int
    random_seed: int
    
    # 性能指标
    auc: float
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    
    # 训练信息
    best_epoch: int
    train_loss: float
    val_loss: float
    
    # 其他信息
    training_time: float  # 秒
    checkpoint_path: str = ""
    
    def to_dict(self):
        """转换为字典"""
        return asdict(self)


class ExperimentManager:
    """实验管理器"""
    
    def __init__(
        self,
        experiment_name: str,
        config: ExperimentConfig,
        output_dir: str = "experiments/results"
    ):
        """
        Args:
            experiment_name: 实验名称
            config: 实验配置
            output_dir: 输出目录
        """
        self.experiment_name = experiment_name
        self.config = config
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 标准随机种子列表（用于多次运行）
        self.standard_seeds = [42, 123, 456, 789, 2024]
        
        # 结果存储
        self.results: List[ExperimentResult] = []
        
        # 创建结果目录
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
    
    def set_seed(self, seed: int):
        """设置随机种子（确保可复现性）"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def run_experiment(
        self,
        num_runs: int = 5,
        seeds: Optional[List[int]] = None,
        verbose: bool = True
    ) -> List[ExperimentResult]:
        """
        运行实验（多次运行，用于统计检验）
        
        Args:
            num_runs: 运行次数
            seeds: 随机种子列表（如果None，使用标准种子）
            verbose: 是否打印详细信息
        
        Returns:
            results: 实验结果列表
        """
        if seeds is None:
            seeds = self.standard_seeds[:num_runs]
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"开始运行实验: {self.experiment_name}")
            print(f"运行次数: {num_runs}")
            print(f"随机种子: {seeds}")
            print(f"{'='*80}\n")
        
        results = []
        for run_id, seed in enumerate(seeds, 1):
            if verbose:
                print(f"\n运行 {run_id}/{num_runs} (seed={seed})...")
            
            # 设置随机种子
            self.set_seed(seed)
            
            # 运行单次实验
            result = self._run_single(run_id, seed, verbose)
            results.append(result)
            
            # 保存单次结果
            self._save_single_result(result, run_id)
            
            if verbose:
                print(f"  完成! AUC: {result.auc:.4f}, Acc: {result.accuracy:.4f}")
        
        self.results = results
        
        # 保存所有结果
        self._save_all_results()
        
        # 计算统计信息
        stats = self._compute_statistics()
        self._save_statistics(stats)
        
        if verbose:
            self._print_statistics(stats)
        
        return results
    
    def _run_single(
        self,
        run_id: int,
        seed: int,
        verbose: bool = True
    ) -> ExperimentResult:
        """
        运行单次实验
        
        Args:
            run_id: 运行ID
            seed: 随机种子
            verbose: 是否打印详细信息
        
        Returns:
            result: 实验结果
        """
        import time
        start_time = time.time()
        
        # 更新配置中的随机种子
        self.config.random_seed = seed
        
        # 创建模型（根据实验类型）
        model = self._create_model()
        
        # 创建训练器
        trainer = self._create_trainer(model, run_id)
        
        # 训练模型
        train_result = trainer.train()
        
        # 评估模型
        eval_result = trainer.evaluate()
        
        training_time = time.time() - start_time
        
        # 构建结果
        result = ExperimentResult(
            experiment_name=self.experiment_name,
            run_id=run_id,
            random_seed=seed,
            auc=eval_result['auc'],
            accuracy=eval_result['accuracy'],
            precision=eval_result['precision'],
            recall=eval_result['recall'],
            specificity=eval_result['specificity'],
            f1_score=eval_result['f1_score'],
            best_epoch=train_result['best_epoch'],
            train_loss=train_result['train_loss'],
            val_loss=train_result['val_loss'],
            training_time=training_time,
            checkpoint_path=str(trainer.checkpoint_path) if hasattr(trainer, 'checkpoint_path') else ""
        )
        
        return result
    
    def _create_model(self):
        """创建模型（根据配置）"""
        # 这里需要根据实验类型创建不同的模型
        # 例如：Simple Fusion, Standard CLIP, Full Model等
        from models.bio_cot_v3 import create_bio_cot_v3
        from config import BioCOT_v3_Config
        
        # 根据配置创建模型
        if self.config.method == 'simple_fusion':
            # 创建Simple Fusion模型
            pass
        elif self.config.method == 'standard_clip':
            # 创建Standard CLIP模型
            pass
        elif self.config.method == 'full_model':
            # 创建完整模型
            config_obj = BioCOT_v3_Config()
            # 更新配置
            config_obj.use_visual_notes = self.config.use_visual_notes
            config_obj.use_ot = self.config.use_ot
            config_obj.use_dual = self.config.use_dual
            # ...
            model = create_bio_cot_v3(config_obj)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
        
        return model
    
    def _create_trainer(self, model, run_id: int):
        """创建训练器"""
        # 这里需要根据实验类型创建不同的训练器
        # 暂时返回一个占位符
        raise NotImplementedError("需要实现具体的训练器创建逻辑")
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """计算统计信息"""
        if not self.results:
            return {}
        
        metrics = ['auc', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score']
        stats = {}
        
        for metric in metrics:
            values = [getattr(r, metric) for r in self.results]
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
            
            # 计算95%置信区间
            from scipy import stats as scipy_stats
            n = len(values)
            if n > 1:
                ci = scipy_stats.t.interval(
                    0.95, n-1,
                    loc=np.mean(values),
                    scale=scipy_stats.sem(values)
                )
                stats[metric]['ci_95'] = ci
            else:
                stats[metric]['ci_95'] = (values[0], values[0])
        
        return stats
    
    def _save_single_result(self, result: ExperimentResult, run_id: int):
        """保存单次结果"""
        result_path = self.results_dir / f"result_run_{run_id}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _save_all_results(self):
        """保存所有结果"""
        # 保存为JSON
        all_results = [r.to_dict() for r in self.results]
        results_path = self.results_dir / "all_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # 保存为CSV
        df = pd.DataFrame(all_results)
        csv_path = self.results_dir / "all_results.csv"
        df.to_csv(csv_path, index=False)
    
    def _save_statistics(self, stats: Dict[str, Any]):
        """保存统计信息"""
        stats_path = self.results_dir / "statistics.json"
        # 移除values（避免JSON过大）
        stats_clean = {}
        for key, value in stats.items():
            stats_clean[key] = {k: v for k, v in value.items() if k != 'values'}
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_clean, f, indent=2, ensure_ascii=False)
    
    def _print_statistics(self, stats: Dict[str, Any]):
        """打印统计信息"""
        print(f"\n{'='*80}")
        print(f"实验统计信息: {self.experiment_name}")
        print(f"{'='*80}")
        print(f"{'指标':<15} {'均值':<10} {'标准差':<10} {'95% CI':<20} {'范围':<20}")
        print(f"{'-'*80}")
        
        for metric, stat in stats.items():
            mean = stat['mean']
            std = stat['std']
            ci = stat['ci_95']
            min_val = stat['min']
            max_val = stat['max']
            
            print(f"{metric:<15} {mean:.4f}     {std:.4f}     [{ci[0]:.4f}, {ci[1]:.4f}]  [{min_val:.4f}, {max_val:.4f}]")
        
        print(f"{'='*80}\n")
    
    def compare_with_baseline(
        self,
        baseline_results: List[ExperimentResult],
        test_type: str = 'ttest'
    ) -> Dict[str, Any]:
        """
        与baseline对比（统计显著性检验）
        
        Args:
            baseline_results: baseline实验结果
            test_type: 检验类型 ('ttest' or 'wilcoxon')
        
        Returns:
            comparison: 对比结果
        """
        from .statistics import test_significance
        
        if not self.results or not baseline_results:
            return {}
        
        # 提取AUC值
        our_aucs = [r.auc for r in self.results]
        baseline_aucs = [r.auc for r in baseline_results]
        
        # 统计检验
        significance = test_significance(our_aucs, baseline_aucs, test_type=test_type)
        
        # 计算改进幅度
        our_mean = np.mean(our_aucs)
        baseline_mean = np.mean(baseline_aucs)
        improvement = our_mean - baseline_mean
        improvement_pct = (improvement / baseline_mean) * 100
        
        comparison = {
            'our_mean': our_mean,
            'baseline_mean': baseline_mean,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'significance': significance
        }
        
        return comparison


if __name__ == '__main__':
    # 示例用法
    config = ExperimentConfig(
        experiment_name="test_experiment",
        method="full_model",
        random_seed=42
    )
    
    manager = ExperimentManager("test_experiment", config)
    results = manager.run_experiment(num_runs=3, verbose=True)

