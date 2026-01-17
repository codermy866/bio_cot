#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT优化训练脚本（针对低Acc和低Loss问题）
优化策略：
1. 提高学习率，加快收敛
2. 调整损失权重，平衡各损失项
3. 增强分类损失权重，提高分类准确性
4. 优化学习率调度策略
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import roc_auc_score
import time
from datetime import datetime

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
from src.models.bida.bio_cot_model import BioCOTModel
from src.models.bida.prior_net import StudentPriorNet, build_clinical_vector
from src.utils.anti_overfitting import FocalLoss, mixup_data, mixup_criterion, cutmix_data, cutmix_criterion
from src.utils.model_init import apply_initialization


class OptimizedBioCOTArgs:
    """Bio-COT优化训练参数（针对低Acc问题 + 加速训练）"""
    def __init__(self):
        # 基础参数（优化：充分利用GPU内存，加速训练）
        self.data_root = '/data2/hmy/5Center_datas/5centers_multi_leave_centers_out'
        self.device = 'cuda:1'
        self.num_epochs = 50
        self.batch_size = 32  # 进一步减小到32，增加batch数量，减少过拟合风险
        self.num_workers = 0   # 使用单进程模式（避免多进程阻塞，确保稳定性）
        self.pin_memory = True  # 启用pin_memory，加速GPU传输
        self.prefetch_factor = 2  # 预取2个batch，减少等待时间
        self.dataloader_timeout = 600  # 10分钟超时
        
        # OCT帧数配置（减少处理的帧数，加速训练）
        self.oct_num_frames = 60  # 从120帧减少到60帧，加速训练（仍能捕获关键信息）
        self.oct_points = 12
        self.oct_frames_per_point = 5  # 每个点位5帧，总共60帧
        self.load_all_frames = False  # 不需要加载全部120帧
        self.learning_rate = 3e-5  # 平衡学习率：防止过拟合
        self.weight_decay = 3e-3   # 增强正则化：防止过拟合
        self.label_smoothing = 0.2  # 增强标签平滑：防止过拟合
        self.gradient_accumulation_steps = 1  # 梯度累积步数（可调整）
        
        # 数据增强参数（防过拟合 - 增强版）
        self.use_mixup = True  # 启用MixUp（特征向量使用MixUp）
        self.mixup_alpha = 0.2  # MixUp参数：降低到0.2，防止过度混合
        self.use_cutmix = False  # 禁用CutMix：CutMix仅适用于图像，不适用于特征向量
        self.cutmix_alpha = 1.0  # CutMix参数（已禁用）
        self.mixup_prob = 0.4  # MixUp使用概率：降低到0.4，减少数据增强强度
        self.mixup_start_epoch = 5  # 从第5个epoch开始使用MixUp（延迟启用，防止早期过拟合）
        self.cutmix_start_epoch = 999  # CutMix已禁用
        
        # Focal Loss参数（早期训练使用标准CE，后期使用Focal Loss）
        self.use_focal_loss = False  # 早期训练禁用Focal Loss，使用标准CrossEntropy
        self.focal_start_epoch = 10  # 从第10个epoch开始使用Focal Loss
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        
        # Bio-COT损失权重（优化：平衡分类和辅助损失，提升性能）
        self.lambda_cls = 2.0      # 分类损失权重：保持2.0，确保分类任务占主导
        self.lambda_ot = 0.1       # OT损失权重：从0.05提高到0.1，增强语义对齐
        self.lambda_consist = 0.2  # 一致性损失权重：从0.1提高到0.2，增强因果解耦
        self.lambda_adv = 0.05     # 对抗损失权重：从0.01提高到0.05，增强域不变性
        
        # Early Stopping（改进：更严格的策略，防止过拟合）
        self.patience = 8  # 验证集指标8个epoch不提升则停止
        self.min_delta = 0.001  # 最小改善阈值（要求至少0.1%的提升）
        self.monitor_metric = 'val_acc'  # 监控验证集准确率
        
        # 学习率调度（优化：更平衡的调度策略）
        self.warmup_epochs = 3     # Warmup 3个epoch
        self.warmup_lr = 1e-5
        self.min_lr = 5e-7  # 降低最小学习率，允许更长时间训练
        
        # Student Prior预训练
        self.pretrain_student_prior = True
        self.vlm_features_cache = Path(self.data_root) / 'vlm_features_cache' / 'train_vlm_features.npy'
        self.student_pretrain_epochs = 20
        self.student_pretrain_lr = 2e-3
        
        # VLM编码器配置（启用大模型进行特征融合）
        self.use_vlm_encoder = True  # 启用VLM编码器
        # 注意：如果模型无法下载，代码会自动尝试其他模型或回退到传统MLP
        self.vlm_model = "Qwen/Qwen2-VL-2B-Instruct"  # 使用2B模型（更快，更稳定）
        # 备选模型列表（按优先级）：如果2B不可用，会自动尝试
        self.vlm_model_fallbacks = [
            "Qwen/Qwen2-VL-2B-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2-VL"
        ]
        self.freeze_vlm = True  # 冻结VLM参数，仅作为特征提取器
        
        # 输出目录
        self.output_dir = Path(__file__).parent / 'exp_bio_cot'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)


def pretrain_student_prior(args, train_dataset, device):
    """预训练Student Prior网络"""
    print("🔄 预训练Student Prior网络...")
    
    if not args.vlm_features_cache.exists():
        print(f"⚠️ VLM特征缓存不存在: {args.vlm_features_cache}")
        return None
    
    vlm_features = np.load(str(args.vlm_features_cache), allow_pickle=True).item()
    print(f"✅ 加载VLM特征缓存: {len(vlm_features)} 个样本")
    
    student_prior = StudentPriorNet(input_dim=7, output_dim=768).to(device)
    optimizer = torch.optim.Adam(student_prior.parameters(), lr=args.student_pretrain_lr)
    criterion = nn.MSELoss()
    
    vlm_proj = nn.Linear(1536, 768).to(device)
    optimizer_proj = torch.optim.Adam(vlm_proj.parameters(), lr=args.student_pretrain_lr)
    
    student_prior.train()
    vlm_proj.train()
    
    # 快速模式：直接从CSV加载
    train_csv = Path(args.data_root) / 'train_labels.csv'
    if train_csv.exists():
        print("⚡ 使用快速模式：直接从CSV加载clinical数据（跳过OCT）")
        df = pd.read_csv(train_csv)
        clinical_data_list = []
        vlm_feat_list = []
        
        for idx, row in df.iterrows():
            sample_id = row.get('sample_id', f'sample_{idx}')
            if sample_id in vlm_features:
                clinical_data = {
                    'hpv': int(row.get('hpv', 0)),
                    'tct': str(row.get('tct', 'NILM')),
                    'age': int(row.get('age', 45))
                }
                clinical_data_list.append(clinical_data)
                vlm_feat_list.append(vlm_features[sample_id])
        
        print(f"✅ 快速加载完成: {len(clinical_data_list)} 个有效样本")
        
        # 训练循环
        print(f"🔄 开始预训练，共 {args.student_pretrain_epochs} 个epoch...")
        print("⚡ 使用快速模式训练（跳过OCT数据加载）")
        
        batch_size = args.batch_size
        num_batches = (len(clinical_data_list) + batch_size - 1) // batch_size
        
        for epoch in range(args.student_pretrain_epochs):
            total_loss = 0.0
            
            # 随机打乱
            indices = np.random.permutation(len(clinical_data_list))
            
            pbar = tqdm(range(num_batches), desc=f'Pretrain Epoch {epoch+1}/{args.student_pretrain_epochs}')
            for batch_idx in pbar:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(clinical_data_list))
                batch_indices = indices[start_idx:end_idx]
                
                batch_clinical = [clinical_data_list[i] for i in batch_indices]
                batch_vlm = [vlm_feat_list[i] for i in batch_indices]
                
                try:
                    clinical_vec = build_clinical_vector(batch_clinical, device=device)
                except Exception as e:
                    continue
                
                vlm_feat_tensor = torch.from_numpy(np.array(batch_vlm)).to(device)
                vlm_feat_proj = vlm_proj(vlm_feat_tensor)
                
                student_output = student_prior(clinical_vec)
                loss = criterion(student_output, vlm_feat_proj.detach())
                
                optimizer.zero_grad()
                optimizer_proj.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_proj.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        print("✅ Student Prior预训练完成！")
        model_path = args.output_dir / 'student_prior_pretrained.pth'
        torch.save(student_prior.state_dict(), model_path)
        print(f"💾 保存预训练模型: {model_path}")
        return student_prior
    
    return None


def train_epoch(model, train_loader, criterion, optimizer, scaler, args, device, epoch=0):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_ot_loss = 0.0
    total_consist_loss = 0.0
    total_adv_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    
    for batch_idx, batch in enumerate(pbar):
        # 添加batch开始提示
        if batch_idx == 0:
            print(f"\n🔄 开始处理第 {batch_idx + 1} 个batch...", flush=True)
        if isinstance(batch, dict):
            # 确保数据转移到GPU（修复：显式指定device，并验证）
            oct_feat = batch['oct_features'].to(device, non_blocking=True)
            colpo_feat = batch['colposcopy_features'].to(device, non_blocking=True)
            clinical_feat = batch['clinical_features'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            center_labels = batch.get('center_id', torch.zeros(len(labels), dtype=torch.long, device=device))
            
            # 获取原始图像（如果使用VLM）
            oct_images = batch.get('oct_images', None)
            colpo_images = batch.get('col_images', None)
            if oct_images is not None:
                oct_images = oct_images.to(device, non_blocking=True)
            if colpo_images is not None:
                colpo_images = colpo_images.to(device, non_blocking=True)
            
            # 验证数据确实在GPU上（仅在第一个batch检查）
            if batch_idx == 0:
                assert oct_feat.device.type == 'cuda', f"❌ oct_feat在{oct_feat.device}，应该在{device}"
                assert colpo_feat.device.type == 'cuda', f"❌ colpo_feat在{colpo_feat.device}，应该在{device}"
                assert clinical_feat.device.type == 'cuda', f"❌ clinical_feat在{clinical_feat.device}，应该在{device}"
                assert labels.device.type == 'cuda', f"❌ labels在{labels.device}，应该在{device}"
                print(f"✅ 验证：所有数据已转移到GPU ({device})")
                # 调试：打印oct_images信息
                if oct_images is not None:
                    print(f"   🔍 调试：oct_images形状 = {oct_images.shape}, dtype = {oct_images.dtype}")
                else:
                    print(f"   ⚠️ 警告：oct_images为None，将无法使用VLM处理120帧")
            
            # 构建clinical_data（优化：批量处理，减少.item()调用）
            batch_size = clinical_feat.size(0)
            # 批量提取特征，避免循环中的.item()调用
            clinical_feat_cpu = clinical_feat.detach().cpu() if clinical_feat.requires_grad else clinical_feat.cpu()
            hpv_values = (clinical_feat_cpu[:, 1] > 0.5).int().tolist()
            age_values = (clinical_feat_cpu[:, 0] * 100).int().tolist()
            tct_indices = clinical_feat_cpu[:, 2:7].argmax(dim=1).tolist()
            tct_categories = ['ASC-US', 'LSIL', 'HSIL', 'NILM', 'OTHER']
            clinical_data = {
                'hpv': hpv_values,
                'tct': [tct_categories[idx] for idx in tct_indices],
                'age': age_values
            }
        else:
            continue
        
        # 梯度累积：只在第一步清零
        if batch_idx % (args.gradient_accumulation_steps if hasattr(args, 'gradient_accumulation_steps') else 1) == 0:
            optimizer.zero_grad()
        
        with autocast():
            # 数据增强：MixUp和CutMix（仅在训练时且启用时使用，且达到起始epoch）
            mixup_start_epoch = getattr(args, 'mixup_start_epoch', 0)
            cutmix_start_epoch = getattr(args, 'cutmix_start_epoch', 999)  # 默认不启用
            use_mixup = (
                model.training and 
                getattr(args, 'use_mixup', False) and 
                epoch >= mixup_start_epoch and
                np.random.rand() < getattr(args, 'mixup_prob', 0.5)
            )
            use_cutmix = (
                model.training and 
                getattr(args, 'use_cutmix', False) and 
                epoch >= cutmix_start_epoch and
                not use_mixup  # MixUp和CutMix互斥
            )
            
            if use_mixup:
                # MixUp：对特征进行混合
                oct_feat_mixed, labels_a, labels_b, lam = mixup_data(
                    oct_feat, labels, alpha=getattr(args, 'mixup_alpha', 0.2)
                )
                colpo_feat_mixed, _, _, _ = mixup_data(
                    colpo_feat, labels, alpha=getattr(args, 'mixup_alpha', 0.2)
                )
            elif use_cutmix:
                # CutMix：仅适用于图像，不适用于特征向量
                # 如果启用CutMix但输入是特征向量，回退到MixUp
                print("⚠️ 警告：CutMix仅适用于图像，特征向量使用MixUp代替")
                oct_feat_mixed, labels_a, labels_b, lam = mixup_data(
                    oct_feat, labels, alpha=getattr(args, 'mixup_alpha', 0.4)
                )
                colpo_feat_mixed, _, _, _ = mixup_data(
                    colpo_feat, labels, alpha=getattr(args, 'mixup_alpha', 0.4)
                )
            else:
                oct_feat_mixed = oct_feat
                colpo_feat_mixed = colpo_feat
                labels_a = labels
                labels_b = labels
                lam = 1.0
            
            # 添加模型前向传播前的提示
            if batch_idx == 0:
                print(f"   📊 开始模型前向传播（batch {batch_idx + 1}）...", flush=True)
            
            try:
                outputs = model(
                    oct_features=oct_feat_mixed,
                    colpo_features=colpo_feat_mixed,
                    clinical_features=clinical_feat,
                    clinical_data=clinical_data,
                    center_labels=center_labels,
                    return_loss_components=True,
                    use_counterfactual=True,
                    oct_images=oct_images,  # 传递原始OCT图像用于VLM
                    colpo_images=colpo_images  # 传递原始Colposcopy图像用于VLM
                )
                if batch_idx == 0:
                    print(f"   ✅ 模型前向传播完成（batch {batch_idx + 1}）", flush=True)
            except Exception as e:
                print(f"   ❌ 模型前向传播失败（batch {batch_idx + 1}）: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise
            
            logits = outputs['logits']
            
            # 计算分类损失（支持MixUp和CutMix）
            if (use_mixup or use_cutmix) and lam < 1.0:
                if use_mixup:
                    cls_loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                else:  # use_cutmix
                    cls_loss = cutmix_criterion(criterion, logits, labels_a, labels_b, lam)
            else:
                cls_loss = criterion(logits, labels)
            
            loss_components = outputs['loss_components']
            ot_loss = loss_components['L_ot']
            consist_loss = loss_components['L_consist']
            adv_loss = loss_components['L_adv']
            
            # 检查损失有效性
            if torch.isnan(cls_loss) or torch.isinf(cls_loss):
                cls_loss = torch.tensor(0.0, device=cls_loss.device, requires_grad=True)
            if torch.isnan(ot_loss) or torch.isinf(ot_loss):
                ot_loss = torch.tensor(0.0, device=ot_loss.device, requires_grad=True)
            if torch.isnan(consist_loss) or torch.isinf(consist_loss):
                consist_loss = torch.tensor(0.0, device=consist_loss.device, requires_grad=True)
            if torch.isnan(adv_loss) or torch.isinf(adv_loss):
                adv_loss = torch.tensor(0.0, device=adv_loss.device, requires_grad=True)
            
            # 总损失（优化后的权重）
            # 训练初期动态调整权重：前5个epoch降低辅助损失权重
            epoch_factor = 1.0 if epoch >= 5 else (0.3 + 0.7 * epoch / 5)  # 前5个epoch逐渐增加辅助损失
            total_loss_batch = (
                args.lambda_cls * cls_loss +
                epoch_factor * args.lambda_ot * ot_loss +
                epoch_factor * args.lambda_consist * consist_loss +
                epoch_factor * args.lambda_adv * adv_loss
            )
            
            if torch.isnan(total_loss_batch) or torch.isinf(total_loss_batch):
                total_loss_batch = args.lambda_cls * cls_loss
        
        # 检查损失是否有效（在反向传播前）
        if torch.isnan(total_loss_batch) or torch.isinf(total_loss_batch):
            if batch_idx < 3:  # 前几个batch打印详细信息
                print(f"⚠️ Batch {batch_idx}: total_loss_batch无效 (nan/inf)，跳过")
                print(f"   cls_loss: {cls_loss.item():.6f}, ot_loss: {ot_loss.item():.6f}")
                print(f"   consist_loss: {consist_loss.item():.6f}, adv_loss: {adv_loss.item():.6f}")
            continue
        
        # 梯度累积：除以累积步数
        accumulation_steps = args.gradient_accumulation_steps if hasattr(args, 'gradient_accumulation_steps') else 1
        scaled_loss = total_loss_batch / accumulation_steps
        
        # 在反向传播前检查损失和模型参数
        if batch_idx < 3:
            # 检查模型参数是否有nan/inf
            has_nan_params = False
            for name, param in model.named_parameters():
                if param.requires_grad and (torch.isnan(param).any() or torch.isinf(param).any()):
                    print(f"⚠️ Batch {batch_idx}: 模型参数 {name} 包含nan/inf")
                    has_nan_params = True
                    break
            if has_nan_params:
                print(f"⚠️ Batch {batch_idx}: 检测到模型参数包含nan/inf，跳过反向传播")
                scaler.update()
                optimizer.zero_grad()
                continue
            
            # 检查损失值
            print(f"🔍 Batch {batch_idx}: 反向传播前检查")
            print(f"   scaled_loss: {scaled_loss.item():.6f}")
            print(f"   total_loss_batch: {total_loss_batch.item():.6f}")
            print(f"   cls_loss: {cls_loss.item():.6f}, ot_loss: {ot_loss.item():.6f}")
            print(f"   consist_loss: {consist_loss.item():.6f}, adv_loss: {adv_loss.item():.6f}")
        
        # 在反向传播前检查损失值
        if batch_idx < 3:
            print(f"🔍 Batch {batch_idx}: 反向传播前检查")
            print(f"   scaled_loss: {scaled_loss.item():.6f}")
            print(f"   total_loss_batch: {total_loss_batch.item():.6f}")
            print(f"   cls_loss: {cls_loss.item():.6f}, ot_loss: {ot_loss.item():.6f}")
            print(f"   consist_loss: {consist_loss.item():.6f}, adv_loss: {adv_loss.item():.6f}")
        
        # 检查损失是否包含nan/inf（在反向传播前）
        if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
            if batch_idx < 3:
                print(f"⚠️ Batch {batch_idx}: scaled_loss包含nan/inf，跳过反向传播")
            scaler.update()
            optimizer.zero_grad()
            continue
        
        try:
            scaler.scale(scaled_loss).backward()
            
            # 在unscale之前，先检查模型参数是否有nan/inf（可能导致梯度nan）
            if batch_idx < 3:
                has_nan_params = False
                for name, param in model.named_parameters():
                    if param.requires_grad and (torch.isnan(param).any() or torch.isinf(param).any()):
                        print(f"⚠️ Batch {batch_idx}: 模型参数 {name} 包含nan/inf（反向传播后）")
                        has_nan_params = True
                        break
                if has_nan_params:
                    print(f"⚠️ Batch {batch_idx}: 检测到模型参数包含nan/inf，清理梯度")
                    optimizer.zero_grad()
                    try:
                        scaler.unscale_(optimizer)
                    except:
                        pass
                    scaler.update()
                    continue
        except RuntimeError as e:
            if batch_idx < 3:
                print(f"⚠️ Batch {batch_idx}: 反向传播失败: {e}")
                import traceback
                traceback.print_exc()
            # 如果反向传播失败，需要先unscale再update
            try:
                scaler.unscale_(optimizer)
            except:
                pass
            scaler.update()
            optimizer.zero_grad()
            continue
        
        # 标记batch是否成功处理（用于统计）
        batch_success = True  # 默认成功，除非遇到问题
        
        # 只在累积步数达到时才更新
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            try:
                # 修复：只在需要更新时才unscale，避免重复调用
                scaler.unscale_(optimizer)
                
                # 在clip之前，先检查并清理nan/inf梯度（现在可以检查了，因为已经unscale）
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            if batch_idx < 3:
                                print(f"⚠️ Batch {batch_idx}: 清理参数 {name} 的nan/inf梯度")
                            param.grad = torch.where(
                                torch.isnan(param.grad) | torch.isinf(param.grad),
                                torch.zeros_like(param.grad),
                                param.grad
                            )
                
                # 使用更大的max_norm，避免过度裁剪导致梯度消失
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    if batch_idx < 3:
                        print(f"⚠️ Batch {batch_idx}: 梯度无效 (nan/inf)，跳过")
                        print(f"   grad_norm: {grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm}")
                    # 如果梯度无效，需要先更新scaler状态，再清零梯度
                    scaler.update()  # 更新scaler状态，避免下次unscale报错
                    optimizer.zero_grad()
                    batch_success = False  # 标记batch失败
                else:
                    if batch_idx < 3:
                        print(f"✅ Batch {batch_idx}: 梯度有效, grad_norm={grad_norm.item():.6f}")
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            except RuntimeError as e:
                if batch_idx < 3:
                    print(f"⚠️ Batch {batch_idx}: 优化器更新失败: {e}")
                    import traceback
                    traceback.print_exc()
                # 确保在异常情况下也正确更新scaler
                try:
                    scaler.update()
                except:
                    pass
                optimizer.zero_grad()
                batch_success = False  # 标记batch失败
        
        # 统计（无论batch是否成功，都要统计accuracy，但loss只在成功时统计）
        try:
            # 总是统计accuracy（即使batch失败，也要知道预测结果）
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 只在成功处理的batch时统计loss
            if batch_success:
                total_loss += total_loss_batch.item()
                total_cls_loss += cls_loss.item()
                total_ot_loss += ot_loss.item()
                total_consist_loss += consist_loss.item()
                total_adv_loss += adv_loss.item()
            
            # 调试信息（前3个batch）
            if batch_idx < 3:
                print(f"\n🔍 Batch {batch_idx} 调试信息:")
                print(f"   batch_success: {batch_success}")
                print(f"   logits shape: {logits.shape}, range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
                print(f"   logits mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
                # 计算softmax概率，检查预测置信度
                probs = F.softmax(logits, dim=1)
                print(f"   probs range: [{probs.min().item():.4f}, {probs.max().item():.4f}], mean: {probs.mean().item():.4f}")
                print(f"   predicted: {predicted[:5].cpu().tolist() if len(predicted) >= 5 else predicted.cpu().tolist()}")
                print(f"   labels: {labels[:5].cpu().tolist() if len(labels) >= 5 else labels.cpu().tolist()}")
                print(f"   correct: {predicted.eq(labels).sum().item()}/{labels.size(0)}")
                print(f"   total: {total}, correct: {correct}, acc: {100.*correct/total if total > 0 else 0.0:.2f}%")
                if batch_success:
                    print(f"   cls_loss: {cls_loss.item():.6f}")
                    print(f"   ot_loss: {ot_loss.item():.6f} (weighted: {epoch_factor * args.lambda_ot * ot_loss.item():.6f})")
                    print(f"   consist_loss: {consist_loss.item():.6f} (weighted: {epoch_factor * args.lambda_consist * consist_loss.item():.6f})")
                    print(f"   adv_loss: {adv_loss.item():.6f} (weighted: {epoch_factor * args.lambda_adv * adv_loss.item():.6f})")
                    print(f"   epoch_factor: {epoch_factor:.3f} (辅助损失权重调整因子)")
        except Exception as e:
            if batch_idx < 3:
                print(f"⚠️ Batch {batch_idx}: 统计失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 防止除零错误（在进度条显示时）
        acc_display = 100. * correct / total if total > 0 else 0.0
        pbar.set_postfix({
            'Loss': f'{total_loss_batch.item():.4f}',
            'Acc': f'{acc_display:.2f}%',
            'Cls': f'{cls_loss.item():.4f}',
            'OT': f'{ot_loss.item():.4f}',
            'Consist': f'{consist_loss.item():.4f}',
            'Adv': f'{adv_loss.item():.4f}'
        })
    
    # 防止除零错误：如果没有处理任何有效batch
    num_valid_batches = len(train_loader) if len(train_loader) > 0 else 1
    if total == 0:
        print("⚠️ 警告: 没有处理任何有效batch，所有batch可能因梯度无效被跳过")
        return {
            'loss': 0.0,
            'cls_loss': 0.0,
            'ot_loss': 0.0,
            'consist_loss': 0.0,
            'adv_loss': 0.0,
            'acc': 0.0
        }
    
    return {
        'loss': total_loss / num_valid_batches,
        'cls_loss': total_cls_loss / num_valid_batches,
        'ot_loss': total_ot_loss / num_valid_batches,
        'consist_loss': total_consist_loss / num_valid_batches,
        'adv_loss': total_adv_loss / num_valid_batches,
        'acc': 100. * correct / total if total > 0 else 0.0
    }


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            if isinstance(batch, dict):
                oct_feat = batch['oct_features'].to(device)
                colpo_feat = batch['colposcopy_features'].to(device)
                clinical_feat = batch['clinical_features'].to(device)
                labels = batch['label'].to(device)
                
                # 获取原始图像（如果使用VLM）
                oct_images = batch.get('oct_images', None)
                colpo_images = batch.get('col_images', None)
                if oct_images is not None:
                    oct_images = oct_images.to(device)
                if colpo_images is not None:
                    colpo_images = colpo_images.to(device)
                
                batch_size = clinical_feat.size(0)
                # 批量提取特征，避免循环中的.item()调用
                clinical_feat_cpu = clinical_feat.detach().cpu() if clinical_feat.requires_grad else clinical_feat.cpu()
                hpv_values = (clinical_feat_cpu[:, 1] > 0.5).int().tolist()
                age_values = (clinical_feat_cpu[:, 0] * 100).int().tolist()
                tct_indices = clinical_feat_cpu[:, 2:7].argmax(dim=1).tolist()
                tct_categories = ['ASC-US', 'LSIL', 'HSIL', 'NILM', 'OTHER']
                clinical_data = {
                    'hpv': hpv_values,
                    'tct': [tct_categories[idx] for idx in tct_indices],
                    'age': age_values
                }
            else:
                continue
            
            outputs = model(
                oct_features=oct_feat,
                colpo_features=colpo_feat,
                clinical_features=clinical_feat,
                clinical_data=clinical_data,
                return_loss_components=False,
                oct_images=oct_images,  # 传递原始OCT图像用于VLM
                colpo_images=colpo_images  # 传递原始Colposcopy图像用于VLM
            )
            
            logits = outputs['logits']
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    try:
        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = 0.0
    except:
        auc = 0.0
    
    # 防止除零错误
    num_valid_batches = len(val_loader) if len(val_loader) > 0 else 1
    if total == 0:
        print("⚠️ 警告: 验证时没有处理任何有效batch")
        return {
            'loss': 0.0,
            'acc': 0.0,
            'auc': 0.0
        }
    
    return {
        'loss': total_loss / num_valid_batches,
        'acc': 100. * correct / total if total > 0 else 0.0,
        'auc': auc
    }


def main():
    args = OptimizedBioCOTArgs()
    device = torch.device(args.device)
    
    # 验证GPU可用性
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            print(f"⚠️ 警告: CUDA不可用，但device设置为{device}，将使用CPU")
            device = torch.device('cpu')
        else:
            # Ensure device index is valid
            device_idx = device.index if device.index is not None else 0
            if device_idx >= torch.cuda.device_count():
                print(f"⚠️ 警告: 指定的GPU设备 {device_idx} 不可用，将使用默认GPU 0")
                device = torch.device('cuda:0')
                args.device = 'cuda:0'  # Update args to reflect actual device
                device_idx = 0
            
            print(f"✅ 使用设备: {device}")
            try:
                # 确保device_idx有效
                if 0 <= device_idx < torch.cuda.device_count():
                    print(f"   GPU名称: {torch.cuda.get_device_name(device_idx)}")
                    print(f"   GPU内存: {torch.cuda.get_device_properties(device_idx).total_memory / 1024**3:.2f} GB")
                else:
                    print(f"   ⚠️ 设备索引 {device_idx} 无效（可用设备数: {torch.cuda.device_count()}）")
            except (AssertionError, RuntimeError, Exception) as e:
                print(f"   ⚠️ 无法获取GPU信息: {e}")
                print(f"   设备索引: {device_idx}, 可用设备数: {torch.cuda.device_count()}")
    else:
        print(f"⚠️ 使用CPU设备: {device}")
    
    print("📂 加载数据集...")
    print("   使用EnhancedMultimodalCervicalDataset直接加载...")
    # 创建dummy args用于数据集初始化（包含所有必需的属性）
    class DummyArgs:
        def __init__(self):
            self.data_root = args.data_root
            self.data_path = args.data_root
            self.input_size = 224  # 图像输入尺寸（默认224）
            # 使用args中的配置（60帧，加速训练）
            self.oct_num_frames = getattr(args, 'oct_num_frames', 60)  # 60帧：12点位×5帧
            self.oct_points = getattr(args, 'oct_points', 12)
            self.oct_frames_per_point = getattr(args, 'oct_frames_per_point', 5)
            self.load_all_frames = getattr(args, 'load_all_frames', False)  # 不需要加载全部120帧
            self.use_enhanced_oct = True
            self.use_pretrained_backbones = True  # 启用以返回原始图像（用于VLM）
            # 设置OCT缓存目录（如果存在）
            oct_cache_dir = Path(args.data_root) / 'oct_features_cache'
            self.oct_cache_dir = str(oct_cache_dir) if oct_cache_dir.exists() else None
    
    dummy_args = DummyArgs()
    
    # 检查OCT特征缓存
    oct_cache_train = Path(args.data_root) / 'oct_features_cache' / 'train'
    oct_cache_val = Path(args.data_root) / 'oct_features_cache' / 'val'
    
    if oct_cache_train.exists() and len(list(oct_cache_train.glob("*_enhanced.pt"))) > 0:
        num_cached = len(list(oct_cache_train.glob("*_enhanced.pt")))
        print(f"✅ 发现OCT特征缓存: {oct_cache_train} ({num_cached} 个文件)")
        dummy_args.oct_cache_dir = str(oct_cache_train.parent)
    else:
        print(f"⚠️ 未找到OCT特征缓存，训练会很慢！")
        print(f"   建议先运行: python experiments/exp2_bida/precompute_oct_features.py")
        print(f"   继续使用实时处理（会很慢）...")
        dummy_args.oct_cache_dir = None
    
    train_dataset = EnhancedMultimodalCervicalDataset(
        root=Path(args.data_root) / 'internal_train' / 'train',
        is_train='train',
        args=dummy_args,
        use_enhanced_oct=True,
        cache_oct_features=True  # 启用缓存
    )
    val_dataset = EnhancedMultimodalCervicalDataset(
        root=Path(args.data_root) / 'internal_train' / 'val',
        is_train='val',
        args=dummy_args,
        use_enhanced_oct=True,
        cache_oct_features=True  # 启用缓存
    )
    
    print(f"✅ 数据集加载完成: 训练集 {len(train_dataset)} 个样本, 验证集 {len(val_dataset)} 个样本")
    
    # 检查OCT缓存完整性，决定是否使用多进程
    num_cached_train = len(list(oct_cache_train.glob("*_enhanced.pt"))) if oct_cache_train.exists() else 0
    num_cached_val = len(list(oct_cache_val.glob("*_enhanced.pt"))) if oct_cache_val.exists() else 0
    cache_complete = (num_cached_train >= len(train_dataset) * 0.9) and (num_cached_val >= len(val_dataset) * 0.9)
    
    # 注意：即使缓存完整，为了稳定性，使用单进程模式
    # 因为加载120帧时，即使有缓存，也可能因为其他处理导致变慢
    if cache_complete:
        print(f"✅ OCT缓存完整: 训练集{num_cached_train}/{len(train_dataset)}, 验证集{num_cached_val}/{len(val_dataset)}")
        print(f"   使用单进程模式（确保稳定性，避免超时）")
        # 使用单进程模式，避免多进程阻塞和超时
        use_multiprocess = False
        num_workers_train = 0
        num_workers_val = 0
    else:
        print(f"⚠️ OCT缓存不完整: 训练集{num_cached_train}/{len(train_dataset)}, 验证集{num_cached_val}/{len(val_dataset)}")
        print(f"   使用单进程模式（避免超时）")
        use_multiprocess = False
        num_workers_train = 0
        num_workers_val = 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers_train,
        pin_memory=args.pin_memory if hasattr(args, 'pin_memory') and use_multiprocess else False,
        prefetch_factor=args.prefetch_factor if hasattr(args, 'prefetch_factor') and use_multiprocess else None,
        persistent_workers=True if num_workers_train > 0 else False,
        timeout=0 if num_workers_train == 0 else getattr(args, 'dataloader_timeout', 600)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers_val,
        pin_memory=True if use_multiprocess else False,
        prefetch_factor=2 if use_multiprocess else None,
        persistent_workers=True if num_workers_val > 0 else False,
        timeout=600 if use_multiprocess else 0
    )
    
    print(f"✅ 数据加载器创建完成: train_batches={len(train_loader)}, val_batches={len(val_loader)}")
    
    print("🏗️ 创建Bio-COT模型...")
    # 创建模型（支持VLM编码器，带自动回退）
    use_vlm = getattr(args, 'use_vlm_encoder', False)
    vlm_model = getattr(args, 'vlm_model', "Qwen/Qwen2-VL-2B-Instruct")
    freeze_vlm = getattr(args, 'freeze_vlm', True)
    
    # 尝试使用VLM，如果失败则回退到传统MLP
    vlm_loaded = False
    if use_vlm:
        print(f"🚀 尝试使用VLM图像编码器: {vlm_model}")
        print(f"   VLM参数冻结: {freeze_vlm}")
        print(f"   ⚠️ 注意：VLM加载可能需要较长时间（网络下载），请耐心等待...")
        
        # 设置HuggingFace环境变量（在模型加载前）
        import os
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'  # 10分钟超时
        os.environ['HF_HUB_DOWNLOAD_RETRY'] = '5'  # 重试5次
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # 启用快速下载
        
        # 尝试主模型
        models_to_try = [vlm_model]
        # 添加备选模型
        fallback_models = getattr(args, 'vlm_model_fallbacks', [
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2-VL-2B-Instruct"
        ])
        for fallback_model in fallback_models:
            if fallback_model not in models_to_try:
                models_to_try.append(fallback_model)
        
        for model_name in models_to_try:
            try:
                print(f"\n🔄 尝试加载VLM模型: {model_name}")
                print(f"   （这可能需要几分钟，请耐心等待...）")
                # 先尝试创建模型（会触发VLM加载，内部已有重试机制）
                model = BioCOTModel(
                    embed_dim=768,
                    num_classes=2,
                    num_centers=5,
                    input_dim=512,
                    use_vlm_encoder=True,
                    vlm_model=model_name,
                    freeze_vlm=freeze_vlm
                )
                vlm_loaded = True
                print(f"\n✅ VLM模型加载成功: {model_name}")
                vlm_model = model_name  # 更新使用的模型名称
                break
            except Exception as e:
                error_msg = str(e)
                print(f"\n⚠️ VLM模型 {model_name} 加载失败: {error_msg[:300]}")
                if model_name != models_to_try[-1]:
                    print(f"   尝试下一个备选模型...")
                continue
        
        if not vlm_loaded:
            print(f"\n❌ 所有VLM模型加载失败，使用传统MLP")
            print(f"   失败原因可能是：")
            print(f"   1. 网络连接问题（请检查网络或使用镜像）")
            print(f"   2. 模型文件损坏（请清理缓存后重试）")
            print(f"   3. 显存不足（请检查GPU显存）")
            use_vlm = False
    
    if not vlm_loaded:
        print("📊 使用传统MLP图像编码器")
        model = BioCOTModel(
            embed_dim=768,
            num_classes=2,
            num_centers=5,
            input_dim=512,
            use_vlm_encoder=False,
            vlm_model=None,
            freeze_vlm=True
        )
    
    model = model.to(device)
    
    # 应用更好的初始化策略
    print("🔧 应用模型初始化策略...")
    apply_initialization(model, init_type='xavier', num_classes=2)
    
    # 预训练Student Prior
    vlm_cache_path = Path(args.data_root) / 'vlm_features_cache' / 'train_vlm_features.npy'
    if args.pretrain_student_prior and vlm_cache_path.exists():
        print(f"🔄 发现VLM特征缓存: {vlm_cache_path}")
        print("   开始预训练Student Prior...")
        args.vlm_features_cache = vlm_cache_path
        student_prior = pretrain_student_prior(args, train_dataset, device)
        if student_prior is not None:
            model.student_prior.load_state_dict(student_prior.state_dict())
            print("✅ 加载预训练的Student Prior")
        else:
            print("⚠️ Student Prior预训练失败，使用随机初始化")
    else:
        print("ℹ️ Student Prior预训练已禁用，使用随机初始化")
    
    # 创建损失函数（将在训练循环中动态切换）
    focal_start_epoch = getattr(args, 'focal_start_epoch', 10)
    use_focal_loss = getattr(args, 'use_focal_loss', False)
    
    # 初始使用CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing if hasattr(args, 'label_smoothing') else 0.0)
    focal_criterion = None
    if use_focal_loss:
        focal_criterion = FocalLoss(
            alpha=getattr(args, 'focal_alpha', 0.25),
            gamma=getattr(args, 'focal_gamma', 2.0),
            label_smoothing=getattr(args, 'label_smoothing', 0.1)
        )
        print(f"📊 使用CrossEntropyLoss（将在Epoch {focal_start_epoch}切换为Focal Loss）")
    else:
        print("📊 使用CrossEntropyLoss")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler()
    
    # 学习率调度
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=args.warmup_lr / args.learning_rate,
        end_factor=1.0,
        total_iters=args.warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs - args.warmup_epochs,
        eta_min=args.min_lr
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs]
    )
    
    best_val_acc = 0.0
    best_val_auc = 0.0
    patience_counter = 0
    best_epoch = 0
    best_train_val_gap = float('inf')  # 记录最佳的训练-验证差距
    best_model_path = None  # 记录最佳模型路径
    
    # 记录训练历史（用于可视化）
    train_history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'train_val_gap': [],
        'cls_loss': [],
        'ot_loss': [],
        'consist_loss': [],
        'adv_loss': [],
        'lr': []
    }
    
    print("🚀 开始训练（防过拟合优化版）...")
    print(f"   优化配置:")
    print(f"   - 学习率: {args.learning_rate}")
    print(f"   - Weight Decay: {args.weight_decay}")
    print(f"   - Label Smoothing: {args.label_smoothing}")
    print(f"   - Batch Size: {args.batch_size}")
    print(f"   - 分类损失权重: {args.lambda_cls}")
    print(f"   - OT损失权重: {args.lambda_ot}")
    print(f"   - 一致性损失权重: {args.lambda_consist}")
    print(f"   - 对抗损失权重: {args.lambda_adv}")
    if hasattr(args, 'use_mixup') and args.use_mixup:
        print(f"   - MixUp: 启用 (alpha={getattr(args, 'mixup_alpha', 0.2)})")
    if hasattr(args, 'use_cutmix') and args.use_cutmix:
        print(f"   - CutMix: 启用 (alpha={getattr(args, 'cutmix_alpha', 1.0)})")
    if hasattr(args, 'use_focal_loss') and args.use_focal_loss:
        print(f"   - Focal Loss: 启用 (alpha={getattr(args, 'focal_alpha', 0.25)}, gamma={getattr(args, 'focal_gamma', 2.0)})")
    if hasattr(args, 'patience'):
        print(f"   - Early Stopping Patience: {args.patience}")
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 80)
        
        # 动态切换损失函数（从CrossEntropy切换到Focal Loss）
        focal_start_epoch = getattr(args, 'focal_start_epoch', 10)
        if focal_criterion is not None and epoch >= focal_start_epoch and criterion.__class__.__name__ != 'FocalLoss':
            criterion = focal_criterion
            print(f"🔄 Epoch {epoch}: 切换到Focal Loss")
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, args, device, epoch=epoch)
        val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 计算过拟合指标
        train_val_gap = train_metrics['acc'] - val_metrics['acc']
        
        print(f"\n训练指标: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['acc']:.2f}%")
        print(f"验证指标: Loss={val_metrics['loss']:.4f}, Acc={val_metrics['acc']:.2f}%, AUC={val_metrics['auc']:.4f}")
        print(f"训练-验证差距: {train_val_gap:.2f}%")
        print(f"当前学习率: {current_lr:.6f}")
        print(f"损失分解: Cls={train_metrics['cls_loss']:.4f}, OT={train_metrics['ot_loss']:.4f}, "
              f"Consist={train_metrics['consist_loss']:.4f}, Adv={train_metrics['adv_loss']:.4f}")
        
        # 记录训练历史
        train_history['epoch'].append(epoch)
        train_history['train_loss'].append(train_metrics['loss'])
        train_history['train_acc'].append(train_metrics['acc'])
        train_history['val_loss'].append(val_metrics['loss'])
        train_history['val_acc'].append(val_metrics['acc'])
        train_history['val_auc'].append(val_metrics['auc'])
        train_history['train_val_gap'].append(train_val_gap)
        train_history['cls_loss'].append(train_metrics['cls_loss'])
        train_history['ot_loss'].append(train_metrics['ot_loss'])
        train_history['consist_loss'].append(train_metrics['consist_loss'])
        train_history['adv_loss'].append(train_metrics['adv_loss'])
        train_history['lr'].append(current_lr)
        
        # 过拟合警告
        if train_val_gap > 20:
            print(f"⚠️ 警告：训练-验证准确率差距过大 ({train_val_gap:.2f}%)，可能存在过拟合！")
        elif train_val_gap > 10:
            print(f"⚠️ 注意：训练-验证准确率差距较大 ({train_val_gap:.2f}%)")
        
        # 改进的Early Stopping：同时考虑验证准确率和过拟合程度
        monitor_metric = getattr(args, 'monitor_metric', 'val_acc')
        if monitor_metric == 'val_acc':
            current_metric = val_metrics['acc']
        else:
            current_metric = val_metrics['auc']
        
        # 判断是否改善：验证指标提升且过拟合程度不显著增加
        min_delta = getattr(args, 'min_delta', 0.001)
        is_improved = (
            current_metric > best_val_acc + min_delta and
            train_val_gap <= 25.0  # 允许一定的过拟合，但不能太严重
        )
        
        # Early Stopping检查
        improved = False
        if hasattr(args, 'patience'):
            if is_improved:
                improved = True
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"\n⏹️ Early Stopping: 验证集指标{args.patience}个epoch未提升")
                    print(f"   最佳Epoch: {best_epoch}, 最佳Acc: {best_val_acc:.2f}%, 最佳AUC: {best_val_auc:.4f}")
                    print(f"   最佳训练-验证差距: {best_train_val_gap:.2f}%")
                    break
        else:
            # 没有early stopping，使用原来的逻辑
            if val_metrics['acc'] > best_val_acc:
                improved = True
        
        # 保存最佳模型
        if improved:
            best_val_acc = val_metrics['acc']
            best_val_auc = val_metrics['auc']
            best_train_val_gap = train_val_gap
            best_epoch = epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_auc': best_val_auc,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            checkpoint_path = args.output_dir / f'best_model_epoch_{epoch}_acc_{best_val_acc:.2f}.pth'
            torch.save(checkpoint, checkpoint_path)
            best_model_path = checkpoint_path  # 更新最佳模型路径
            print(f"💾 保存最佳模型: {checkpoint_path}")
        elif hasattr(args, 'patience'):
            print(f"⏳ 验证集未提升 ({patience_counter}/{args.patience})")
    
    print(f"\n✅ 训练完成！最佳Epoch: {best_epoch}, 最佳验证Acc: {best_val_acc:.2f}%, 最佳AUC: {best_val_auc:.4f}")
    
    # 保存训练历史
    import json
    history_path = args.output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    print(f"💾 训练历史已保存: {history_path}")
    
    # 训练完成后自动生成可视化结果
    print("\n" + "=" * 80)
    print("🎨 开始生成可视化结果...")
    print("=" * 80)
    
    # 1. 绘制训练曲线
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        import matplotlib.pyplot as plt
        import numpy as np
        
        epochs = train_history['epoch']
        
        # 创建多子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # 1. Loss曲线
        ax = axes[0, 0]
        ax.plot(epochs, train_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, train_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Loss Curves', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 2. Accuracy曲线
        ax = axes[0, 1]
        ax.plot(epochs, train_history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        ax.plot(epochs, train_history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy Curves', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 3. AUC曲线
        ax = axes[0, 2]
        ax.plot(epochs, train_history['val_auc'], 'g-', label='Val AUC', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title('AUC Curve', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 4. 训练-验证差距
        ax = axes[1, 0]
        ax.plot(epochs, train_history['train_val_gap'], 'm-', linewidth=2)
        ax.axhline(y=20, color='r', linestyle='--', label='Warning Threshold (20%)')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Train-Val Gap (%)', fontsize=12)
        ax.set_title('Overfitting Indicator', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 5. 损失组件分解
        ax = axes[1, 1]
        ax.plot(epochs, train_history['cls_loss'], 'b-', label='Cls Loss', linewidth=2)
        ax.plot(epochs, train_history['ot_loss'], 'g-', label='OT Loss', linewidth=2)
        ax.plot(epochs, train_history['consist_loss'], 'r-', label='Consist Loss', linewidth=2)
        ax.plot(epochs, train_history['adv_loss'], 'm-', label='Adv Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Loss Components', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 6. 学习率曲线
        ax = axes[1, 2]
        ax.plot(epochs, train_history['lr'], 'c-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        training_curves_path = args.output_dir / 'training_curves.png'
        plt.savefig(training_curves_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 训练曲线已保存: {training_curves_path}")
    except Exception as e:
        print(f"⚠️ 训练曲线生成失败: {e}")
    
    # 2. 生成反事实轨迹可视化（如果最佳模型存在）
    if best_model_path is not None and best_model_path.exists():
        try:
            print(f"\n🔄 生成反事实轨迹可视化...")
            print(f"   使用模型: {best_model_path}")
            
            # 动态导入可视化函数（使用相对路径）
            import importlib.util
            viz_script_path = Path(__file__).parent / 'visualize_counterfactual.py'
            if viz_script_path.exists():
                spec = importlib.util.spec_from_file_location("visualize_counterfactual", viz_script_path)
                viz_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(viz_module)
                visualize_counterfactual_trajectory = viz_module.visualize_counterfactual_trajectory
            else:
                raise FileNotFoundError(f"可视化脚本不存在: {viz_script_path}")
            
            # 加载验证集
            val_dataset = EnhancedMultimodalCervicalDataset(
                root=Path(args.data_root) / 'internal_train' / 'val',
                labels_file=Path(args.data_root) / 'val_labels.csv',
                use_pretrained_backbones=True
            )
            
            # 加载最佳模型
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # 生成可视化
            counterfactual_viz_path = args.output_dir / 'counterfactual_trajectory.pdf'
            visualize_counterfactual_trajectory(
                model, val_dataset, device,
                num_samples=10,
                num_centers=5,
                output_file=str(counterfactual_viz_path)
            )
            print(f"✅ 反事实轨迹可视化已保存: {counterfactual_viz_path}")
        except Exception as e:
            print(f"⚠️ 反事实轨迹可视化生成失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ 未找到最佳模型，跳过反事实轨迹可视化")
    
    print("\n" + "=" * 80)
    print("✅ 所有可视化结果生成完成！")
    print("=" * 80)
    print(f"\n📁 输出目录: {args.output_dir}")
    print(f"   - 训练历史: {history_path}")
    if best_model_path:
        print(f"   - 最佳模型: {best_model_path}")
    print(f"   - 训练曲线: {args.output_dir / 'training_curves.png'}")
    if best_model_path and best_model_path.exists():
        print(f"   - 反事实轨迹: {args.output_dir / 'counterfactual_trajectory.pdf'}")


if __name__ == '__main__':
    main()

