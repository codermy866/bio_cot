#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 3: Bio-COT训练脚本
整合Student Prior、Sinkhorn OT、Memory Bank等新模块
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import roc_auc_score

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 注意：需要根据实际的数据集类调整导入
# 如果数据集类需要args参数，需要创建args对象
try:
    from src.data.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
except ImportError:
    # Fallback: 使用build_enhanced_dataset
    from src.data.enhanced_multimodal_dataset import build_enhanced_dataset
from src.models.bida.bio_cot_model import BioCOTModel
from src.models.bida.prior_net import StudentPriorNet, build_clinical_vector


class BioCOTArgs:
    """Bio-COT训练参数（优化版：加速训练+提高效果）"""
    def __init__(self):
        # 基础参数（优化：增大batch size加速训练）
        self.data_root = '/data2/hmy/5Center_datas/5centers_multi_leave_centers_out'
        self.device = 'cuda:1'
        self.num_epochs = 50
        self.batch_size = 64  # 从32增加到64，加速训练
        self.num_workers = 4  # 从8降到4，减少内存占用
        self.learning_rate = 1e-4  # 降低学习率，使训练更稳定
        self.weight_decay = 5e-4   # 降低权重衰减，减少正则化强度
        
        # Bio-COT损失权重（优化：降低损失权重，使loss更稳定）
        self.lambda_cls = 1.0      # 分类损失权重
        self.lambda_ot = 0.5       # OT损失权重（降低，避免过大）
        self.lambda_consist = 1.0  # 一致性损失权重（降低）
        self.lambda_adv = 0.5      # 对抗损失权重（降低）
        
        # 学习率调度（优化：更快warmup）
        self.warmup_epochs = 3  # 从5降到3，更快进入正常训练
        self.warmup_lr = 1e-5
        self.min_lr = 1e-6
        
        # Student Prior预训练（优化：减少epochs加速）
        self.pretrain_student_prior = True
        self.vlm_features_cache = Path(self.data_root) / 'vlm_features_cache' / 'train_vlm_features.npy'
        self.student_pretrain_epochs = 20  # 从50降到20，加速预训练
        self.student_pretrain_lr = 2e-3  # 从1e-3提高到2e-3，加快预训练收敛
        
        # 输出目录
        self.output_dir = Path(__file__).parent / 'exp_bio_cot'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)


def pretrain_student_prior(args, train_dataset, device):
    """预训练Student Prior网络（优化版：直接使用CSV数据，跳过OCT加载）"""
    print("🔄 预训练Student Prior网络...")
    
    # 加载VLM特征缓存
    if not args.vlm_features_cache.exists():
        print(f"⚠️ VLM特征缓存不存在: {args.vlm_features_cache}")
        print("   请先运行 extract_vlm_features.py 提取VLM特征")
        return None
    
    vlm_features = np.load(str(args.vlm_features_cache), allow_pickle=True).item()
    print(f"✅ 加载VLM特征缓存: {len(vlm_features)} 个样本")
    
    # 创建Student Prior
    student_prior = StudentPriorNet(input_dim=7, output_dim=768).to(device)
    optimizer = torch.optim.Adam(student_prior.parameters(), lr=args.student_pretrain_lr)
    criterion = nn.MSELoss()
    
    # 投影层：VLM特征从1536维投影到768维
    vlm_proj = nn.Linear(1536, 768).to(device)
    optimizer_proj = torch.optim.Adam(vlm_proj.parameters(), lr=args.student_pretrain_lr)
    
    student_prior.train()
    vlm_proj.train()
    
    # 优化：直接从CSV加载clinical数据，避免OCT数据加载瓶颈
    import pandas as pd
    train_csv = Path(args.data_root) / 'train_labels.csv'
    if train_csv.exists():
        print("⚡ 使用快速模式：直接从CSV加载clinical数据（跳过OCT）")
        df = pd.read_csv(train_csv)
        # 构建clinical数据列表
        clinical_data_list = []
        vlm_feat_list = []
        
        for idx, row in df.iterrows():
            # 构建clinical_data
            clinical_data = {
                'hpv': int(row.get('hpv', 0)) if pd.notna(row.get('hpv')) else 0,
                'tct': str(row.get('tct', 'NILM')) if pd.notna(row.get('tct')) else 'NILM',
                'age': int(row.get('age', 50)) if pd.notna(row.get('age')) else 50
            }
            
            # 尝试获取VLM特征（使用patient_id或索引）
            patient_id = str(row.get('patient_id', f'sample_{idx}'))
            vlm_feat = None
            for pid_key in [patient_id, f'sample_{idx}', str(idx)]:
                if pid_key in vlm_features:
                    vlm_feat = vlm_features[pid_key]
                    break
            
            if vlm_feat is not None:
                clinical_data_list.append(clinical_data)
                vlm_feat_list.append(vlm_feat)
        
        print(f"✅ 快速加载完成: {len(clinical_data_list)} 个有效样本")
        
        # 创建简化的数据加载器（不使用原始dataset）
        train_loader = None  # 将使用列表直接迭代
    else:
        print("⚠️ 未找到train_labels.csv，使用原始数据集（可能较慢）")
        # 创建数据加载器（优化：减少num_workers，使用pin_memory加速）
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=0,  # 设为0，避免多进程导致的OCT加载问题
            pin_memory=False,  # 单进程不需要pin_memory
            prefetch_factor=None,
            persistent_workers=False
        )
        clinical_data_list = None
        vlm_feat_list = None
    
    print(f"🔄 开始预训练，共 {args.student_pretrain_epochs} 个epoch...")
    
    # 快速模式：直接使用列表数据
    if clinical_data_list is not None and vlm_feat_list is not None:
        print("⚡ 使用快速模式训练（跳过OCT数据加载）")
        n_samples = len(clinical_data_list)
        indices = np.arange(n_samples)
        
        for epoch in range(args.student_pretrain_epochs):
            total_loss = 0.0
            n_batches = 0
            
            # 随机打乱
            np.random.shuffle(indices)
            
            # 批量处理
            for batch_start in tqdm(range(0, n_samples, args.batch_size), 
                                   desc=f"Pretrain Epoch {epoch+1}/{args.student_pretrain_epochs}"):
                batch_indices = indices[batch_start:batch_start + args.batch_size]
                batch_clinical = [clinical_data_list[i] for i in batch_indices]
                batch_vlm = [vlm_feat_list[i] for i in batch_indices]
                
                # 构建临床向量
                # batch_clinical是字典列表，需要确保格式正确
                try:
                    # 验证batch_clinical格式
                    if isinstance(batch_clinical, list) and len(batch_clinical) > 0:
                        # 检查每个元素是否是字典
                        if not isinstance(batch_clinical[0], dict):
                            # 如果不是字典，跳过这个batch
                            continue
                        clinical_vec = build_clinical_vector(batch_clinical, device=device)
                    else:
                        continue
                except Exception as e:
                    # 静默跳过，避免过多warning
                    continue
                
                # VLM特征
                vlm_feat_tensor = torch.from_numpy(np.array(batch_vlm)).to(device)  # [B, 1536]
                vlm_feat_proj = vlm_proj(vlm_feat_tensor)  # [B, 768]
                
                # Student Prior输出
                z_sem = student_prior(clinical_vec)  # [B, 768]
                
                # 计算损失
                loss = criterion(z_sem, vlm_feat_proj)
                
                # 反向传播
                optimizer.zero_grad()
                optimizer_proj.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_proj.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            if n_batches > 0:
                avg_loss = total_loss / n_batches
                if (epoch + 1) % 5 == 0 or (epoch + 1) == args.student_pretrain_epochs:
                    print(f"Epoch {epoch+1}/{args.student_pretrain_epochs}: Loss = {avg_loss:.6f}")
    
    else:
        # 原始模式：使用DataLoader（可能较慢）
        print("⚠️ 使用原始数据集模式（可能较慢）")
        for epoch in range(args.student_pretrain_epochs):
            total_loss = 0.0
            n_batches = 0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{args.student_pretrain_epochs}")):
                if isinstance(batch, dict):
                    clinical_feat = batch.get('clinical_features')
                if clinical_feat is None:
                    continue
                
                # 从clinical_feat构建clinical_data
                batch_size = clinical_feat.size(0)
                clinical_data = {
                    'hpv': [1 if clinical_feat[i, 1].item() > 0.5 else 0 for i in range(batch_size)],
                    'tct': [],
                    'age': [int(clinical_feat[i, 0].item() * 100) for i in range(batch_size)]
                }
                tct_categories = ['ASC-US', 'LSIL', 'HSIL', 'NILM', 'OTHER']
                for i in range(batch_size):
                    tct_onehot = clinical_feat[i, 2:7]
                    tct_idx = tct_onehot.argmax().item()
                    clinical_data['tct'].append(tct_categories[tct_idx])
                
                # 构建临床向量
                try:
                    clinical_vec = build_clinical_vector(clinical_data, device=device)
                except Exception as e:
                    print(f"Warning: Failed to build clinical vector: {e}")
                    continue
                
                # 获取VLM特征（使用batch索引作为patient_id的fallback）
                batch_vlm_feats = []
                valid_indices = []
                start_idx = batch_idx * args.batch_size
                
                for i in range(batch_size):
                    # 尝试多种patient_id格式
                    pid_candidates = [
                        f'sample_{start_idx + i}',
                        f'batch_{batch_idx}_{i}',
                        str(start_idx + i)
                    ]
                    
                    vlm_feat = None
                    for pid in pid_candidates:
                        if pid in vlm_features:
                            vlm_feat = vlm_features[pid]
                            break
                    
                    if vlm_feat is not None:
                        batch_vlm_feats.append(vlm_feat)
                        valid_indices.append(i)
                
                if len(batch_vlm_feats) == 0:
                    continue
                
                # 过滤有效的临床向量
                clinical_vec = clinical_vec[valid_indices]
                
                # VLM特征
                vlm_feat_tensor = torch.from_numpy(np.array(batch_vlm_feats)).to(device)  # [B, 1536]
                vlm_feat_proj = vlm_proj(vlm_feat_tensor)  # [B, 768]
                
                # Student Prior输出
                z_sem = student_prior(clinical_vec)  # [B, 768]
                
                # 计算损失
                loss = criterion(z_sem, vlm_feat_proj)
                
                # 反向传播
                optimizer.zero_grad()
                optimizer_proj.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_proj.step()
                
                total_loss += loss.item()
                n_batches += 1
        
    
    print(f"✅ Student Prior预训练完成！")
    
    # 保存模型
    checkpoint_path = args.output_dir / 'student_prior_pretrained.pth'
    torch.save({
        'student_prior': student_prior.state_dict(),
        'vlm_proj': vlm_proj.state_dict(),
    }, checkpoint_path)
    print(f"💾 保存预训练模型: {checkpoint_path}")
    
    return student_prior


def train_epoch(model, train_loader, criterion, optimizer, scaler, args, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_ot_loss = 0.0
    total_consist_loss = 0.0
    total_adv_loss = 0.0
    correct = 0
    total = 0
    
    print(f"\n🔍 开始训练循环，总batch数: {len(train_loader)}")
    print(f"   数据加载器类型: {type(train_loader)}")
    print(f"   开始迭代数据加载器...")
    
    import time
    loop_start = time.time()
    
    try:
        pbar = tqdm(train_loader, desc='Training')
        print(f"   ✅ tqdm进度条创建成功")
        
        # 尝试获取第一个batch
        print(f"   ⏱️ 开始获取第一个batch...")
        iter_start = time.time()
        
        for batch_idx, batch in enumerate(pbar):
            if batch_idx == 0:
                iter_time = time.time() - iter_start
                print(f"   ✅ 第一个batch获取成功，耗时: {iter_time:.2f}秒")
            
            # 添加调试信息（第一个batch）
            if batch_idx == 0:
                print(f"\n🔍 调试: 处理第一个batch...")
                print(f"   Batch类型: {type(batch)}")
                if isinstance(batch, dict):
                    print(f"   Batch keys: {list(batch.keys())}")
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            print(f"   {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                        else:
                            print(f"   {k}: type={type(v)}")
                start_time = time.time()
                print(f"   ⏱️ 开始处理batch {batch_idx}...")
            
            # 解析batch
            if isinstance(batch, dict):
                if batch_idx == 0:
                    parse_start = time.time()
                    print(f"   ⏱️ 开始解析batch数据...")
                
                oct_feat = batch['oct_features'].to(device)
                colpo_feat = batch['colposcopy_features'].to(device)
                clinical_feat = batch['clinical_features'].to(device)
                labels = batch['label'].to(device)
                center_labels = batch.get('center_id', torch.zeros(len(labels), dtype=torch.long)).to(device)
                
                # 从clinical_feat构建clinical_data
                batch_size = clinical_feat.size(0)
                clinical_data = {
                    'hpv': [1 if clinical_feat[i, 1].item() > 0.5 else 0 for i in range(batch_size)],
                    'tct': [],
                    'age': [int(clinical_feat[i, 0].item() * 100) for i in range(batch_size)]
                }
                tct_categories = ['ASC-US', 'LSIL', 'HSIL', 'NILM', 'OTHER']
                for i in range(batch_size):
                    tct_onehot = clinical_feat[i, 2:7]
                    tct_idx = tct_onehot.argmax().item()
                    clinical_data['tct'].append(tct_categories[tct_idx])
                
                if batch_idx == 0:
                    parse_time = time.time() - parse_start
                    print(f"   ✅ 数据解析完成，耗时: {parse_time:.2f}秒")
                    print(f"   ⏱️ 开始移动到GPU...")
                    move_start = time.time()
            else:
                if batch_idx == 0:
                    print(f"   ⚠️ Batch不是字典类型，跳过")
                continue
            
            # 移动到GPU
            if batch_idx == 0:
                if isinstance(batch, dict):
                    move_time = time.time() - move_start
                    print(f"   ✅ 数据移动到GPU完成，耗时: {move_time:.2f}秒")
                print(f"   ⏱️ 开始前向传播...")
                forward_start = time.time()
            
            optimizer.zero_grad()
            
            with autocast():
                if batch_idx == 0:
                    print(f"   ⏱️ 调用model.forward...")
                    model_start = time.time()
                
                outputs = model(
                    oct_features=oct_feat,
                    colpo_features=colpo_feat,
                    clinical_features=clinical_feat,
                    clinical_data=clinical_data,
                    center_labels=center_labels,
                    return_loss_components=True,
                    use_counterfactual=True
                )
                
                if batch_idx == 0:
                    model_time = time.time() - model_start
                    print(f"   ✅ 模型前向传播完成，耗时: {model_time:.2f}秒")
                    print(f"   ⏱️ 计算损失...")
                    loss_start = time.time()
                
                logits = outputs['logits']
                cls_loss = criterion(logits, labels)
                
                if batch_idx == 0:
                    loss_time = time.time() - loss_start
                    print(f"   ✅ 分类损失计算完成，耗时: {loss_time:.2f}秒")
                
                loss_components = outputs['loss_components']
                ot_loss = loss_components['L_ot']
                consist_loss = loss_components['L_consist']
                adv_loss = loss_components['L_adv']
                
                # 检查损失是否有nan或inf
                if torch.isnan(cls_loss) or torch.isinf(cls_loss):
                    cls_loss = torch.tensor(0.0, device=cls_loss.device, requires_grad=True)
                if torch.isnan(ot_loss) or torch.isinf(ot_loss):
                    ot_loss = torch.tensor(0.0, device=ot_loss.device, requires_grad=True)
                if torch.isnan(consist_loss) or torch.isinf(consist_loss):
                    consist_loss = torch.tensor(0.0, device=consist_loss.device, requires_grad=True)
                if torch.isnan(adv_loss) or torch.isinf(adv_loss):
                    adv_loss = torch.tensor(0.0, device=adv_loss.device, requires_grad=True)
                
                # 总损失
                total_loss_batch = (
                    args.lambda_cls * cls_loss +
                    args.lambda_ot * ot_loss +
                    args.lambda_consist * consist_loss +
                    args.lambda_adv * adv_loss
                )
                
                if torch.isnan(total_loss_batch) or torch.isinf(total_loss_batch):
                    total_loss_batch = args.lambda_cls * cls_loss
            
            if batch_idx == 0:
                forward_time = time.time() - forward_start
                print(f"   ✅ 前向传播总耗时: {forward_time:.2f}秒")
                print(f"   ⏱️ 开始反向传播...")
                backward_start = time.time()
            
            # 检查损失是否有效
            if torch.isnan(total_loss_batch) or torch.isinf(total_loss_batch):
                if batch_idx == 0:
                    print(f"   ⚠️ 损失无效，跳过batch")
                continue
            
            scaler.scale(total_loss_batch).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                optimizer.zero_grad()
                continue
            
            scaler.step(optimizer)
            scaler.update()
            
            if batch_idx == 0:
                backward_time = time.time() - backward_start
                total_time = time.time() - start_time
                print(f"   ✅ 反向传播耗时: {backward_time:.2f}秒")
                print(f"   ✅ 总耗时: {total_time:.2f}秒")
                print(f"   ✅ 第一个batch完成！\n")
            
            # 统计
            total_loss += total_loss_batch.item()
            total_cls_loss += cls_loss.item()
            total_ot_loss += ot_loss.item()
            total_consist_loss += consist_loss.item()
            total_adv_loss += adv_loss.item()
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'Cls': f'{cls_loss.item():.4f}',
                'OT': f'{ot_loss.item():.4f}',
                'Consist': f'{consist_loss.item():.4f}',
                'Adv': f'{adv_loss.item():.4f}'
            })
    except Exception as e:
        print(f"\n❌ 训练循环异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    return {
        'loss': total_loss / len(train_loader),
        'cls_loss': total_cls_loss / len(train_loader),
        'ot_loss': total_ot_loss / len(train_loader),
        'consist_loss': total_consist_loss / len(train_loader),
        'adv_loss': total_adv_loss / len(train_loader),
        'acc': 100. * correct / total
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
        for batch in tqdm(val_loader, desc='Validating'):
            if isinstance(batch, dict):
                oct_feat = batch['oct_features'].to(device)
                colpo_feat = batch['colposcopy_features'].to(device)
                clinical_feat = batch['clinical_features'].to(device)
                labels = batch['label'].to(device)
                
                # 从clinical_feat构建clinical_data
                batch_size = clinical_feat.size(0)
                clinical_data = {
                    'hpv': [1 if clinical_feat[i, 1].item() > 0.5 else 0 for i in range(batch_size)],
                    'tct': [],
                    'age': [int(clinical_feat[i, 0].item() * 100) for i in range(batch_size)]
                }
                tct_categories = ['ASC-US', 'LSIL', 'HSIL', 'NILM', 'OTHER']
                for i in range(batch_size):
                    tct_onehot = clinical_feat[i, 2:7]
                    tct_idx = tct_onehot.argmax().item()
                    clinical_data['tct'].append(tct_categories[tct_idx])
            else:
                continue
            
            with autocast():
                outputs = model(
                    oct_features=oct_feat,
                    colpo_features=colpo_feat,
                    clinical_features=clinical_feat,
                    clinical_data=clinical_data,
                    return_loss_components=False,
                    use_counterfactual=False
                )
                logits = outputs['logits']
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    try:
        if len(set(all_labels)) < 2:
            auc = 0.0
        else:
            auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return {
        'loss': total_loss / len(val_loader),
        'acc': 100. * correct / total,
        'auc': auc
    }


def main():
    """主函数"""
    args = BioCOTArgs()
    device = torch.device(args.device)
    
    print("📂 加载数据集...")
    # 创建args对象（数据集需要）
    class DummyArgs:
        def __init__(self):
            self.data_path = args.data_root
            self.input_size = 224
            self.oct_num_frames = 48
            self.oct_points = 12
            self.oct_frames_per_point = 10
            self.oct_cache_dir = None
            self.use_pretrained_backbones = True
    
    dummy_args = DummyArgs()
    
    # 直接使用EnhancedMultimodalCervicalDataset（避免timm依赖问题）
    print("   使用EnhancedMultimodalCervicalDataset直接加载...")
    train_dataset = EnhancedMultimodalCervicalDataset(
        root=Path(args.data_root) / 'internal_train' / 'train',
        is_train='train',
        args=dummy_args,
        use_enhanced_oct=True
    )
    val_dataset = EnhancedMultimodalCervicalDataset(
        root=Path(args.data_root) / 'internal_train' / 'val',
        is_train='val',
        args=dummy_args,
        use_enhanced_oct=True
    )
    print(f"✅ 数据集加载完成: 训练集 {len(train_dataset)} 个样本, 验证集 {len(val_dataset)} 个样本")
    
    # 优化数据加载器：单进程模式，避免多进程阻塞（OCT加载慢）
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,  # 单进程模式，避免多进程导致的OCT加载问题
        pin_memory=False,  # 单进程不需要pin_memory
        prefetch_factor=None,  # num_workers=0时不需要prefetch
        persistent_workers=False,  # num_workers=0时不需要persistent
        timeout=0  # num_workers=0时timeout必须为0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    print(f"✅ 数据加载器创建完成: train_batches={len(train_loader)}, val_batches={len(val_loader)}")
    
    print("🏗️ 创建Bio-COT模型...")
    model = BioCOTModel(
        embed_dim=768,
        num_classes=2,
        num_centers=5,
        input_dim=512
    ).to(device)
    
    # 预训练Student Prior（如果启用且VLM特征缓存存在）
    vlm_cache_path = Path(args.data_root) / 'vlm_features_cache' / 'train_vlm_features.npy'
    if args.pretrain_student_prior and vlm_cache_path.exists():
        print(f"🔄 发现VLM特征缓存: {vlm_cache_path}")
        print("   开始预训练Student Prior...")
        args.vlm_features_cache = vlm_cache_path  # 更新路径
        student_prior = pretrain_student_prior(args, train_dataset, device)
        if student_prior is not None:
            model.student_prior.load_state_dict(student_prior.state_dict())
            print("✅ 加载预训练的Student Prior")
        else:
            print("⚠️ Student Prior预训练失败，使用随机初始化")
    else:
        if args.pretrain_student_prior:
            print(f"⚠️ VLM特征缓存不存在: {vlm_cache_path}")
            print("   跳过Student Prior预训练，使用随机初始化")
            print("   提示：可以先用 extract_vlm_features.py 提取VLM特征，然后重新训练")
        else:
            print("ℹ️ Student Prior预训练已禁用，使用随机初始化")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler()
    
    # 学习率调度器
    warmup_scheduler = LinearLR(optimizer, start_factor=args.warmup_lr/args.learning_rate, 
                                end_factor=1.0, total_iters=len(train_loader) * args.warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=(args.num_epochs - args.warmup_epochs) * len(train_loader),
                                       eta_min=args.min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler],
                             milestones=[args.warmup_epochs * len(train_loader)])
    
    best_auc = 0.0
    
    print("🚀 开始训练...")
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 80)
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, args, device)
        scheduler.step()
        
        val_metrics = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, Train Acc={train_metrics['acc']:.2f}%, "
              f"Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['acc']:.2f}%, Val AUC={val_metrics['auc']:.4f}")
        
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(model.state_dict(), args.output_dir / 'best_model.pth')
            print(f"✅ New best model saved! AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()

