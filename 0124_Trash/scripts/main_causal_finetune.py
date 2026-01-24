import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import models_causal_gnn
from util.datasets import build_dataset
from engine_finetune import train_one_epoch, evaluate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import timm
sys.stdout.reconfigure(line_buffering=True)
sys.path.append('.')
# from llm_integration import MedicalReportGenerator
from visualize_causal_graph import visualize_causal_graph_networkx

def get_args_parser():
    parser = argparse.ArgumentParser('Causal Multimodal Cervical Cancer Detection', add_help=False)
    # Basic training parameters
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--start_epoch', default=0, type=int, help='Start epoch')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--data_path', default='./multi_dataset/', type=str, help='Dataset path')
    
    # Causal GNN parameters
    parser.add_argument('--causal_dim', default=64, type=int, help='Causal factor dimension')
    parser.add_argument('--causal_consistency_weight', default=1.0, type=float, help='Causal consistency loss weight')
    parser.add_argument('--intervention_weight', default=0.5, type=float, help='Intervention loss weight')
    parser.add_argument('--counterfactual_weight', default=0.3, type=float, help='Counterfactual loss weight')
    
    # Arguments from your command
    parser.add_argument('--savemodel', action='store_true', help='Save the model')
    parser.add_argument('--global_pool', action='store_true', help='Use global pooling')
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    # 新增：支持backbone和embed_dim选择
    parser.add_argument('--backbone', default='vit_base_patch16_224', type=str, help='Backbone model name (vit_base_patch16_224, vit_large_patch16_224, biomedclip, clip)')
    parser.add_argument('--embed_dim', default=1280, type=int, help='Embedding dimension for backbone output')
    parser.add_argument('--blr', type=float, default=5e-3, help='Base learning rate')
    parser.add_argument('--layer_decay', type=float, default=0.65, help='Layer-wise learning rate decay')
    parser.add_argument('--drop_path', type=float, default=0.2, help='Drop path rate')
    parser.add_argument('--nb_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--input_size', default=224, type=int, help='Input image size')
    parser.add_argument('--task', default='causal_cervical_cancer', type=str, help='Task name')
    
    # Augmentation parameters for build_transform
    parser.add_argument('--color_jitter', type=float, default=0.4, help='Color jitter factor')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', help='Auto-augmentation policy')
    parser.add_argument('--reprob', type=float, default=0.25, help='Random erasing probability')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erasing mode')
    parser.add_argument('--recount', type=int, default=1, help='Random erasing count')
    
    # Causal analysis parameters
    parser.add_argument('--enable_intervention', action='store_true', help='Enable causal intervention')
    parser.add_argument('--enable_counterfactual', action='store_true', help='Enable counterfactual generation')
    parser.add_argument('--save_causal_analysis', action='store_true', help='Save causal analysis results')
    
    # 新增：支持置信度温度
    parser.add_argument('--temperature', type=float, default=0.5, help='Softmax/temperature scaling temperature for confidence (lower=higher confidence)')
    
    return parser

def create_intervention_data(batch_size, causal_dim, device):
    """创建干预数据用于训练"""
    # 随机选择干预目标
    intervention_targets = torch.zeros(batch_size, causal_dim, device=device)
    for i in range(batch_size):
        num_targets = np.random.randint(1, min(4, causal_dim))
        targets = np.random.choice(causal_dim, num_targets, replace=False)
        intervention_targets[i, targets] = 1.0
    
    # 随机生成干预值
    intervention_values = torch.randn(batch_size, causal_dim, device=device) * 0.5
    
    return intervention_targets, intervention_values

def visualize_causal_graph(causal_adj, save_path=None):
    """可视化因果图"""
    plt.figure(figsize=(10, 8))
    
    # 计算平均因果邻接矩阵
    avg_adj = causal_adj.mean(dim=0).detach().cpu().numpy()
    
    # 创建热力图
    sns.heatmap(avg_adj, annot=True, cmap='RdBu_r', center=0, 
                square=True, cbar_kws={'shrink': 0.8})
    plt.title('Causal Graph Adjacency Matrix')
    plt.xlabel('Causal Factors')
    plt.ylabel('Causal Factors')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def analyze_causal_effects(model, outputs, targets, epoch, save_dir='./causal_analysis'):
    """分析因果效应"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取因果解释
    causal_effects = model.interpret_causal_effects(outputs)
    
    # 可视化直接效应
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    direct_effects = causal_effects['direct_effects'].detach().cpu().numpy()
    sns.heatmap(direct_effects, annot=True, cmap='Reds', square=True)
    plt.title('Direct Causal Effects')
    
    plt.subplot(1, 3, 2)
    indirect_effects = causal_effects['indirect_effects'].detach().cpu().numpy()
    sns.heatmap(indirect_effects, annot=True, cmap='Blues', square=True)
    plt.title('Indirect Causal Effects')
    
    plt.subplot(1, 3, 3)
    total_effects = causal_effects['total_effects'].detach().cpu().numpy()
    sns.heatmap(total_effects, annot=True, cmap='Purples', square=True)
    plt.title('Total Causal Effects')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/causal_effects_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存数值结果
    np.save(f'{save_dir}/causal_effects_epoch_{epoch}.npy', {
        'direct_effects': direct_effects,
        'indirect_effects': indirect_effects,
        'total_effects': total_effects,
        'causal_strength': causal_effects['causal_strength'].item()
    })

def ensure_tensor_on_device(label, device):
    import torch
    if not torch.is_tensor(label):
        label = torch.tensor(label)
    if label.ndim == 0:
        label = label.unsqueeze(0)
    return label.to(device)

def train_one_epoch_causal(model, criterion, data_loader, optimizer, epoch, args, device):
    """带因果推断的训练一个epoch"""
    model.train()
    total_loss = 0
    total_causal_loss = 0
    total_combined_loss = 0
    num_batches = len(data_loader)
    
    # 因果损失权重
    causal_weight = args.causal_consistency_weight if hasattr(args, 'causal_consistency_weight') else 0.1
    
    for batch_idx, (oct_img, col_img, clinical_data) in enumerate(data_loader):
        # 只在前几个batch或每50个batch打印一次调试信息
        verbose = batch_idx < 3 or batch_idx % 50 == 0
        
        if verbose:
            print(f'\n[Batch {batch_idx}] 开始...', flush=True)
        
        # 数据移动到设备
        oct_img = oct_img.to(device)
        col_img = col_img.to(device)
        clinical_data = clinical_data.to(device)
        
        # 从原始数据中获取标签
        start_idx = batch_idx * data_loader.batch_size
        end_idx = min(start_idx + data_loader.batch_size, len(data_loader.dataset))
        labels = data_loader.dataset.data_df['label'].values[start_idx:end_idx]
        targets = torch.tensor(labels, dtype=torch.long, device=device)
        
        if verbose:
            print(f'[Batch {batch_idx}] targets shape: {targets.shape}, dtype: {targets.dtype}', flush=True)
        
        # 数据预处理和NaN检查
        if torch.isnan(clinical_data).any():
            clinical_data = torch.nan_to_num(clinical_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        clinical_data_mean = clinical_data.mean(dim=0, keepdim=True)
        clinical_data_std = clinical_data.std(dim=0, keepdim=True) + 1e-6
        
        # 检查标准差是否为0或NaN
        if torch.isnan(clinical_data_std).any() or (clinical_data_std == 0).any():
            print("Warning: Zero or NaN std detected, using identity normalization", flush=True)
            clinical_data = torch.zeros_like(clinical_data)
        else:
            clinical_data = (clinical_data - clinical_data_mean) / clinical_data_std
            clinical_data = torch.clamp(clinical_data, min=-5, max=5)
        
        # 最终NaN检查
        if torch.isnan(clinical_data).any():
            print("Warning: NaN detected after normalization, replacing with zeros", flush=True)
            clinical_data = torch.nan_to_num(clinical_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        print('clinical_data after norm min:', clinical_data.min().item(), 'max:', clinical_data.max().item(), flush=True)
        # 确保targets为[B]形状的int64类型
        if targets.dtype != torch.int64:
            targets = targets.long()
        print(f'[Batch {batch_idx}] targets shape: {targets.shape}, dtype: {targets.dtype}', flush=True)
        print('targets unique:', torch.unique(targets), flush=True)
        
        # 创建干预数据（如果启用）
        intervention_targets = None
        intervention_values = None
        if args.enable_intervention and np.random.random() < 0.5:  # 提升到50%概率进行干预
            intervention_targets, intervention_values = create_intervention_data(
                oct_img.size(0), args.causal_dim, device
            )
        
        # 前向传播
        outputs = model(oct_img, col_img, clinical_data, 
                       intervention_targets, intervention_values,
                       args.enable_counterfactual)
        
        # 检查模型输出
        out = outputs['output']
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"[WARNING] Batch {batch_idx}: NaN/Inf detected in model output, skipping", flush=True)
            continue
            
        # 计算基础损失
        base_loss = criterion(out, targets)
        if torch.isnan(base_loss):
            print(f"[WARNING] Batch {batch_idx}: NaN in base loss, skipping", flush=True)
            continue
        
        total_loss += base_loss.item()
        
        if verbose:
            print(f'[Batch {batch_idx}] 基础损失: {base_loss.item():.4f}', flush=True)
        # 计算因果损失（通用聚合，自动包含新增项：filter_sparsity/decorr/cf_diversity等）
        losses = model.compute_losses(outputs, targets)
        causal_loss_tensor = torch.tensor(0.0, device=device)
        causal_loss_components = []
        for loss_name, loss_tensor in losses.items():
            if loss_name in ('task_loss', 'total_loss'):
                continue
            if torch.is_tensor(loss_tensor):
                causal_loss_tensor += loss_tensor
                try:
                    causal_loss_components.append(f"{loss_name}:{loss_tensor.item():.4f}")
                except Exception:
                    pass
        causal_loss_value = sum([float(x.split(':')[-1]) for x in causal_loss_components]) if causal_loss_components else 0.0
        total_causal_loss += causal_loss_value
        
        # 合并基础损失和因果损失
        combined_loss = base_loss + causal_weight * causal_loss_tensor
        total_combined_loss += combined_loss.item()
        
        if verbose and causal_loss_components:
            print(f'[Batch {batch_idx}] 因果损失组件: {", ".join(causal_loss_components)}', flush=True)
            print(f'[Batch {batch_idx}] 合并损失: {combined_loss.item():.4f} (基础:{base_loss.item():.4f} + 因果:{causal_loss_value:.4f})', flush=True)
        
        # 每10个batch显示一次进度摘要
        if batch_idx % 10 == 0:
            progress = (batch_idx + 1) / num_batches * 100
            print(f"", flush=True)
            print(f"{'='*60}", flush=True)
            print(f"📊 Epoch {epoch} 进度: {progress:.1f}% ({batch_idx+1}/{num_batches})", flush=True)
            print(f"🔥 当前损失: 基础={base_loss.item():.4f} | 因果={causal_loss_value:.4f} | 合并={combined_loss.item():.4f}", flush=True)
            print(f"📈 平均损失: 基础={total_loss/(batch_idx+1):.4f} | 因果={total_causal_loss/(batch_idx+1):.4f} | 合并={total_combined_loss/(batch_idx+1):.4f}", flush=True)
            if causal_loss_components:
                print(f"🧠 因果组件: {', '.join(causal_loss_components)}", flush=True)
            print(f"{'='*60}", flush=True)
            print(f"", flush=True)
        
        # 反向传播使用合并后的损失
        optimizer.zero_grad()
        combined_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 检查梯度是否有NaN
        nan_in_grad = any(param.grad is not None and torch.isnan(param.grad).any() 
                         for param in model.parameters())
        
        if nan_in_grad:
            if verbose:
                print(f"[WARNING] Batch {batch_idx}: NaN in gradients, skipping", flush=True)
            continue
            
        optimizer.step()
        
        if verbose:
            print(f'[Batch {batch_idx}] 优化完成', flush=True)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_causal_loss = total_causal_loss / num_batches if num_batches > 0 else 0.0
    avg_combined_loss = total_combined_loss / num_batches if num_batches > 0 else 0.0
    
    print(f"\n[Epoch 完成] 平均基础损失: {avg_loss:.4f}, 平均因果损失: {avg_causal_loss:.4f}, 平均合并损失: {avg_combined_loss:.4f}", flush=True)
    return avg_combined_loss, avg_causal_loss

def main(args):
    print("[INFO] 解析命令行参数完成，开始初始化设备...", flush=True)
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}", flush=True)
    print("[INFO] 开始加载数据集...", flush=True)
    # 加载数据集和类别权重
    dataset_train, class_weights = build_dataset(is_train='train', args=args)
    print("[INFO] 数据集加载完成，样本数:", len(dataset_train), flush=True)
    # 统计并打印类别分布
    labels = dataset_train.data_df['label'].values
    unique, counts = np.unique(labels, return_counts=True)
    print("[INFO] 类别分布:", dict(zip(unique, counts)), flush=True)
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 强制单进程，避免死锁
        pin_memory=True,
        drop_last=True
    )
    print("[INFO] DataLoader构建完成", flush=True)
    print("[INFO] 开始加载模型...", flush=True)
    print("[DEBUG] 即将加载ViT...", flush=True)
    oct_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    print("[DEBUG] oct_model加载完成", flush=True)
    col_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    print("[DEBUG] col_model加载完成", flush=True)
    import torch.nn as nn
    # 移除分类头，只保留特征提取部分
    oct_model.head = nn.Identity()
    col_model.head = nn.Identity()
    print(f"[INFO] ViT模型加载完成", flush=True)
    print(f"[INFO] Backbone: {args.backbone}, embed_dim: {args.embed_dim}", flush=True)
    print("[DEBUG] 即将构建CausalMultimodalTransformer...", flush=True)
    model = models_causal_gnn.CausalMultimodalTransformer(
        oct_model=oct_model,
        col_model=col_model,
        num_classes=args.nb_classes,
        embed_dim=args.embed_dim,
        causal_dim=args.causal_dim,
        dropout_rate=args.drop_path
    )
    print("[DEBUG] CausalMultimodalTransformer构建完成", flush=True)
    model.to(device)
    print("[INFO] 因果多模态模型构建完成", flush=True)
    # 设置损失函数和优化器
    # 初始化损失函数，自动加权类别
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print("[INFO] 优化器和学习率调度器初始化完成", flush=True)
    # 初始化 LLM 诊断报告生成器（暂时注释掉）
    # from path_config import get_model_path
    # report_generator = MedicalReportGenerator(llm_model=get_model_path('gpt2'), use_api=False, embed_dim=embed_dim)
    # 训练循环
    best_loss = float('inf')
    causal_analysis_interval = 10  # 每10个epoch进行一次因果分析
    print("[INFO] 开始训练主循环...", flush=True)
    for epoch in range(args.start_epoch, args.epochs):
        print(f"[INFO] Epoch {epoch}/{args.epochs - 1} 开始", flush=True)
        # 训练一个epoch
        avg_combined_loss, avg_causal_loss = train_one_epoch_causal(
            model, criterion, data_loader_train, optimizer, epoch, args, device
        )
        
        # 评估训练数据
        metrics = evaluate(model, criterion, data_loader_train, device)
        
        # 格式化输出每个epoch的结果
        print(f"\n", flush=True)
        print(f"🎯{'='*78}🎯", flush=True)
        print(f"🔥                    EPOCH {epoch:02d}/{args.epochs-1:02d} 完成                    🔥", flush=True)
        print(f"🎯{'='*78}🎯", flush=True)
        print(f"", flush=True)
        
        # 损失指标 - 格式化表格
        print(f"💰 损失指标:", flush=True)
        print(f"┌─────────────────────┬──────────┬──────────────────────────────┐", flush=True)
        print(f"│ 指标类型            │ 数值     │ 说明                         │", flush=True)
        print(f"├─────────────────────┼──────────┼──────────────────────────────┤", flush=True)
        print(f"│ 训练损失(合并)      │ {avg_combined_loss:8.4f} │ 基础分类+因果约束            │", flush=True)
        print(f"│ 因果损失            │ {avg_causal_loss:8.4f} │ 因果一致性+稀疏性+DAG        │", flush=True)
        print(f"│ 评估损失            │ {metrics['loss']:8.4f} │ 验证集损失                   │", flush=True)
        print(f"└─────────────────────┴──────────┴──────────────────────────────┘", flush=True)
        print(f"", flush=True)
        
        # 分类性能指标 - 格式化表格
        print(f"📊 分类性能指标:", flush=True)
        print(f"┌─────────────────────┬──────────┬─────────────────────────────────┐", flush=True)
        print(f"│ 指标名称            │ 数值     │ 性能评价                        │", flush=True)
        print(f"├─────────────────────┼──────────┼─────────────────────────────────┤", flush=True)
        
        # 准确率评价
        acc_eval = "🟢优秀" if metrics['accuracy'] > 0.9 else "🟡良好" if metrics['accuracy'] > 0.8 else "🟠一般" if metrics['accuracy'] > 0.7 else "🔴需改进"
        print(f"│ 准确率 (Accuracy)   │ {metrics['accuracy']:8.4f} │ {acc_eval:23s}     │", flush=True)
        
        # F1分数评价
        f1_eval = "🟢优秀" if metrics['f1'] > 0.9 else "🟡良好" if metrics['f1'] > 0.8 else "🟠一般" if metrics['f1'] > 0.7 else "🔴需改进"
        print(f"│ F1分数 (F1-Score)   │ {metrics['f1']:8.4f} │ {f1_eval:23s}     │", flush=True)
        
        # AUC评价
        auc_eval = "🟢优秀" if metrics['auc'] > 0.9 else "🟡良好" if metrics['auc'] > 0.8 else "🟠一般" if metrics['auc'] > 0.7 else "🔴需改进"
        print(f"│ AUC                 │ {metrics['auc']:8.4f} │ {auc_eval:23s}     │", flush=True)
        
        print(f"│ 精确率 (Precision)  │ {metrics['precision']:8.4f} │ 阳性预测准确性              │", flush=True)
        print(f"│ 召回率 (Recall)     │ {metrics['recall']:8.4f} │ 阳性样本识别率              │", flush=True)
        print(f"└─────────────────────┴──────────┴─────────────────────────────────┘", flush=True)
        print(f"", flush=True)
        
        # 医学指标 - 格式化表格
        print(f"🏥 医学诊断指标:", flush=True)
        print(f"┌─────────────────────┬──────────┬─────────────────────────────────┐", flush=True)
        print(f"│ 医学指标            │ 数值     │ 临床意义                        │", flush=True)
        print(f"├─────────────────────┼──────────┼─────────────────────────────────┤", flush=True)
        print(f"│ 敏感性 (Sensitivity)│ {metrics['sensitivity']:8.4f} │ 识别阳性病例能力            │", flush=True)
        print(f"│ 特异性 (Specificity)│ {metrics['specificity']:8.4f} │ 识别阴性病例能力            │", flush=True)
        print(f"│ 阳性预测值 (PPV)    │ {metrics['ppv']:8.4f} │ 阳性结果的可靠性            │", flush=True)
        print(f"│ 阴性预测值 (NPV)    │ {metrics['npv']:8.4f} │ 阴性结果的可靠性            │", flush=True)
        print(f"└─────────────────────┴──────────┴─────────────────────────────────┘", flush=True)
        print(f"🎯{'='*78}🎯\n", flush=True)
        
        # 学习率调度
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"当前学习率: {current_lr:.2e}", flush=True)
        
        # 保存模型
        if args.savemodel and ((epoch + 1) % 5 == 0 or epoch == args.epochs - 1):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': args
            }, f'causal_model_epoch_{epoch}.pth')
            print(f"Saved causal model at epoch {epoch}", flush=True)
        
        # 因果分析（定期进行）
        if args.save_causal_analysis and epoch % causal_analysis_interval == 0:
            model.eval()
            with torch.no_grad():
                # 获取一个批次的数据进行因果分析
                sample_batch = next(iter(data_loader_train))
                if len(sample_batch) == 3:
                    # 数据集返回 (oct_img, col_img, clinical_data)
                    oct_img, col_img, clinical_data = sample_batch
                    # 从 clinical_data 中分离 meta 和 temporal（最后一个元素是temporal）
                    meta = clinical_data[:, :-1]  # 除了最后一列的所有列
                    temporal = clinical_data[:, -1:]  # 最后一列
                    # 获取label需要从数据集中提取
                    label = torch.tensor([dataset_train.data_df.iloc[0]['label']] * clinical_data.shape[0])
                else:
                    # 兼容其他可能的返回格式
                    oct_img, col_img, meta, label, temporal = sample_batch[:5]
                
                # 确保所有输入都移动到正确的设备上
                oct_img = oct_img.to(device)
                col_img = col_img.to(device)
                clinical_data = clinical_data.to(device)
                label = ensure_tensor_on_device(label, device)
                
                outputs = model(oct_img, col_img, clinical_data)
                analyze_causal_effects(model, outputs, label, epoch)
                
                # 可视化因果图（热力图 + 带权重网络图）
                causal_adj = outputs['clinical_output']['causal_adj']
                visualize_causal_graph(causal_adj, f'./causal_analysis/causal_graph_epoch_{epoch}.png')
                visualize_causal_graph_networkx(causal_adj, f'./causal_analysis/causal_graph_network_epoch_{epoch}.png', title=f"Causal Graph (Epoch {epoch})")
                # 保存数值化邻接矩阵（按batch均值）以便后处理提取Top-K边
                try:
                    import numpy as np
                    avg_adj_np = causal_adj.mean(dim=0).detach().cpu().numpy()
                    np.save(f'./causal_analysis/causal_adj_epoch_{epoch}.npy', avg_adj_np)
                except Exception as save_adj_e:
                    print(f"[WARNING] 保存因果邻接矩阵失败: {save_adj_e}", flush=True)
        
        # 反事实解释（在训练结束时）
        if args.enable_counterfactual and epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(data_loader_train))
                if len(sample_batch) == 3:
                    # 数据集返回 (oct_img, col_img, clinical_data)
                    oct_img, col_img, clinical_data = sample_batch
                    # 从 clinical_data 中分离 meta 和 temporal（最后一个元素是temporal）
                    meta = clinical_data[:, :-1]  # 除了最后一列的所有列
                    temporal = clinical_data[:, -1:]  # 最后一列
                    # 获取label需要从数据集中提取
                    label = torch.tensor([dataset_train.data_df.iloc[0]['label']] * clinical_data.shape[0])
                else:
                    # 兼容其他可能的返回格式
                    oct_img, col_img, meta, label, temporal = sample_batch[:5]
                
                # 确保所有输入都移动到正确的设备上
                oct_img = oct_img.to(device)
                col_img = col_img.to(device)
                clinical_data = clinical_data.to(device)
                label = ensure_tensor_on_device(label, device)
                
                cf_explanations = model.generate_counterfactual_explanations(
                    oct_img, col_img, clinical_data, label
                )
                
                print("Counterfactual Analysis:", flush=True)
                print(f"Factual vs Counterfactual prediction difference: "
                      f"{cf_explanations['explanation_difference'].mean().item():.4f}", flush=True)
                print(f"Counterfactual confidence: "
                      f"{cf_explanations['counterfactual_confidence'].mean().item():.4f}", flush=True)
        
        # === 自动生成并保存诊断报告 ===
        try:
            model.eval()
            with torch.no_grad():
                # 取一个batch
                sample_batch = next(iter(data_loader_train))
                if len(sample_batch) == 3:
                    # 数据集返回 (oct_img, col_img, clinical_data)
                    oct_img, col_img, clinical_data = sample_batch
                    # 从 clinical_data 中分离 meta 和 temporal（最后一个元素是temporal）
                    meta = clinical_data[:, :-1]  # 除了最后一列的所有列
                    temporal = clinical_data[:, -1:]  # 最后一列
                    # 获取label需要从数据集中提取
                    label = torch.tensor([dataset_train.data_df.iloc[0]['label']] * clinical_data.shape[0])
                    patient_id_batch = ['sample_patient'] * clinical_data.shape[0]  # 默认patient_id
                else:
                    # 兼容其他可能的返回格式
                    sample_data = list(sample_batch) + [None] * (6 - len(sample_batch))  # 填充到6个元素
                    oct_img, col_img, meta, label, temporal, patient_id_batch = sample_data[:6]
                    if patient_id_batch is None:
                        patient_id_batch = ['sample_patient'] * oct_img.shape[0]
                
                # 全部转到 device 并 float 并 contiguous
                oct_img = oct_img.to(device).float().contiguous()
                # === col_img 终极自动加固 ===
                try:
                    col_img = col_img.to(device).float().contiguous()
                    # 只要不是4维，全部自动修正
                    if col_img.dim() == 5:
                        # [B, 3, C, H, W] -> [B*3, C, H, W]
                        col_img = col_img.view(-1, *col_img.shape[2:]).contiguous()
                    elif col_img.dim() == 4:
                        # [B, C, H, W]，假设B=3的情况，直接用
                        pass
                    elif col_img.dim() == 3:
                        # [C, H, W]，补batch
                        col_img = col_img.unsqueeze(0)
                    else:
                        print(f"[ERROR] col_img shape异常: {col_img.shape}, dtype: {col_img.dtype}, device: {col_img.device}, 跳过本batch", flush=True)
                        continue
                    # 兜底：如果还不是4维，自动报错并跳过
                    if col_img.dim() != 4 or col_img.shape[1] != 3:
                        print(f"[ERROR] col_img shape异常: {col_img.shape}, dtype: {col_img.dtype}, device: {col_img.device}, 跳过本batch", flush=True)
                        continue
                except Exception as colimg_e:
                    print(f"[ERROR] col_img 自动修正失败: {colimg_e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    print("[WARNING] 本batch自动跳过，继续主流程...", flush=True)
                    continue
                # 处理clinical_data
                if 'clinical_data' not in locals():
                    # 如果是从分离的meta和temporal构建clinical_data
                    meta = meta.to(device).float().contiguous()
                    temporal = temporal.to(device).float().contiguous()
                    clinical_data = torch.cat((meta, temporal), dim=1).to(device).float().contiguous()
                else:
                    # 如果已经有clinical_data，直接使用
                    clinical_data = clinical_data.to(device).float().contiguous()
                label = ensure_tensor_on_device(label, device)
                try:
                    # 调用模型获取输出
                    outputs = model(oct_img, col_img, clinical_data)
                except Exception as model_e:
                    print(f"[ERROR] 模型推理失败: {model_e}", flush=True)
                    print("oct_img shape:", oct_img.shape, oct_img.dtype, oct_img.device, flush=True)
                    print("col_img shape:", col_img.shape, col_img.dtype, col_img.device, flush=True)
                    print("clinical_data shape:", clinical_data.shape, clinical_data.dtype, clinical_data.device, flush=True)
                    import traceback
                    traceback.print_exc()
                    print("[WARNING] 本batch自动跳过，继续主流程...", flush=True)
                    continue
                # 获取模型特征
                features = outputs['fused_features'].detach().cpu() if 'fused_features' in outputs else torch.randn(1, 1024)
                # 构建临床数据字典
                clinical_dict = {
                    'age': meta[0, 0].item() if meta.size(1) > 0 else 50,
                    'gender': meta[0, 1].item() if meta.size(1) > 1 else 1,
                    'symptoms': meta[0, 2].item() if meta.size(1) > 2 else 0,
                    'temporal_info': temporal[0, 0].item() if temporal.size(1) > 0 else 0
                }
                # 计算置信度
                from llm_integration import ConfidenceCalculator
                from confidence_config import get_confidence_config
                
                # 使用配置文件
                confidence_config = get_confidence_config('temperature_scaled', temperature=args.temperature)
                confidence_calc = ConfidenceCalculator(
                    confidence_method=confidence_config.confidence_method,
                    confidence_threshold=confidence_config.confidence_threshold,
                    uncertainty_method=confidence_config.uncertainty_method,
                    temperature=args.temperature
                )
                
                # 获取logits用于置信度计算
                logits = outputs.get('logits', outputs.get('output'))
                if logits is not None:
                    confidence_result = confidence_calc.calculate_confidence(logits, num_classes=2)
                    confidence_stats = confidence_calc.get_confidence_stats(confidence_result)
                else:
                    confidence_result = {'confidence': [0.5] * features.shape[0]}
                    confidence_stats = {
                        'mean_confidence': 0.5,
                        'confident_ratio': 0.0,
                        'mean_uncertainty': 0.5
                    }
                # 构建图像分析结果
                image_analysis = {
                    'oct_analysis': f"OCT图像特征维度: {oct_img.shape}",
                    'colposcopy_analysis': f"阴道镜图像特征维度: {col_img.shape}",
                    'prediction': outputs['logits'].softmax(dim=1).argmax(dim=1).item() if 'logits' in outputs else 0,
                    'confidence': confidence_stats['mean_confidence'],
                    'uncertainty': confidence_stats['mean_uncertainty'],
                    'is_confident': confidence_stats['confident_ratio']
                }
                try:
                    # 生成并自动保存诊断报告，传递patient_id_batch和每个样本的置信度
                    report_generator(features, clinical_dict, image_analysis, patient_ids=patient_id_batch, confidence=confidence_result['confidence'])
                    print(f"[INFO] Epoch {epoch} 诊断报告生成完成", flush=True)
                except Exception as report_e:
                    print(f"[ERROR] 诊断报告生成失败: {report_e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    print("[WARNING] 本batch自动跳过，继续主流程...", flush=True)
                    continue
        except Exception as e:
            print(f"[WARNING] 诊断报告生成主流程异常: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    print("Training completed!", flush=True)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args) 