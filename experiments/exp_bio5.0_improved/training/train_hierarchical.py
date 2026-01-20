#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 5.0 Improved (HM-VR): 训练脚本

特性：
- Stage 1: Hierarchical mHC + Clinical Query Evolution (CoT)
- Stage 2: VLM Anchor + Visual Notes (SCG)
- 支持 --dry-run：不加载真实数据，只做前向检查
"""

from __future__ import annotations

import argparse
import os
import sys
import importlib
import importlib.util
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# add project root
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BioCOT_V5_Improved_Config
from datasets.multimodal_dataset import BioCOT_MultimodalDataset, DatasetArgs

# 🔥 消除 Tokenizers 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# 🔥 Label Smoothing Loss 实现
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=2, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = -1

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# 🔥 EMA (Exponential Moving Average) 实现
class EMA:
    """
    Exponential Moving Average for model weights.
    这是ViT训练的"核武器"，使用历史权重的平滑平均值，通常能带来2-3个点的稳定AUC提升。
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """注册所有参数的影子副本"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新EMA权重：shadow = decay * shadow + (1-decay) * param"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """将EMA权重应用到模型"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def seed_everything(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    config: BioCOT_V5_Improved_Config,
    epoch: int,
    max_batches: Optional[int] = None,
    ema: Optional[EMA] = None,  # 🔥 新增：EMA对象
) -> Dict[str, float]:
    model.train()
    model.set_epoch(epoch)
    
    # 🔥 使用Label Smoothing Loss：防止模型过度自信
    label_smoothing = getattr(config, 'label_smoothing', 0.0)
    if label_smoothing > 0:
        criterion = LabelSmoothingLoss(classes=2, smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 存储EMA对象供内部使用
    train_one_epoch._ema = ema

    losses: List[float] = []
    cls_losses: List[float] = []
    ot_losses: List[float] = []
    ortho_losses: List[float] = []
    noise_losses: List[float] = []  # 🔥 新增：Noise Loss 记录

    all_probs: List[float] = []
    all_labels: List[int] = []

    pbar = tqdm(loader, desc=f"Train {epoch}", dynamic_ncols=True)
    for bi, batch in enumerate(pbar):
        if max_batches is not None and bi >= max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        clinical = batch["clinical"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        image_names = batch["image_name"]
        clinical_info = batch.get("clinical_info_str", None)
        
        # 🔥 新增：获取 center_ids（用于 NoiseAwareMHC）
        center_ids = batch.get("center_idx", None)
        if center_ids is not None:
            center_ids = center_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            images=images,
            clinical_features=clinical,
            center_ids=center_ids,  # 🔥 传入 center_ids
            image_names=image_names,
            clinical_info=clinical_info,
            return_loss_components=True,
        )

        logits = outputs["logits"]
        loss_cls = criterion(logits, labels)
        loss = config.lambda_cls * loss_cls

        # LACT-like anchor loss
        loss_ot = torch.tensor(0.0, device=device)
        if config.lambda_ot > 0 and "z_anchor" in outputs:
            zc = nn.functional.normalize(outputs["z_causal"], dim=1)
            za = nn.functional.normalize(outputs["z_anchor"].detach(), dim=1)
            loss_ot = 1.0 - torch.sum(zc * za, dim=1).mean()
            loss = loss + config.lambda_ot * loss_ot

        # Orthogonality loss
        loss_ortho = torch.tensor(0.0, device=device)
        if config.lambda_ortho > 0 and "loss_components" in outputs:
            if "L_ortho" in outputs["loss_components"]:
                loss_ortho = outputs["loss_components"]["L_ortho"]
                loss = loss + config.lambda_ortho * loss_ortho
        
        # 🔥 新增：Noise Regularization Loss
        # 我们希望噪声门控是稀疏的（大部分区域应该是干净的），且能捕捉到特定模式
        loss_noise = torch.tensor(0.0, device=device)
        if config.lambda_noise > 0 and "noise_probs" in outputs:
            noise_probs = outputs["noise_probs"]  # list of [B, N, 1]
            for np_map in noise_probs:
                loss_noise = loss_noise + torch.mean(np_map)
            loss_noise = loss_noise / len(noise_probs) if noise_probs else loss_noise
            loss = loss + config.lambda_noise * loss_noise

        loss.backward()
        # 梯度裁剪：防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 🔥 EMA更新（在每个batch后）
        if hasattr(train_one_epoch, '_ema') and train_one_epoch._ema is not None:
            train_one_epoch._ema.update()

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        labs = labels.detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labs.tolist())

        losses.append(loss.item())
        cls_losses.append(loss_cls.item())
        ot_losses.append(loss_ot.item())
        ortho_losses.append(loss_ortho.item())
        noise_losses.append(loss_noise.item())  # 🔥 新增

        pbar.set_postfix(
            loss=np.mean(losses),
            cls=np.mean(cls_losses),
            ot=np.mean(ot_losses),
            ortho=np.mean(ortho_losses),
            noise=np.mean(noise_losses),  # 🔥 新增
        )

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    pred = (np.array(all_probs) >= 0.5).astype(int)
    acc = accuracy_score(all_labels, pred)
    f1 = f1_score(all_labels, pred)

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "cls_loss": float(np.mean(cls_losses)) if cls_losses else float("nan"),
        "ot_loss": float(np.mean(ot_losses)) if ot_losses else float("nan"),
        "ortho_loss": float(np.mean(ortho_losses)) if ortho_losses else float("nan"),
        "noise_loss": float(np.mean(noise_losses)) if noise_losses else float("nan"),  # 🔥 新增
        "auc": float(auc),
        "acc": float(acc),
        "f1": float(f1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: BioCOT_V5_Improved_Config,
    epoch: int,
    max_batches: Optional[int] = None,
    use_ema: bool = False,  # 🔥 新增：是否使用EMA模型
) -> Dict[str, float]:
    model.eval()
    
    # 🔥 如果使用EMA，先应用EMA权重
    if use_ema and hasattr(validate, '_ema') and validate._ema is not None:
        validate._ema.apply_shadow()
    model.set_epoch(epoch)
    criterion = nn.CrossEntropyLoss()

    losses: List[float] = []
    all_probs: List[float] = []
    all_labels: List[int] = []

    pbar = tqdm(loader, desc=f"Val {epoch}", dynamic_ncols=True)
    for bi, batch in enumerate(pbar):
        if max_batches is not None and bi >= max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        clinical = batch["clinical"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        image_names = batch["image_name"]
        clinical_info = batch.get("clinical_info_str", None)
        
        # 🔥 新增：获取 center_ids（用于 NoiseAwareMHC）
        center_ids = batch.get("center_idx", None)
        if center_ids is not None:
            center_ids = center_ids.to(device, non_blocking=True)

        outputs = model(
            images=images,
            clinical_features=clinical,
            center_ids=center_ids,  # 🔥 传入 center_ids
            image_names=image_names,
            clinical_info=clinical_info,
            return_loss_components=False,
        )
        logits = outputs["logits"]
        loss = criterion(logits, labels)
        losses.append(loss.item())

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        labs = labels.detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labs.tolist())

        pbar.set_postfix(loss=np.mean(losses))

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float("nan")
    pred = (np.array(all_probs) >= 0.5).astype(int)
    acc = accuracy_score(all_labels, pred)
    f1 = f1_score(all_labels, pred)
    
    # 🔥 验证结束后恢复原始权重（如果使用了EMA）
    if use_ema and hasattr(validate, '_ema') and validate._ema is not None:
        validate._ema.restore()
    
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "auc": float(auc),
        "acc": float(acc),
        "f1": float(f1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, default="full", choices=["full", "no_vlm", "no_notes", "no_dual"])
    parser.add_argument("--dry-run", action="store_true", help="只做前向检查，不加载真实数据")
    parser.add_argument("--max-batches", type=int, default=None, help="每个epoch最多跑多少个batch（调试用）")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置文件的epochs（调试用）")
    parser.add_argument("--run-id", type=str, default=None, help="指定本次run的ID（用于固定输出目录名）")
    args = parser.parse_args()

    config = BioCOT_V5_Improved_Config()
    if args.epochs is not None:
        config.epochs = int(args.epochs)
    if args.ablation == "no_vlm":
        config.use_vlm_anchor = False
    elif args.ablation == "no_notes":
        config.use_visual_notes = False
    elif args.ablation == "no_dual":
        config.use_dual = False
        config.lambda_ortho = 0.0

    seed_everything(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 输出目录：固定写到“方法目录”下，方便归档/复现实验
    exp_root = Path(__file__).resolve().parents[1]
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = exp_root / "runs" / f"hmvr_{run_id}"
    config.output_dir = str(run_root / "results")
    config.checkpoint_dir = str(run_root / "checkpoints")
    config.log_dir = str(run_root / "logs")
    config.ensure_dirs()

    # 保存 config 快照
    cfg_path = Path(config.log_dir) / "config.json"
    try:
        import json

        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    print("=" * 80)
    print("Bio-COT 5.5 Pro (HM-VR + Noise-Aware) 训练启动")
    print(f"device: {device}")
    print(f"ablation: {args.ablation}")
    print(f"config: {config}")
    print("=" * 80)

    # 兼容性处理：避免顶层 'models' 包名冲突（项目根目录/第三方库可能也存在 models）
    models_dir = exp_root / "models"
    pkg_name = "_hmvr_models"
    if pkg_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            pkg_name,
            str(models_dir / "__init__.py"),
            submodule_search_locations=[str(models_dir)],
        )
        assert spec and spec.loader
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = pkg
        spec.loader.exec_module(pkg)  # type: ignore[attr-defined]

    BioCOT_V5_Hierarchical = importlib.import_module(f"{pkg_name}.bio_cot_hierarchical").BioCOT_V5_Hierarchical

    model = BioCOT_V5_Hierarchical(config, num_centers=config.num_centers).to(device)  # 🔥 传入 num_centers
    
    # 🔥 [Fix 3] EMA初始化
    use_ema = getattr(config, 'use_ema', False)
    ema = None
    if use_ema:
        ema_decay = getattr(config, 'ema_decay', 0.999)
        ema = EMA(model, decay=ema_decay)
        print(f"📊 EMA已启用 (decay={ema_decay})")
    
    # 🔥 分层学习率：Backbone 学习率是 Head 的 1/10，保护预训练知识
    backbone_params = list(map(id, model.visual_encoder.parameters()))
    head_params = filter(lambda p: id(p) not in backbone_params, model.parameters())
    
    optimizer = optim.AdamW([
        {'params': head_params, 'lr': config.lr},
        {'params': model.visual_encoder.parameters(), 'lr': config.lr * 0.1}  # Backbone 学习率是 Head 的 1/10
    ], weight_decay=config.weight_decay)
    
    # 🔥 学习率调度器（ReduceLROnPlateau）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )

    if args.dry_run:
        print("⚠️ dry-run: 使用随机数据做结构验证")
        b = 4
        dummy_img = torch.randn(b, 3, 224, 224, device=device)
        dummy_clin = torch.randn(b, config.clinical_input_dim, device=device)
        dummy_names = ["dummy.jpg"] * b
        dummy_clin_str = ["HPV: positive, TCT: HSIL, Age: 40"] * b
        dummy_center_ids = torch.randint(0, config.num_centers, (b,), device=device)  # 🔥 新增：测试 center_ids
        out = model(
            images=dummy_img,
            clinical_features=dummy_clin,
            center_ids=dummy_center_ids,  # 🔥 传入 center_ids
            image_names=dummy_names,
            clinical_info=dummy_clin_str,
            return_loss_components=True,
        )
        print(f"✅ forward ok, logits={tuple(out['logits'].shape)}")
        return

    # resolve csv path
    train_csv = Path(config.data_root) / config.train_csv
    val_csv = Path(config.data_root) / config.val_csv

    # resolve VLM cache path relative to this experiment directory
    vlm_path = exp_root / config.vlm_json_path
    if not vlm_path.exists():
        # fallback to exp_bio5.0
        alt = exp_root.parent / "exp_bio5.0" / config.vlm_json_path
        if alt.exists():
            vlm_path = alt
            print(f"⚠️ 使用备用VLM缓存路径: {vlm_path}")
        else:
            print(f"⚠️ 未找到VLM缓存: {vlm_path}，将禁用VLM anchor")
            config.use_vlm_anchor = False

    train_ds = BioCOT_MultimodalDataset(
        DatasetArgs(
            csv_path=str(train_csv),
            vlm_json_path=str(vlm_path) if config.use_vlm_anchor else None,
            image_source=config.image_source,
            oct_num_frames=20,
            max_col_images=3,
        )
    )
    val_ds = BioCOT_MultimodalDataset(
        DatasetArgs(
            csv_path=str(val_csv),
            vlm_json_path=str(vlm_path) if config.use_vlm_anchor else None,
            image_source=config.image_source,
            oct_num_frames=20,
            max_col_images=3,
        )
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    best_auc = -1.0
    best_val_loss = float('inf')
    early_stop_patience = 10  # 验证loss连续10个epoch不下降就停止
    early_stop_counter = 0
    
    # 🔥 渐进式训练：前N个epoch冻结ViT骨干
    freeze_epochs = getattr(config, 'freeze_backbone_epochs', 0)
    if freeze_epochs > 0:
        print(f"🔒 前{freeze_epochs}个epoch将冻结ViT骨干，只训练新模块（mHC + Evolver）")
        model.freeze_backbone()
    
    print(f"🚀 [Bio-COT 5.5 Ultimate] Starting HM-VR Training with Anti-Overfitting Protocol")
    print(f"   - 架构瘦身: hidden_dim={config.hidden_dim}, mhc_latent_dim={config.mhc_latent_dim}")
    print(f"   - DropPath Rate: {getattr(config, 'drop_path_rate', 0.0)}")
    print(f"   - Dropout Rate: {getattr(config, 'dropout_rate', 0.3)}")
    print(f"   - Weight Decay: {config.weight_decay}")
    print(f"   - Label Smoothing: {getattr(config, 'label_smoothing', 0.0)}")
    print(f"   - EMA: {use_ema} (decay={getattr(config, 'ema_decay', 0.999) if use_ema else 'N/A'})")
    print(f"   - Freeze Epochs: {freeze_epochs}")
    
    for epoch in range(1, config.epochs + 1):
        # 🔥 在指定epoch解冻ViT
        if epoch == freeze_epochs + 1:
            print(f"🔓 Epoch {epoch}: 解冻ViT骨干，开始端到端训练")
            model.unfreeze_backbone()
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
            epoch=epoch,
            max_batches=args.max_batches,
            ema=ema,  # 🔥 传递EMA对象
        )
        va = validate(
            model=model,
            loader=val_loader,
            device=device,
            config=config,
            epoch=epoch,
            max_batches=args.max_batches,
            use_ema=use_ema,  # 🔥 传递EMA标志
        )
        # 存储EMA对象供validate使用
        validate._ema = ema

        # 🔥 更新学习率调度器（基于验证loss）
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(va['loss'])
        new_lr = optimizer.param_groups[0]['lr']
        lr_changed = "📉 LR降低" if new_lr < current_lr else ""

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={tr['loss']:.4f} auc={tr['auc']:.4f} | "
            f"val_loss={va['loss']:.4f} auc={va['auc']:.4f} "
            f"lr={current_lr:.2e} {lr_changed}"
        )

        # 🔥 保存最佳模型（基于验证AUC）
        if not np.isnan(va["auc"]) and va["auc"] > best_auc:
            best_auc = va["auc"]
            ckpt = Path(config.checkpoint_dir) / f"best_{run_id}.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            
            # 🔥 如果使用EMA，保存EMA权重而不是原始权重
            if use_ema and ema is not None:
                ema.apply_shadow()  # 临时应用EMA权重
                state_dict_to_save = model.state_dict()
                ema.restore()  # 恢复原始权重
            else:
                state_dict_to_save = model.state_dict()
            
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": state_dict_to_save,  # 🔥 保存EMA权重（如果启用）
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": asdict(config),
                    "best_auc": best_auc,
                    "val_loss": va['loss'],
                    "ema_state": ema.shadow if (use_ema and ema is not None) else None,  # 保存EMA影子权重
                },
                ckpt,
            )
            print(f"✅ 保存最佳模型: {ckpt} (AUC={best_auc:.4f}, val_loss={va['loss']:.4f})")

        # 🔥 早停机制（基于验证loss）
        if va['loss'] < best_val_loss:
            best_val_loss = va['loss']
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"⚠️ 早停触发！验证loss连续{early_stop_patience}个epoch未下降")
                print(f"   最佳验证AUC: {best_auc:.4f}, 最佳验证Loss: {best_val_loss:.4f}")
                print(f"   当前epoch: {epoch}/{config.epochs}")
                break


if __name__ == "__main__":
    main()


