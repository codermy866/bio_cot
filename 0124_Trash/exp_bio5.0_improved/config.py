from dataclasses import dataclass
import os

@dataclass
class BioCOT_V5_Improved_Config:
    # --- Experiment Setup ---
    project_name: str = "BioCOT_V5_HM_VR_Ultimate"
    version: str = "v5.5_EMA_Reduced"
    output_dir: str = "results/bio_cot_v5_pro_fix"
    seed: int = 42
    
    # --- Data ---
    data_root: str = "/data2/hmy/5Center_datas/5centers_multi_positive_sites_multimodal"
    train_csv: str = "internal_train/labels.csv"
    val_csv: str = "internal_val/labels.csv"
    vlm_json_path: str = "data/vlm_profiles_v1.json" 
    image_source: str = "colposcopy"  # colposcopy | oct_first
    
    # --- Model Architecture (瘦身版) ---
    vit_arch: str = 'vit_base_patch16_224'
    visual_dim: int = 768
    clinical_input_dim: int = 7  # 你的日志显示是 7
    
    # 🔥 [Fix 1] 降维打击：从 256 降至 128，大幅减少参数量
    hidden_dim: int = 128        
    mhc_latent_dim: int = 128    
    
    # 🔥 [Fix 2] 激进的正则化参数
    dropout_rate: float = 0.5      # 非常高的 Dropout (通常 0.5 是极限)
    drop_path_rate: float = 0.3    # ViT 的随机深度丢弃
    attention_dropout: float = 0.1 
    
    # Hierarchical Setup
    extract_layers: tuple = (2, 5, 8, 11) 
    sinkhorn_iters: int = 3
    mhc_epsilon: float = 0.05
    use_visual_notes: bool = True
    use_vlm_anchor: bool = True
    text_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    use_dual: bool = True
    
    # --- Training ---
    batch_size: int = 24
    num_workers: int = 4
    
    # 优化策略
    lr: float = 2e-5               # 保持较低学习率
    weight_decay: float = 0.05     # 强 L2 正则
    epochs: int = 100
    warmup_epochs: int = 5
    freeze_backbone_epochs: int = 10
    
    # 🔥 [Fix 3] EMA 设置 (关键)
    use_ema: bool = True           
    ema_decay: float = 0.999       # 越接近 1 模型越平滑
    
    # Loss Weights
    lambda_cls: float = 1.0
    lambda_ot: float = 1.0
    lambda_ortho: float = 0.5
    lambda_noise: float = 0.1
    
    label_smoothing: float = 0.1
    
    # Center-aware settings
    num_centers: int = 5
    
    # --- Output ---
    checkpoint_dir: str = "checkpoints/bio_cot_v5_pro_fix"
    log_dir: str = "logs/bio_cot_v5_pro_fix"
    
    def ensure_dirs(self) -> None:
        from pathlib import Path
        for d in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)
