#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试脚本：验证所有模块是否正常工作
"""

import sys
from pathlib import Path
import torch
import numpy as np

# 添加项目路径
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("Bio-COT 3.0 快速测试")
print("=" * 80)

# 测试1: VisualNoteLayer
print("\n[测试1] VisualNoteLayer...")
try:
    from models.visual_notes import VisualNoteLayer
    
    B, N, D = 4, 196, 768
    layer = VisualNoteLayer(img_dim=D, text_dim=D, hidden_dim=256)
    img_feats = torch.randn(B, N, D)
    text_feats = torch.randn(B, D)
    
    img_focused, attn_map = layer(img_feats, text_feats, beta=0.1)
    
    assert img_focused.shape == (B, N, D), f"img_focused形状错误: {img_focused.shape}"
    assert attn_map.shape == (B, N, 1), f"attn_map形状错误: {attn_map.shape}"
    
    print(f"  ✅ VisualNoteLayer测试通过")
    print(f"     输入: img_feats {img_feats.shape}, text_feats {text_feats.shape}")
    print(f"     输出: img_focused {img_focused.shape}, attn_map {attn_map.shape}")
except Exception as e:
    print(f"  ❌ VisualNoteLayer测试失败: {e}")
    sys.exit(1)

# 测试2: VisualNotesModule (Warm-up)
print("\n[测试2] VisualNotesModule (Warm-up)...")
try:
    from models.visual_notes import VisualNotesModule
    
    module = VisualNotesModule(img_dim=768, text_dim=768, hidden_dim=256, warmup_epochs=5)
    
    # 测试不同epoch的Beta值
    for epoch in [0, 5, 10, 20, 30]:
        module.set_epoch(epoch)
        beta = module.get_beta()
        print(f"     Epoch {epoch}: Beta = {beta:.3f}")
    
    print(f"  ✅ VisualNotesModule测试通过")
except Exception as e:
    print(f"  ❌ VisualNotesModule测试失败: {e}")
    sys.exit(1)

# 测试3: Bio-COT v3模型
print("\n[测试3] Bio-COT v3模型...")
try:
    from models.bio_cot_v3 import BioCOT_v3
    
    model = BioCOT_v3(
        embed_dim=768,
        num_classes=2,
        num_centers=5,
        input_dim=768,
        use_visual_notes=True,
        use_ot=True,
        use_dual=True,
        use_cross_attn=True
    )
    
    B, N = 4, 196
    f_oct = torch.randn(B, N, 768)
    f_colpo = torch.randn(B, N, 768)
    note_embeds = torch.randn(B, 768)
    center_labels = torch.randint(0, 5, (B,))
    
    outputs = model(
        f_oct=f_oct,
        f_colpo=f_colpo,
        note_embeds=note_embeds,
        center_labels=center_labels,
        return_loss_components=True,
        current_beta=0.1
    )
    
    assert 'pred' in outputs, "缺少pred键"
    assert 'z_causal' in outputs, "缺少z_causal键"
    assert 'z_sem' in outputs, "缺少z_sem键"
    assert 'attn_maps' in outputs, "缺少attn_maps键"
    assert 'loss_components' in outputs, "缺少loss_components键"
    
    print(f"  ✅ Bio-COT v3模型测试通过")
    print(f"     输出键: {list(outputs.keys())}")
    print(f"     pred形状: {outputs['pred'].shape}")
    print(f"     损失组件: {list(outputs['loss_components'].keys())}")
except Exception as e:
    print(f"  ❌ Bio-COT v3模型测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: 稀疏性损失
print("\n[测试4] 稀疏性损失...")
try:
    from models.bio_cot_v3 import sparse_loss
    
    attn_maps = [
        torch.randn(4, 196, 1),
        torch.randn(4, 196, 1)
    ]
    
    loss = sparse_loss(attn_maps)
    
    assert loss.item() > 0, "稀疏性损失应该大于0"
    
    print(f"  ✅ 稀疏性损失测试通过")
    print(f"     损失值: {loss.item():.6f}")
except Exception as e:
    print(f"  ❌ 稀疏性损失测试失败: {e}")
    sys.exit(1)

# 测试5: 动态Beta策略
print("\n[测试5] 动态Beta策略...")
try:
    from training.train_bio_cot_v3 import get_dynamic_beta
    
    for epoch in [0, 5, 10, 20, 30]:
        beta = get_dynamic_beta(epoch, max_epochs=100)
        print(f"     Epoch {epoch}: Beta = {beta:.3f}")
    
    # 验证策略
    assert get_dynamic_beta(0) == 1.0, "Epoch 0的Beta应该是1.0"
    assert get_dynamic_beta(5) == 1.0, "Epoch 5的Beta应该是1.0"
    assert get_dynamic_beta(20) == 0.1, "Epoch 20的Beta应该是0.1"
    assert get_dynamic_beta(30) == 0.1, "Epoch 30的Beta应该是0.1"
    
    print(f"  ✅ 动态Beta策略测试通过")
except Exception as e:
    print(f"  ❌ 动态Beta策略测试失败: {e}")
    sys.exit(1)

# 测试6: 配置文件
print("\n[测试6] 配置文件...")
try:
    from config import BioCOT_v3_Config
    
    config = BioCOT_v3_Config()
    
    assert config.embed_dim == 768, "embed_dim应该是768"
    assert config.use_visual_notes == True, "use_visual_notes应该是True"
    assert config.lambda_sparse > 0, "lambda_sparse应该大于0"
    
    print(f"  ✅ 配置文件测试通过")
    print(f"     embed_dim: {config.embed_dim}")
    print(f"     use_visual_notes: {config.use_visual_notes}")
    print(f"     lambda_sparse: {config.lambda_sparse}")
except Exception as e:
    print(f"  ❌ 配置文件测试失败: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ 所有测试通过！可以开始训练了。")
print("=" * 80)
print("\n下一步：")
print("1. 运行: python knowledge_base/build_knowledge_base.py")
print("2. 运行: python knowledge_base/generate_knowledge_notes.py --csv_paths ...")
print("3. 运行: python training/train_bio_cot_v3.py")

