#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试Bio-COT代码逻辑（不依赖VLM特征）
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.bida.prior_net import StudentPriorNet, build_clinical_vector
from src.models.bida.losses import SinkhornDistance, CounterfactualConsistencyLoss, AdversarialLoss
from src.models.bida.memory_bank import NoiseMemoryBank, CenterDiscriminator
from src.models.bida.bio_cot_model import BioCOTModel

def test_modules():
    """测试各个模块"""
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 1. 测试Student Prior
    print("\n1. 测试Student Prior网络...")
    student_prior = StudentPriorNet(input_dim=7, output_dim=768).to(device)
    clinical_data = {
        'hpv': [1, 0, 1],
        'tct': ['NILM', 'ASC-US', 'LSIL'],
        'age': [45, 50, 35]
    }
    clinical_vec = build_clinical_vector(clinical_data, device=device)
    z_sem = student_prior(clinical_vec)
    print(f"   ✅ Student Prior输出形状: {z_sem.shape}")
    
    # 2. 测试Sinkhorn Loss
    print("\n2. 测试Sinkhorn OT Loss...")
    sinkhorn_loss = SinkhornDistance(eps=0.1, max_iter=10).to(device)
    z_causal = torch.randn(3, 768).to(device)
    loss_ot = sinkhorn_loss(z_causal, z_sem)
    print(f"   ✅ Sinkhorn Loss: {loss_ot.item():.4f}")
    
    # 3. 测试Memory Bank
    print("\n3. 测试Memory Bank...")
    memory_bank = NoiseMemoryBank(num_centers=5, feat_dim=768, capacity=10).to(device)
    z_noise = torch.randn(3, 768).to(device)
    center_ids = torch.tensor([0, 1, 2]).to(device)
    memory_bank.update(z_noise, center_ids)
    z_noise_cf = memory_bank.get_counterfactual_noise(center_ids)
    print(f"   ✅ Memory Bank输出形状: {z_noise_cf.shape}")
    
    # 4. 测试Bio-COT模型
    print("\n4. 测试Bio-COT模型...")
    model = BioCOTModel(embed_dim=768, num_classes=2, num_centers=5, input_dim=512).to(device)
    oct_feat = torch.randn(3, 512).to(device)
    colpo_feat = torch.randn(3, 512).to(device)
    clinical_feat = torch.randn(3, 7).to(device)
    
    outputs = model(
        oct_features=oct_feat,
        colpo_features=colpo_feat,
        clinical_features=clinical_feat,
        clinical_data=clinical_data,
        center_labels=center_ids,
        return_loss_components=True,
        use_counterfactual=True
    )
    print(f"   ✅ 模型输出keys: {outputs.keys()}")
    print(f"   ✅ Logits形状: {outputs['logits'].shape}")
    print(f"   ✅ 损失组件: {list(outputs['loss_components'].keys())}")
    
    # 5. 测试损失计算
    print("\n5. 测试损失计算...")
    logits = outputs['logits']
    labels = torch.tensor([0, 1, 0]).to(device)
    cls_loss = nn.CrossEntropyLoss()(logits, labels)
    
    loss_components = outputs['loss_components']
    total_loss = (
        1.0 * cls_loss +
        1.0 * loss_components['L_ot'] +
        2.0 * loss_components['L_consist'] +
        0.5 * loss_components['L_adv']
    )
    print(f"   ✅ 总损失: {total_loss.item():.4f}")
    print(f"      - 分类损失: {cls_loss.item():.4f}")
    print(f"      - OT损失: {loss_components['L_ot'].item():.4f}")
    print(f"      - 一致性损失: {loss_components['L_consist'].item():.4f}")
    print(f"      - 对抗损失: {loss_components['L_adv'].item():.4f}")
    
    print("\n✅ 所有模块测试通过！")
    return True

if __name__ == '__main__':
    try:
        test_modules()
        print("\n🎉 Bio-COT代码逻辑验证成功！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

