#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型初始化工具
使用更好的初始化策略提升训练效果
"""

import torch
import torch.nn as nn
import math


def init_weights_xavier_uniform(m):
    """Xavier均匀初始化"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def init_weights_kaiming_normal(m):
    """Kaiming正态初始化（适用于ReLU/GELU）"""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def init_weights_orthogonal(m):
    """正交初始化（适用于RNN/LSTM，但也可以用于全连接层）"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def init_classifier_head(classifier: nn.Module, num_classes: int = 2):
    """
    专门初始化分类器头部
    使用较小的初始化，避免初始预测过于自信
    """
    for m in classifier.modules():
        if isinstance(m, nn.Linear):
            # 最后一层使用较小的初始化
            if m.out_features == num_classes:
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)
            else:
                # 其他层使用标准初始化
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def apply_initialization(model: nn.Module, init_type: str = 'xavier', num_classes: int = 2):
    """
    对整个模型应用初始化策略
    
    Args:
        model: 要初始化的模型
        init_type: 初始化类型 ('xavier', 'kaiming', 'orthogonal')
        num_classes: 分类数量（用于特殊处理分类器）
    """
    if init_type == 'xavier':
        init_fn = init_weights_xavier_uniform
    elif init_type == 'kaiming':
        init_fn = init_weights_kaiming_normal
    elif init_type == 'orthogonal':
        init_fn = init_weights_orthogonal
    else:
        raise ValueError(f"Unknown init_type: {init_type}")
    
    # 先应用通用初始化
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            init_fn(module)
    
    # 特殊处理分类器头部
    if hasattr(model, 'classifier'):
        init_classifier_head(model.classifier, num_classes)
    
    print(f"✅ 模型初始化完成: {init_type}")

