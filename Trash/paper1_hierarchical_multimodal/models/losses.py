"""
优化的损失函数
包括Focal Loss和Label Smoothing CrossEntropy Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    支持类别权重（alpha可以是标量或tensor）
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', class_weights=None):
        super().__init__()
        # alpha可以是标量或每个类别的权重tensor
        if isinstance(alpha, (list, tuple)):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            pass
        else:
            alpha = torch.tensor([alpha] * 2, dtype=torch.float32)  # 默认2类
        
        self.register_buffer('alpha', alpha)
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
        if class_weights is not None:
            if isinstance(class_weights, (list, tuple)):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', class_weights)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.class_weights)
        
        # 计算p_t (预测概率)
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # 获取每个样本对应类别的概率
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        
        # 获取每个样本对应类别的alpha权重
        alpha_t = self.alpha[targets]
        
        # 数值稳定性：限制pt的范围
        pt = torch.clamp(pt, min=1e-8, max=1.0 - 1e-8)
        
        # Focal Loss: -alpha_t * (1 - pt)^gamma * log(pt)
        focal_loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt)
        
        # 检查NaN
        if torch.isnan(focal_loss).any():
            # 如果出现NaN，回退到加权交叉熵
            return ce_loss.mean() if self.reduction == 'mean' else ce_loss.sum() if self.reduction == 'sum' else ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing CrossEntropy Loss
    """
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        num_classes = inputs.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = -torch.sum(true_dist * log_probs, dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    结合Focal Loss和Label Smoothing的损失函数
    支持类别权重以处理类别不平衡
    """
    def __init__(self, focal_alpha=1.0, focal_gamma=2.0, label_smoothing=0.1, 
                 focal_weight=0.5, smoothing_weight=0.5, class_weights=None):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, class_weights=class_weights)
        self.smoothing_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.focal_weight = focal_weight
        self.smoothing_weight = smoothing_weight

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        smoothing = self.smoothing_loss(inputs, targets)
        return self.focal_weight * focal + self.smoothing_weight * smoothing


class ClassWeightedCrossEntropy(nn.Module):
    """
    类别加权的交叉熵损失
    直接使用类别权重，简单有效
    """
    def __init__(self, class_weights=None, label_smoothing=0.0):
        super().__init__()
        if class_weights is not None:
            if isinstance(class_weights, (list, tuple)):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        return F.cross_entropy(
            inputs, 
            targets, 
            weight=self.class_weights,
            label_smoothing=self.label_smoothing
        )

