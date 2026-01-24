"""
层次化多粒度多模态融合模型
用于Paper1的一区期刊创新方案
"""

from .multi_granularity_encoder import (
    LocalFeatureEncoder,
    GlobalFeatureEncoder,
    TemporalFeatureEncoder,
    SpatialFeatureEncoder
)
from .cross_modal_aligner import CrossModalAligner
from .multi_granularity_fusion import (
    FineGrainFusion,
    MidGrainFusion,
    CoarseGrainFusion
)
from .adaptive_weighting import (
    QualityAssessor,
    AdaptiveModalityWeighting
)
from .contrastive_alignment import ContrastiveAlignmentModule
from .hierarchical_classifier import HierarchicalClassifier
from .hierarchical_multimodal_model import HierarchicalMultimodalModel
from .losses import (
    FocalLoss, 
    LabelSmoothingCrossEntropy, 
    CombinedLoss,
    ClassWeightedCrossEntropy
)

__all__ = [
    'LocalFeatureEncoder',
    'GlobalFeatureEncoder',
    'TemporalFeatureEncoder',
    'SpatialFeatureEncoder',
    'CrossModalAligner',
    'FineGrainFusion',
    'MidGrainFusion',
    'CoarseGrainFusion',
    'QualityAssessor',
    'AdaptiveModalityWeighting',
    'ContrastiveAlignmentModule',
    'HierarchicalClassifier',
    'HierarchicalMultimodalModel',
    'FocalLoss',
    'LabelSmoothingCrossEntropy',
    'CombinedLoss',
    'ClassWeightedCrossEntropy',
]

