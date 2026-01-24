#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超强数据增强策略
针对医学图像优化的增强策略，提升模型泛化能力
"""

import torch
import torchvision.transforms as transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
try:
    import cv2
except ImportError:
    cv2 = None


class UltraStrongAugmentation:
    """
    超强数据增强策略
    结合timm的AutoAugment和Albumentations的医学图像增强
    """
    
    def __init__(self, input_size: int = 224, is_training: bool = True, use_albumentations: bool = True):
        self.input_size = input_size
        self.is_training = is_training
        self.use_albumentations = use_albumentations
        
        if is_training:
            if use_albumentations:
                # 使用Albumentations进行超强增强
                self.transform = A.Compose([
                    # 基础变换
                    A.Resize(input_size, input_size),
                    
                    # 几何变换（增强）
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.3),
                    A.RandomRotate90(p=0.3),
                    A.ShiftScaleRotate(
                        shift_limit=0.15,  # 增加到0.15
                        scale_limit=0.15,  # 增加到0.15
                        rotate_limit=20,   # 增加到20度
                        border_mode=0,  # cv2.BORDER_CONSTANT
                        value=0,
                        p=0.5
                    ),
                    A.ElasticTransform(
                        alpha=120,
                        sigma=120 * 0.05,
                        alpha_affine=120 * 0.03,
                        border_mode=0,  # cv2.BORDER_CONSTANT
                        value=0,
                        p=0.3
                    ),
                    A.GridDistortion(
                        num_steps=5,
                        distort_limit=0.3,
                        border_mode=0,  # cv2.BORDER_CONSTANT
                        value=0,
                        p=0.3
                    ),
                    A.OpticalDistortion(
                        distort_limit=0.3,
                        shift_limit=0.1,
                        border_mode=0,  # cv2.BORDER_CONSTANT
                        value=0,
                        p=0.3
                    ),
                    
                    # 颜色增强（增强）
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3,  # 增加到0.3
                        contrast_limit=0.3,    # 增加到0.3
                        brightness_by_max=True,
                        p=0.6
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=15,     # 增加到15
                        sat_shift_limit=30,     # 增加到30
                        val_shift_limit=30,     # 增加到30
                        p=0.5
                    ),
                    A.CLAHE(
                        clip_limit=4.0,
                        tile_grid_size=(8, 8),
                        p=0.4
                    ),
                    A.RandomGamma(
                        gamma_limit=(80, 120),
                        p=0.4
                    ),
                    A.ColorJitter(
                        brightness=0.3,
                        contrast=0.3,
                        saturation=0.3,
                        hue=0.1,
                        p=0.4
                    ),
                    
                    # 噪声和模糊（增强）
                    A.GaussNoise(
                        var_limit=(10.0, 100.0),  # 增加到100
                        mean=0,
                        p=0.4
                    ),
                    A.GaussianBlur(
                        blur_limit=(3, 9),  # 增加到9
                        p=0.4
                    ),
                    A.MotionBlur(
                        blur_limit=7,  # 增加到7
                        p=0.3
                    ),
                    A.MedianBlur(
                        blur_limit=5,
                        p=0.2
                    ),
                    
                    # 遮挡和dropout（增强）
                    A.CoarseDropout(
                        max_holes=12,      # 增加到12
                        max_height=48,     # 增加到48
                        max_width=48,      # 增加到48
                        min_holes=2,
                        min_height=16,
                        min_width=16,
                        fill_value=0,
                        mask_fill_value=None,
                        p=0.4
                    ),
                    A.GridDropout(
                        ratio=0.3,
                        holes_number_x=4,
                        holes_number_y=4,
                        shift_x=0,
                        shift_y=0,
                        random_offset=True,
                        fill_value=0,
                        p=0.3
                    ),
                    A.RandomShadow(
                        shadow_roi=(0, 0.5, 1, 1),
                        num_shadows_lower=1,
                        num_shadows_upper=2,
                        p=0.3
                    ),
                    
                    # 混合增强
                    A.MixUp(
                        alpha=0.2,
                        p=0.2
                    ),
                    A.Cutout(
                        num_holes=8,
                        max_h_size=32,
                        max_w_size=32,
                        fill_value=0,
                        p=0.3
                    ),
                    
                    # 归一化
                    A.Normalize(
                        mean=IMAGENET_DEFAULT_MEAN,
                        std=IMAGENET_DEFAULT_STD
                    ),
                    ToTensorV2()
                ])
            else:
                # 使用timm的AutoAugment（更强版本）
                self.transform = create_transform(
                    input_size=input_size,
                    is_training=True,
                    color_jitter=0.5,  # 增加到0.5
                    auto_augment='rand-m9-mstd0.5-inc1',  # 使用最强的AutoAugment
                    interpolation='bicubic',
                    re_prob=0.4,  # 增加到0.4
                    re_mode='pixel',
                    re_count=2,  # 增加到2
                    mean=IMAGENET_DEFAULT_MEAN,
                    std=IMAGENET_DEFAULT_STD,
                )
        else:
            # 验证/测试时的标准化
            if use_albumentations:
                self.transform = A.Compose([
                    A.Resize(input_size, input_size),
                    A.Normalize(
                        mean=IMAGENET_DEFAULT_MEAN,
                        std=IMAGENET_DEFAULT_STD
                    ),
                    ToTensorV2()
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((input_size, input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
                ])
    
    def __call__(self, image):
        if self.use_albumentations and isinstance(image, Image.Image):
            image = np.array(image)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            result = self.transform(image=image)
            return result['image']
        elif isinstance(image, np.ndarray):
            if self.use_albumentations:
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                result = self.transform(image=image)
                return result['image']
            else:
                image = Image.fromarray(image)
                return self.transform(image)
        else:
            return self.transform(image)


def create_ultra_strong_transform(input_size: int = 224, is_training: bool = True, use_albumentations: bool = True):
    """
    创建超强数据增强变换
    
    Args:
        input_size: 输入图像尺寸
        is_training: 是否为训练模式
        use_albumentations: 是否使用Albumentations（推荐，功能更强）
    
    Returns:
        transform: 数据增强变换
    """
    return UltraStrongAugmentation(
        input_size=input_size,
        is_training=is_training,
        use_albumentations=use_albumentations
    )

"""
超强数据增强策略 - 针对医学图像优化
目标：提升模型泛化能力，达到AUC 0.85+
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class UltraStrongAugmentation:
    """
    超强数据增强策略
    结合timm的AutoAugment和Albumentations的高级增强
    """
    
    def __init__(
        self,
        input_size: int = 224,
        is_training: bool = True,
        use_albumentations: bool = True,
        use_timm_aug: bool = True,
    ):
        self.input_size = input_size
        self.is_training = is_training
        self.use_albumentations = use_albumentations
        self.use_timm_aug = use_timm_aug
        
        if is_training:
            if use_albumentations and use_timm_aug:
                # 混合策略：先Albumentations，后timm
                self.alb_transform = self._create_albumentations_transform()
                self.timm_transform = self._create_timm_transform()
            elif use_albumentations:
                self.alb_transform = self._create_albumentations_transform()
                self.timm_transform = None
            elif use_timm_aug:
                self.alb_transform = None
                self.timm_transform = self._create_timm_transform()
            else:
                self.alb_transform = None
                self.timm_transform = None
        else:
            # 验证/测试时只做标准化
            self.alb_transform = None
            self.timm_transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            ])
    
    def _create_albumentations_transform(self):
        """创建Albumentations增强（超强版本）"""
        return A.Compose([
            # 基础变换
            A.Resize(self.input_size, self.input_size, interpolation=cv2.INTER_CUBIC),
            
            # 几何变换（增强版）
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),  # 医学图像可能需要垂直翻转
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=30, p=0.4, border_mode=cv2.BORDER_REFLECT_101),  # 增加旋转角度
            A.ShiftScaleRotate(
                shift_limit=0.15,  # 增加平移范围
                scale_limit=0.2,   # 增加缩放范围
                rotate_limit=30,
                p=0.5,
                border_mode=cv2.BORDER_REFLECT_101
            ),
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                alpha_affine=120 * 0.03,
                p=0.3
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=0.3
            ),
            A.OpticalDistortion(
                distort_limit=0.2,
                shift_limit=0.1,
                p=0.3
            ),
            
            # 颜色增强（增强版）
            A.RandomBrightnessContrast(
                brightness_limit=0.3,  # 增加亮度变化
                contrast_limit=0.3,     # 增加对比度变化
                brightness_by_max=True,
                p=0.6
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,     # 增加色调变化
                sat_shift_limit=30,     # 增加饱和度变化
                val_shift_limit=30,     # 增加明度变化
                p=0.5
            ),
            A.CLAHE(
                clip_limit=4.0,
                tile_grid_size=(8, 8),
                p=0.3
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.3
            ),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=0.4
            ),
            
            # 噪声和模糊（增强版）
            A.GaussNoise(
                var_limit=(10.0, 100.0),  # 增加噪声范围
                p=0.4
            ),
            A.GaussianBlur(
                blur_limit=(3, 9),  # 增加模糊范围
                p=0.3
            ),
            A.MotionBlur(
                blur_limit=7,  # 增加运动模糊
                p=0.3
            ),
            A.MedianBlur(
                blur_limit=7,
                p=0.2
            ),
            
            # 高级增强
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                p=0.3
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0.5,
                p=0.2
            ),
            
            # Cutout/Dropout（增强版）
            A.CoarseDropout(
                max_holes=12,      # 增加孔洞数量
                max_height=48,     # 增加孔洞大小
                max_width=48,
                min_holes=2,
                min_height=16,
                min_width=16,
                fill_value=0,
                p=0.4
            ),
            A.GridDropout(
                ratio=0.3,
                holes_per_x=4,
                holes_per_y=4,
                p=0.3
            ),
            
            # 混合增强
            A.MixUp(
                alpha=0.2,
                p=0.2
            ),
            A.CutMix(
                alpha=1.0,
                p=0.2
            ),
            
            # 标准化
            A.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD
            ),
            ToTensorV2()
        ])
    
    def _create_timm_transform(self):
        """创建timm的AutoAugment增强（超强版本）"""
        return create_transform(
            input_size=self.input_size,
            is_training=True,
            color_jitter=0.5,  # 增加颜色抖动
            auto_augment='rand-m9-mstd0.5-inc1',  # 使用最强的AutoAugment策略
            interpolation='bicubic',
            re_prob=0.4,       # 增加RandomErasing概率
            re_mode='pixel',
            re_count=2,        # 增加RandomErasing次数
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
    
    def __call__(self, image):
        """应用增强"""
        if not self.is_training:
            if self.timm_transform:
                return self.timm_transform(image)
            else:
                # 简单的标准化
                if isinstance(image, Image.Image):
                    image = transforms.Resize((self.input_size, self.input_size))(image)
                    image = transforms.ToTensor()(image)
                    image = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(image)
                return image
        
        # 训练时的增强
        if self.use_albumentations and self.alb_transform:
            # 转换为numpy数组
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            elif isinstance(image, torch.Tensor):
                image_np = image.numpy().transpose(1, 2, 0)
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image
            
            # 应用Albumentations
            augmented = self.alb_transform(image=image_np)['image']
            
            # 如果还需要timm增强，转换为PIL再应用
            if self.use_timm_aug and self.timm_transform:
                # 反标准化
                mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
                std = torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1)
                augmented = augmented * std + mean
                augmented = torch.clamp(augmented, 0, 1)
                
                # 转换为PIL
                augmented_pil = transforms.ToPILImage()(augmented)
                
                # 应用timm增强
                augmented = self.timm_transform(augmented_pil)
            
            return augmented
        
        elif self.use_timm_aug and self.timm_transform:
            return self.timm_transform(image)
        
        else:
            # 默认增强
            if isinstance(image, Image.Image):
                image = transforms.Resize((self.input_size, self.input_size))(image)
                image = transforms.ToTensor()(image)
                image = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(image)
            return image


# 兼容性：导入cv2
try:
    import cv2
except ImportError:
    print("Warning: cv2 not found, some augmentations may not work")
    cv2 = None
