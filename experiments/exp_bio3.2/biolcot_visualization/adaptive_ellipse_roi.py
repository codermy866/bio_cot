#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自适应椭圆ROI检测模块
根据热力图激活分布和图像特征自动确定椭圆中心、大小和方向
"""

import numpy as np
import cv2


def detect_adaptive_ellipse_roi(heatmap, image_rgb=None, min_activation_threshold=0.1, 
                                 min_roi_ratio=0.15, max_roi_ratio=0.45):
    """
    自适应检测椭圆ROI区域
    
    Args:
        heatmap: [H, W] 热力图 (0-1)
        image_rgb: [H, W, 3] RGB图像 (可选，用于颜色先验)
        min_activation_threshold: 最小激活阈值，用于确定激活区域
        min_roi_ratio: 最小ROI比例（相对于图像尺寸）
        max_roi_ratio: 最大ROI比例（相对于图像尺寸）
    
    Returns:
        center_roi_mask: [H, W] 椭圆ROI遮罩 (0-1)
        ellipse_params: dict 包含椭圆参数 {'center': (y, x), 'axes': (a, b), 'angle': angle}
    """
    h, w = heatmap.shape
    
    # --- 方法1: 基于热力图激活分布检测椭圆中心 ---
    # 创建激活区域的二值掩码
    activation_mask = (heatmap > min_activation_threshold).astype(np.float32)
    
    # 计算激活区域的质心（加权质心，权重为激活值）
    if np.sum(activation_mask) > 0:
        # 使用激活值作为权重
        weights = heatmap * activation_mask
        total_weight = np.sum(weights)
        
        if total_weight > 0:
            # 计算加权质心
            Y, X = np.ogrid[:h, :w]
            center_y = np.sum(Y * weights) / total_weight
            center_x = np.sum(X * weights) / total_weight
        else:
            # 如果权重为0，使用几何质心
            moments = cv2.moments((activation_mask * 255).astype(np.uint8))
            if moments['m00'] > 0:
                center_x = moments['m10'] / moments['m00']
                center_y = moments['m01'] / moments['m00']
            else:
                # 如果还是没有，使用图像中心
                center_y, center_x = h // 2, w // 2
    else:
        # 如果没有激活区域，使用图像中心
        center_y, center_x = h // 2, w // 2
    
    # --- 方法2: 结合颜色先验（如果提供了图像）---
    if image_rgb is not None:
        # 检测红色/糜烂区域（宫颈口特征）
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hue = img_hsv[:, :, 0].astype(np.float32)
        sat = img_hsv[:, :, 1].astype(np.float32) / 255.0
        val = img_hsv[:, :, 2].astype(np.float32) / 255.0
        
        # 红色区域掩码
        is_red = ((hue < 15) | (hue > 160)) & (sat > 0.3) & (val > 0.2) & (val < 0.9)
        red_mask = is_red.astype(np.float32)
        
        # 如果红色区域有足够的像素，使用红色区域的质心
        if np.sum(red_mask) > h * w * 0.05:  # 至少5%的像素是红色
            red_weights = red_mask
            total_red_weight = np.sum(red_weights)
            if total_red_weight > 0:
                Y, X = np.ogrid[:h, :w]
                red_center_y = np.sum(Y * red_weights) / total_red_weight
                red_center_x = np.sum(X * red_weights) / total_red_weight
                
                # 结合热力图质心和红色区域质心（加权平均）
                # 如果热力图有激活，更信任热力图；否则更信任红色区域
                if np.sum(activation_mask) > h * w * 0.01:  # 热力图有激活
                    center_y = 0.7 * center_y + 0.3 * red_center_y
                    center_x = 0.7 * center_x + 0.3 * red_center_x
                else:
                    center_y = 0.3 * center_y + 0.7 * red_center_y
                    center_x = 0.3 * center_x + 0.7 * red_center_x
    
    # 确保中心在图像范围内
    center_y = np.clip(center_y, h * 0.1, h * 0.9)
    center_x = np.clip(center_x, w * 0.1, w * 0.9)
    
    # --- 方法3: 基于激活区域的空间分布确定椭圆大小和方向 ---
    # 计算激活区域的协方差矩阵，用于确定椭圆的主轴和方向
    Y, X = np.ogrid[:h, :w]
    
    # 使用激活值作为权重
    weights = heatmap * activation_mask
    total_weight = np.sum(weights)
    
    if total_weight > 0:
        # 计算加权均值
        mean_y = np.sum(Y * weights) / total_weight
        mean_x = np.sum(X * weights) / total_weight
        
        # 计算加权协方差矩阵
        Y_centered = Y - mean_y
        X_centered = X - mean_x
        
        cov_yy = np.sum(Y_centered * Y_centered * weights) / total_weight
        cov_xx = np.sum(X_centered * X_centered * weights) / total_weight
        cov_xy = np.sum(Y_centered * X_centered * weights) / total_weight
        
        # 计算特征值和特征向量（椭圆的主轴）
        # 协方差矩阵: [[cov_xx, cov_xy], [cov_xy, cov_yy]]
        cov_matrix = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # 特征值从大到小排序
        idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # 椭圆的长轴和短轴（使用特征值的平方根，乘以缩放因子）
        # 使用2倍标准差作为椭圆半径（覆盖约95%的数据）
        scale_factor = 2.0
        axis_a = np.sqrt(eigenvals[0]) * scale_factor  # 长轴
        axis_b = np.sqrt(eigenvals[1]) * scale_factor  # 短轴
        
        # 椭圆的角度（主特征向量的角度）
        angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]) * 180 / np.pi
        
        # 限制椭圆大小在合理范围内
        min_axis = min(h, w) * min_roi_ratio
        max_axis = min(h, w) * max_roi_ratio
        
        axis_a = np.clip(axis_a, min_axis, max_axis)
        axis_b = np.clip(axis_b, min_axis, max_axis)
        
        # 确保长轴 >= 短轴
        if axis_a < axis_b:
            axis_a, axis_b = axis_b, axis_a
            angle += 90
        
    else:
        # 如果没有激活区域，使用默认椭圆（中心30% x 35%）
        axis_a = w * 0.35  # 长轴（水平方向）
        axis_b = h * 0.30  # 短轴（垂直方向）
        angle = 0  # 水平方向
    
    # --- 创建椭圆ROI遮罩 ---
    Y, X = np.ogrid[:h, :w]
    
    # 将坐标转换到以椭圆中心为原点的坐标系
    X_centered = X - center_x
    Y_centered = Y - center_y
    
    # 旋转坐标系（逆时针旋转-angle度）
    angle_rad = -angle * np.pi / 180
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    X_rot = X_centered * cos_a - Y_centered * sin_a
    Y_rot = X_centered * sin_a + Y_centered * cos_a
    
    # 计算椭圆距离
    ellipse_dist = np.sqrt((X_rot / axis_a)**2 + (Y_rot / axis_b)**2)
    
    # 创建平滑的椭圆遮罩（sigmoid过渡）
    center_roi_mask = 1.0 / (1.0 + np.exp(10.0 * (ellipse_dist - 1.0)))
    
    # 保存椭圆参数
    ellipse_params = {
        'center': (int(center_y), int(center_x)),
        'axes': (int(axis_a), int(axis_b)),
        'angle': angle
    }
    
    return center_roi_mask, ellipse_params

