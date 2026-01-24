#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试脚本 - 验证模型和CUDA=1环境
"""

import torch
import numpy as np
import os
import sys

def test_cuda_environment():
    """测试CUDA环境"""
    print("🔍 测试CUDA环境...")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    print(f"✅ CUDA可用，版本: {torch.version.cuda}")
    print(f"✅ GPU数量: {torch.cuda.device_count()}")
    
    # 检查指定GPU
    if torch.cuda.device_count() > 1:
        print(f"✅ GPU 0: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU 1: {torch.cuda.get_device_name(1)}")
    
    return True

def test_model_loading():
    """测试模型加载"""
    print("\n🔍 测试模型加载...")
    
    try:
        # 设置CUDA设备 (CUDA_VISIBLE_DEVICES=1后，GPU 1变成GPU 0)
        device = torch.device('cuda:0')
        print(f"✅ 使用设备: {device}")
        
        # 加载模型检查点
        model_path = 'cnn_training_latest/best_model.pth'
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return False
        
        checkpoint = torch.load(model_path, map_location=device)
        print(f"✅ 模型检查点加载成功")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best F1: {checkpoint.get('best_f1', 'N/A')}")
        
        # 创建模型
        from cnn_multimodal_model import CNNMultimodalTransformer
        model = CNNMultimodalTransformer(num_classes=2, embed_dim=512, clinical_dim=8)
        model = model.to(device)
        
        # 加载权重
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        print(f"✅ 模型加载成功")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return True, model, device
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False, None, None

def test_model_inference():
    """测试模型推理"""
    print("\n🔍 测试模型推理...")
    
    success, model, device = test_model_loading()
    if not success:
        return False
    
    try:
        # 创建测试数据
        batch_size = 2
        oct_images = torch.randn(batch_size, 48, 3, 224, 224).to(device)
        col_images = torch.randn(batch_size, 3, 3, 224, 224).to(device)
        clinical_features = torch.randn(batch_size, 8).to(device)
        
        print(f"✅ 测试数据创建成功")
        print(f"  OCT shape: {oct_images.shape}")
        print(f"  COL shape: {col_images.shape}")
        print(f"  Clinical shape: {clinical_features.shape}")
        
        # 模型推理
        with torch.no_grad():
            logits = model(oct_images, col_images, clinical_features)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        
        print(f"✅ 模型推理成功")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Predictions: {preds.cpu().numpy()}")
        print(f"  Probabilities: {probs.cpu().numpy()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型推理失败: {e}")
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n🔍 测试数据加载...")
    
    try:
        # 检查数据目录
        data_path = '5centers_multi'
        if not os.path.exists(data_path):
            print(f"❌ 数据目录不存在: {data_path}")
            return False
        
        print(f"✅ 数据目录存在: {data_path}")
        
        # 检查标签文件
        train_labels = os.path.join(data_path, 'train_labels.csv')
        test_labels = os.path.join(data_path, 'test_labels.csv')
        
        if os.path.exists(train_labels):
            print(f"✅ 训练标签文件存在")
        else:
            print(f"❌ 训练标签文件不存在")
            return False
        
        if os.path.exists(test_labels):
            print(f"✅ 测试标签文件存在")
        else:
            print(f"❌ 测试标签文件不存在")
            return False
        
        # 检查图像目录
        train_oct = os.path.join(data_path, 'train', 'oct')
        train_col = os.path.join(data_path, 'train', 'col')
        
        if os.path.exists(train_oct):
            oct_folders = os.listdir(train_oct)
            print(f"✅ OCT目录存在，包含{len(oct_folders)}个文件夹")
        else:
            print(f"❌ OCT目录不存在")
            return False
        
        if os.path.exists(train_col):
            col_folders = os.listdir(train_col)
            print(f"✅ COL目录存在，包含{len(col_folders)}个文件夹")
        else:
            print(f"❌ COL目录不存在")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 CUDA=1环境测试")
    print("=" * 50)
    
    # 设置CUDA设备
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    # 运行测试
    tests = [
        ("CUDA环境", test_cuda_environment),
        ("数据加载", test_data_loading),
        ("模型加载", lambda: test_model_loading()[0]),
        ("模型推理", test_model_inference)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                print(f"✅ {test_name}测试通过")
            else:
                print(f"❌ {test_name}测试失败")
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results[test_name] = False
    
    # 总结
    print(f"\n{'='*50}")
    print("📊 测试结果总结:")
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print(f"\n🎉 所有测试通过！CUDA=1环境正常")
        print(f"📋 下一步:")
        print(f"  1. 运行分析脚本")
        print(f"  2. 检查GPU使用情况")
        print(f"  3. 监控分析进度")
    else:
        print(f"\n⚠️ 部分测试失败，需要修复问题")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
