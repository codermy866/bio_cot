#!/usr/bin/env python3
"""
测试修复后的代码是否能正常运行
"""

import sys
import torch
import timm

def test_imports():
    """测试所有必要的导入"""
    print("=== 测试导入 ===")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch导入失败: {e}")
        return False
    
    try:
        import timm
        print(f"✓ timm: {timm.__version__}")
    except ImportError as e:
        print(f"✗ timm导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ numpy: {np.__version__}")
    except ImportError as e:
        print(f"✗ numpy导入失败: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib")
    except ImportError as e:
        print(f"✗ matplotlib导入失败: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ seaborn")
    except ImportError as e:
        print(f"✗ seaborn导入失败: {e}")
        return False
    
    return True

def test_cuda():
    """测试CUDA可用性"""
    print("\n=== 测试CUDA ===")
    
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数: {torch.cuda.device_count()}")
        print(f"当前设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name()}")
        return True
    else:
        print("警告: CUDA不可用，将使用CPU")
        return False

def test_vit_creation():
    """测试ViT模型创建"""
    print("\n=== 测试ViT模型创建 ===")
    
    try:
        # 测试修复后的模型创建方式
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        print("✓ ViT模型创建成功")
        
        # 测试前向传播
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ 前向传播成功，输出形状: {output.shape}")
        
        # 测试移除分类头
        model.head = torch.nn.Identity()
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ 移除分类头后前向传播成功，输出形状: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ ViT模型创建失败: {e}")
        return False

def test_dataset_loading():
    """测试数据集加载"""
    print("\n=== 测试数据集加载 ===")
    
    try:
        from util.datasets import build_dataset
        print("✓ 数据集模块导入成功")
        
        # 这里只是测试导入，不实际加载数据集
        return True
    except Exception as e:
        print(f"✗ 数据集模块导入失败: {e}")
        return False

def test_causal_model():
    """测试因果模型"""
    print("\n=== 测试因果模型 ===")
    
    try:
        import models_causal_gnn
        print("✓ 因果模型模块导入成功")
        
        # 测试模型创建
        oct_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        col_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        
        model = models_causal_gnn.CausalMultimodalTransformer(
            oct_model=oct_model,
            col_model=col_model,
            num_classes=2,
            embed_dim=768,  # ViT base的默认维度
            causal_dim=64,
            dropout_rate=0.2
        )
        print("✓ 因果多模态模型创建成功")
        
        return True
    except Exception as e:
        print(f"✗ 因果模型测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试修复后的代码...")
    
    tests = [
        test_imports,
        test_cuda,
        test_vit_creation,
        test_dataset_loading,
        test_causal_model
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！代码修复成功！")
        return True
    else:
        print("❌ 部分测试失败，需要进一步修复")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 