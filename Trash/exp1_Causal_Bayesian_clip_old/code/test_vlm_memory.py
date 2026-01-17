#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试VLM模型的显存使用情况
在A6000 48GB GPU上测试
"""

import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import sys
import os

def format_bytes(bytes):
    """格式化字节数"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"

def get_memory_info():
    """获取显存信息"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_allocated = torch.cuda.max_memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        return {
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_allocated,
            'total': total
        }
    return None

def test_qwen_vl_2b():
    """测试Qwen2-VL-2B模型"""
    print("=" * 80)
    print("测试 Qwen2-VL-2B-Instruct")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 清空显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    mem_before = get_memory_info()
    if mem_before:
        print(f"\n初始显存:")
        print(f"  已分配: {format_bytes(mem_before['allocated'])}")
        print(f"  已保留: {format_bytes(mem_before['reserved'])}")
        print(f"  总显存: {format_bytes(mem_before['total'])}")
    
    try:
        print("\n📥 加载模型...")
        print("   模型: Qwen/Qwen2-VL-2B-Instruct")
        print("   精度: FP16")
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        
        mem_after_load = get_memory_info()
        if mem_after_load:
            print(f"\n✅ 模型加载完成")
            print(f"  模型显存: {format_bytes(mem_after_load['allocated'] - mem_before['allocated'])}")
            print(f"  总已分配: {format_bytes(mem_after_load['allocated'])}")
            print(f"  剩余显存: {format_bytes(mem_after_load['total'] - mem_after_load['reserved'])}")
        
        # 测试不同batch size
        batch_sizes = [1, 4, 8, 12, 16]
        
        print("\n" + "=" * 80)
        print("测试不同Batch Size的显存使用")
        print("=" * 80)
        
        for bs in batch_sizes:
            print(f"\n🔍 测试 Batch Size = {bs}")
            
            # 清空之前的峰值
            torch.cuda.reset_peak_memory_stats()
            
            try:
                # 创建测试图像
                test_images = torch.randn(bs, 3, 224, 224).half().to(device)
                
                # 处理输入
                inputs = processor(images=test_images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 前向传播
                with torch.no_grad():
                    model.eval()
                    outputs = model.get_image_features(**inputs)
                
                mem_forward = get_memory_info()
                if mem_forward:
                    peak_mem = torch.cuda.max_memory_allocated()
                    forward_mem = peak_mem - mem_after_load['allocated']
                    print(f"  前向传播显存: {format_bytes(forward_mem)}")
                    print(f"  峰值显存: {format_bytes(peak_mem)}")
                    print(f"  剩余显存: {format_bytes(mem_forward['total'] - mem_forward['reserved'])}")
                    
                    if mem_forward['reserved'] / mem_forward['total'] > 0.9:
                        print(f"  ⚠️  显存使用率 > 90%，不建议使用更大batch size")
                        break
                
                # 测试训练模式（前向+反向）
                if bs <= 8:  # 只测试较小的batch size
                    print(f"  测试训练模式（前向+反向）...")
                    model.train()
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
                    
                    torch.cuda.reset_peak_memory_stats()
                    
                    # 前向
                    outputs = model.get_image_features(**inputs)
                    dummy_loss = outputs.mean()
                    
                    # 反向
                    dummy_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    mem_train = get_memory_info()
                    if mem_train:
                        peak_train = torch.cuda.max_memory_allocated()
                        train_mem = peak_train - mem_after_load['allocated']
                        print(f"    训练模式显存: {format_bytes(train_mem)}")
                        print(f"    峰值显存: {format_bytes(peak_train)}")
                
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ❌ OOM错误！Batch Size {bs} 太大")
                    break
                else:
                    raise e
        
        print("\n" + "=" * 80)
        print("✅ 测试完成")
        print("=" * 80)
        
        # 最终显存统计
        mem_final = get_memory_info()
        if mem_final:
            print(f"\n最终显存使用:")
            print(f"  模型显存: {format_bytes(mem_final['allocated'] - mem_before['allocated'])}")
            print(f"  总已分配: {format_bytes(mem_final['allocated'])}")
            print(f"  剩余显存: {format_bytes(mem_final['total'] - mem_final['reserved'])}")
            print(f"  显存使用率: {mem_final['reserved'] / mem_final['total'] * 100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qwen_vl_2b_with_quantization():
    """测试4bit量化的Qwen2-VL-2B"""
    print("\n" + "=" * 80)
    print("测试 Qwen2-VL-2B-Instruct (4bit量化)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        mem_before = get_memory_info()
        
        print("\n📥 加载4bit量化模型...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        
        mem_after = get_memory_info()
        if mem_after and mem_before:
            model_mem = mem_after['allocated'] - mem_before['allocated']
            print(f"\n✅ 4bit量化模型加载完成")
            print(f"  模型显存: {format_bytes(model_mem)}")
            print(f"  相比FP16节省: ~75%")
        
        return True
        
    except ImportError:
        print("\n⚠️  BitsAndBytesConfig未安装，跳过4bit量化测试")
        print("   安装命令: pip install bitsandbytes")
        return False
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return False

def test_integrated_memory():
    """测试完整框架的显存使用"""
    print("\n" + "=" * 80)
    print("测试完整VLM增强框架的显存使用")
    print("=" * 80)
    
    try:
        # 导入完整模型
        sys.path.insert(0, os.path.dirname(__file__))
        from vlm_enhanced_causal_clip import VLMEnhancedCausalBayesianCLIP
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        mem_before = get_memory_info()
        
        print("\n📥 加载完整模型...")
        model = VLMEnhancedCausalBayesianCLIP(
            embed_dim=768,
            clinical_dim=7,
            num_classes=2,
            vlm_model_name="Qwen/Qwen2-VL-2B-Instruct",
            use_medical_kb=False,  # 暂时关闭知识库
            use_vlm_guidance=True,
            use_report_generation=True
        ).to(device)
        
        mem_after = get_memory_info()
        if mem_after and mem_before:
            model_mem = mem_after['allocated'] - mem_before['allocated']
            print(f"\n✅ 完整模型加载完成")
            print(f"  模型显存: {format_bytes(model_mem)}")
        
        # 测试前向传播
        print("\n🔍 测试前向传播 (Batch=4)...")
        batch_size = 4
        oct_images = torch.randn(batch_size, 120, 3, 224, 224).half().to(device)
        colpo_images = torch.randn(batch_size, 3, 3, 224, 224).half().to(device)
        clinical = torch.randn(batch_size, 7).half().to(device)
        
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            model.eval()
            outputs = model(
                oct_images=oct_images,
                colposcopy_images=colpo_images,
                clinical_features=clinical
            )
        
        mem_forward = get_memory_info()
        if mem_forward:
            peak = torch.cuda.max_memory_allocated()
            forward_mem = peak - mem_after['allocated']
            print(f"  前向传播显存: {format_bytes(forward_mem)}")
            print(f"  峰值显存: {format_bytes(peak)}")
            print(f"  剩余显存: {format_bytes(mem_forward['total'] - mem_forward['reserved'])}")
        
        return True
        
    except ImportError as e:
        print(f"\n⚠️  模型未找到: {e}")
        print("   请先确保vlm_enhanced_causal_clip.py存在")
        return False
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("VLM显存使用测试 - A6000 48GB")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("\n❌ CUDA不可用，无法测试")
        return
    
    # 显示GPU信息
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"\n🖥️  GPU信息:")
    print(f"  名称: {gpu_name}")
    print(f"  总显存: {format_bytes(gpu_memory)}")
    print(f"  CUDA版本: {torch.version.cuda}")
    
    # 测试1: Qwen2-VL-2B
    print("\n\n")
    test_qwen_vl_2b()
    
    # 测试2: 4bit量化（可选）
    print("\n\n")
    test_qwen_vl_2b_with_quantization()
    
    # 测试3: 完整框架（可选）
    # print("\n\n")
    # test_integrated_memory()
    
    print("\n" + "=" * 80)
    print("✅ 所有测试完成")
    print("=" * 80)
    
    # 最终显存信息
    mem = get_memory_info()
    if mem:
        print(f"\n最终显存状态:")
        print(f"  已分配: {format_bytes(mem['allocated'])}")
        print(f"  已保留: {format_bytes(mem['reserved'])}")
        print(f"  剩余可用: {format_bytes(mem['total'] - mem['reserved'])}")
        print(f"  使用率: {mem['reserved'] / mem['total'] * 100:.1f}%")

if __name__ == "__main__":
    main()

