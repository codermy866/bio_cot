#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试训练脚本是否能正常运行"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=" * 80)
print("🔍 测试训练脚本导入...")
print("=" * 80)

# 测试导入
try:
    print("1. 测试导入 AdaptiveCausalInterventionCLIP...")
    from src.models.adaptive_causal_intervention_clip import AdaptiveCausalInterventionCLIP
    print("   ✅ 成功")
except ImportError as e:
    print(f"   ❌ 失败: {e}")
    try:
        sys.path.insert(0, str(project_root / 'src'))
        from models.adaptive_causal_intervention_clip import AdaptiveCausalInterventionCLIP
        print("   ✅ 使用备用路径成功")
    except ImportError as e2:
        print(f"   ❌ 备用路径也失败: {e2}")
        sys.exit(1)

try:
    print("2. 测试导入 EnhancedMultimodalCervicalDataset...")
    from utils.enhanced_multimodal_dataset import EnhancedMultimodalCervicalDataset
    print("   ✅ 成功")
except ImportError as e:
    print(f"   ❌ 失败: {e}")
    sys.exit(1)

try:
    print("3. 测试导入 SwinTImageEncoder...")
    from models.SwinT.swin_image_encoder import SwinTImageEncoder
    print("   ✅ 成功")
except ImportError as e:
    print(f"   ❌ 失败: {e}")
    sys.exit(1)

try:
    print("4. 测试导入训练脚本...")
    import training.train_clip_with_innovations as train_script
    print("   ✅ 成功")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ 所有导入测试通过！")
print("=" * 80)


