# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# # Partly revised by YZ @UCL&Moorfields
# # --------------------------------------------------------

# import json


# def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
#     """
#     Parameter groups for layer-wise lr decay
#     Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
#     """
#     param_group_names = {}
#     param_groups = {}

#     if hasattr(model, 'blocks'):
#         num_layers = len(model.blocks) + 1
#     else:
#         # use the number of layers in the ResNet model as a default value
#         num_layers = len(model.layer1) + len(model.layer2) + len(model.layer3) + len(model.layer4) + 1

# # 确定模型中用于逐层衰减的层数 （num_layers）。
# # hasattr（model， 'blocks'）： 检查模型是否具有 Vision Transformer 的典型属性 blocks（例如 timm.models.vision_transformer.VisionTransformer 的 VisionTransformer 中）。
# # 如果为 true，则 num_layers = len（model.blocks） + 1：计算 transformer 块的数量加 1（例如，对于补丁嵌入或 [CLS] 令牌层），假设 model.blocks 是 transformer 层的列表或序列。
# # 如果为 false（例如，对于 ResNet 模型），则使用回退：将层的长度（第 1 层、第 2 层、第 3 层、第 4 层）加 1 相加，假设具有四个层阶段的类似 ResNet 的结构（常见于卷积网络）。
# # 这种灵活性使该函数能够与不同的模型架构（ViT 或 ResNet）一起使用，设置num_layers以扩展学习率。






#     layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

#     for n, p in model.named_parameters():
#         if not p.requires_grad:
#             continue

#         # no decay: all 1D parameters and model specific ones
#         if p.ndim == 1 or n in no_weight_decay_list:
#             g_decay = "no_decay"
#             this_decay = 0.
#         else:
#             g_decay = "decay"
#             this_decay = weight_decay
            
#         layer_id = get_layer_id_for_vit(n, num_layers)
#         group_name = "layer_%d_%s" % (layer_id, g_decay)

#         if group_name not in param_group_names:
#             this_scale = layer_scales[layer_id]

#             param_group_names[group_name] = {
#                 "lr_scale": this_scale,
#                 "weight_decay": this_decay,
#                 "params": [],
#             }
#             param_groups[group_name] = {
#                 "lr_scale": this_scale,
#                 "weight_decay": this_decay,
#                 "params": [],
#             }

#         param_group_names[group_name]["params"].append(n)
#         param_groups[group_name]["params"].append(p)

#     # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

#     return list(param_groups.values())


# def get_layer_id_for_vit(name, num_layers):
#     """
#     Assign a parameter with its layer id
#     Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
#     """
#     if name in ['cls_token', 'pos_embed']:
#         return 0
#     elif name.startswith('patch_embed'):
#         return 0
#     elif name.startswith('blocks'):
#         return int(name.split('.')[1]) + 1
#     else:
#         return num_layers



'''
由于 timm 中的 Vision Transformer 将它们的层存储在 blocks 属性（Block 模块列表）中，因此我们可以计算 oct_vit 和 col_vit 中的块数来估计层数。方法如下：

使用 model.oct_vit.blocks 和 model.col_vit.blocks 的长度来确定转换器层数。
为 GNN 和 head 图层添加一个小数字，以考虑它们的参数。
调整 layer decay logic 以分配适当的学习率。

'''



# util/lr_decay.py
import math
from typing import Iterable, Optional

import torch
from torch import nn, Tensor

def param_groups_lrd(model: nn.Module,
                     weight_decay: float = 0.05,
                     no_weight_decay_list: Optional[list] = None,
                     layer_decay: float = 0.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    no_weight_decay_list = no_weight_decay_list or []
    param_group_names = {}
    param_groups = {}

    # Get number of layers from the Vision Transformer blocks
    # Assuming oct_vit and col_vit have the same number of blocks
    num_layers = len(model.oct_vit.blocks) + len(model.col_vit.blocks) + 2  # +2 for GNN and head
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay for certain parameters
        if any(nd in name for nd in no_weight_decay_list):
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        # Assign layer scale based on the module
        layer_id = 0
        if 'oct_vit.blocks' in name or 'col_vit.blocks' in name:
            # Extract block index from the name (e.g., blocks.5)
            block_idx = int(name.split('.')[2]) if 'blocks' in name and len(name.split('.')) > 2 else 0
            layer_id = block_idx + 1  # Layer 1 to num_blocks
        elif 'gnn' in name:
            layer_id = len(model.oct_vit.blocks) + 1  # Place GNN after ViT blocks
        elif 'head' in name:
            layer_id = num_layers  # Head gets the lowest lr

        group_name = f"layer_{layer_id}_{g_decay}"
        if group_name not in param_group_names:
            param_group_names[group_name] = {
                "params": [],
                "weight_decay": this_decay,
                "param_names": [],
            }
            param_groups[group_name] = {
                "params": [],
                "weight_decay": this_decay,
                "lr_scale": layer_scales[layer_id],
                "param_names": [],
            }

        param_group_names[group_name]["params"].append(param)
        param_group_names[group_name]["param_names"].append(name)
        param_groups[group_name]["params"].append(param)
        param_groups[group_name]["param_names"].append(name)

    return list(param_groups.values())

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(model, name)
            group_name = f"layer_{layer_id}_{group_name}"
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "param_names": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "param_names": [],
                "lr_scale": scale
            }

        parameter_group_names[group_name]["params"].append(param)
        parameter_group_names[group_name]["param_names"].append(name)
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_vars[group_name]["param_names"].append(name)

    return list(parameter_group_vars.values())