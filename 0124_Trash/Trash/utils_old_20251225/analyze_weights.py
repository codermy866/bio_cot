#!/usr/bin/env python3
import os, torch
ckpt="cnn_training_latest/best_model.pth"
state=torch.load(ckpt,map_location="cpu")
sd=state.get("state_dict", state)
keys=list(sd.keys())
print("total_keys",len(keys))
# print top-level module prefixes
from collections import defaultdict
pref=defaultdict(int)
for k in keys:
    root=k.split(".")[0]
    pref[root]+=1
print("roots:",sorted(pref.items(), key=lambda x:-x[1])[:10])
# sample some keys per root
for root in ["oct_encoder","col_encoder","clinical_encoder","cross_modal_attn","fusion_layers","attention_pool","classifier"]:
    ks=[k for k in keys if k.startswith(root)]
    print("\n[",root,"] count=",len(ks))
    for k in ks[:20]:
        v=sd[k]
        shape=getattr(v,'shape',None)
        print(k, tuple(shape) if shape is not None else type(v))
    if ks:
        print("last:", ks[-1], tuple(getattr(sd[ks[-1]],'shape',())) )
# try infer embed dims
for name in [k for k in keys if k.endswith("classifier.12.weight") or k.endswith("classifier.0.weight")]:
    print("DIMHINT", name, sd[name].shape)