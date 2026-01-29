#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2: OCT Grad-CAM for ALL slices of a case.

For a chosen positive and negative case, this script:
  - Generates per-slice OCT CAM overlays (Orig + CAM) for ALL frames (F)
  - Saves them as individual PNGs, so you can montage them into a big figure.
"""

import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXPERIMENT_ROOT))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

from config import BioCOT_v3_2_Config
from models.bio_cot_v3_2 import create_bio_cot_v3_2
from data.dataset_v3_2 import FiveCentersMultimodalDatasetV3_2

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class OctOnlyWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, device: torch.device):
        super().__init__()
        self.model = base_model
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        f_oct = x.to(self.device)
        f_colpo = x.to(self.device)
        image_names = [f"oct_case_{i}.png" for i in range(B)]
        clinical_features = torch.zeros(B, 7, device=self.device)
        out = self.model(
            f_oct=f_oct,
            f_colpo=f_colpo,
            image_names=image_names,
            clinical_info=None,
            center_labels=None,
            clinical_features=clinical_features,
            return_loss_components=False,
            current_beta=0.1,
        )
        return out["logits"]


def denorm_to_rgb(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    return arr.astype(np.float32)


def pick_one_pos_neg_case(model, dataset, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    wrapper = OctOnlyWrapper(model, device).eval()

    pos_case = None
    neg_case = None
    for batch in loader:
        if pos_case is not None and neg_case is not None:
            break
        label = int(batch["label"].item())
        # use middle slice as representative for selection
        oct_images = batch["oct_images"][0]
        F = oct_images.size(0)
        mid = F // 2
        img_t = oct_images[mid]
        inp = img_t.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = wrapper(inp)
            prob_pos = torch.softmax(logits, 1)[0, 1].item()
            pred = int(torch.argmax(logits, 1)[0].item())
        correct = (pred == label)
        if label == 1 and pos_case is None and correct:
            pos_case = batch
        if label == 0 and neg_case is None and correct:
            neg_case = batch
    # fallback: accept any if needed
    if pos_case is None or neg_case is None:
        for batch in loader:
            if pos_case is None and int(batch["label"].item()) == 1:
                pos_case = batch
            if neg_case is None and int(batch["label"].item()) == 0:
                neg_case = batch
            if pos_case is not None and neg_case is not None:
                break
    return pos_case, neg_case


def save_oct_slices_with_cam(model, device, sample, tag, outdir):
    wrapper = OctOnlyWrapper(model, device).eval()
    target_layer = model.visual_encoder.vit.blocks[-1].norm1
    cam = EigenCAM(model=wrapper, target_layers=[target_layer])

    oct_images = sample["oct_images"][0]  # [F,C,H,W]
    label = int(sample["label"].item())
    F = oct_images.size(0)

    case_dir = outdir / f"OCT_{tag}_all_slices"
    case_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(F):
        img_t = oct_images[idx]
        inp = img_t.unsqueeze(0).to(device)
        targets = [ClassifierOutputTarget(label)]
        with torch.no_grad():
            logits = wrapper(inp)
            prob_pos = torch.softmax(logits, 1)[0, 1].item()
        grayscale_cam = cam(input_tensor=inp, targets=targets)[0]
        grayscale_cam = grayscale_cam - grayscale_cam.min()
        grayscale_cam = grayscale_cam / (grayscale_cam.max() + 1e-8)
        grayscale_cam = np.power(grayscale_cam, 0.5)

        rgb = denorm_to_rgb(img_t)
        overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

        # save individual overlays (just CAM overlay,方便你自己排版)
        save_path = case_dir / f"slice_{idx:03d}_label{label}_p{prob_pos:.2f}.png"
        plt.imsave(save_path, overlay)


def main():
    print("=" * 80)
    print("Bio-COT 3.2: OCT Grad-CAM for ALL slices (per case)")
    print("=" * 80)

    config = BioCOT_v3_2_Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("📦 Loading Bio-COT 3.2 model...")
    model = create_bio_cot_v3_2(config).to(device).eval()

    ckpt_dir = Path(config.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = Path(__file__).resolve().parents[2] / ckpt_dir
    ckpts = sorted(ckpt_dir.glob("best_model*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        ckpt_path = ckpts[0]
        print(f"📥 Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = state.get("model_state_dict", state)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Checkpoint loaded.")
    else:
        print(f"⚠️ No checkpoint found in {ckpt_dir}, CAM will not reflect trained model.")

    print("📂 Loading dataset...")
    csv_path = Path(config.data_root) / "temp_val_labels.csv"
    dataset = FiveCentersMultimodalDatasetV3_2(
        csv_path=str(csv_path),
        data_root=str(config.data_root),
    )

    print("🔍 Picking one positive and one negative OCT case...")
    pos_sample, neg_sample = pick_one_pos_neg_case(model, dataset, device)
    if pos_sample is None or neg_sample is None:
        print("⚠️ Could not find both pos and neg cases for OCT.")
        return

    outdir = EXPERIMENT_ROOT / "visualization" / "figures"
    print("🎨 Saving all-slice OCT CAM overlays (positive case)...")
    save_oct_slices_with_cam(model, device, pos_sample, "Pos", outdir)
    print("🎨 Saving all-slice OCT CAM overlays (negative case)...")
    save_oct_slices_with_cam(model, device, neg_sample, "Neg", outdir)

    print("✅ Done. Check OCT_Pos_all_slices/ and OCT_Neg_all_slices/ under visualization/figures.")


if __name__ == "__main__":
    main()


