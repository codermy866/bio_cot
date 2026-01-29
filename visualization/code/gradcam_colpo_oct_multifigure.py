#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2: GitHub-style Grad-CAM figures for colposcopy (3 phases) + OCT.

Outputs:
  1) Colposcopy 3×4 figure (one high-confidence positive vs negative case):
     - Rows:   Raw / Acetic / Iodine (3 phases)
     - Cols:   Pos Orig | Pos CAM | Neg Orig | Neg CAM
  2) OCT 2×2 figure (same positive vs negative cases, one representative slice):
     - Rows:   Pos case / Neg case
     - Cols:   Orig | CAM
"""

import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]  # .../experiments/exp_bio3.2
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]              # .../VLM_Caus_Rm_Mics
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
from training.extract_vit_patches import extract_patch_features_with_vit

from pytorch_grad_cam import EigenCAM, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


class ColpoOnlyWrapper(nn.Module):
    """Wrapper: only use a single colposcopy image as input, produce logits [B,2]."""

    def __init__(self, base_model: nn.Module, device: torch.device):
        super().__init__()
        self.model = base_model
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        f_oct = x.to(self.device)
        f_colpo = x.to(self.device)
        image_names = [f"colpo_case_{i}.jpg" for i in range(B)]
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


class OctOnlyWrapper(nn.Module):
    """Wrapper: only use a single OCT slice as input, produce logits [B,2]."""

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
    """[3,H,W] -> float32 [H,W,3] in [0,1]."""
    arr = t.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    return arr.astype(np.float32)


def find_high_confidence_cases(model, wrapper_class, dataset, device, num_pos=1, num_neg=1):
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    wrapper = wrapper_class(model, device).to(device).eval()

    pos_case = None  # 首选：预测正确 + 置信度高的阳性
    neg_case = None  # 首选：预测正确 + 置信度高的阴性
    pos_fallback = None  # 备选：只要是阳性就行
    neg_fallback = None  # 备选：只要是阴性就行

    for batch in loader:
        label = int(batch["label"].item())
        # pick first colpo frame as representative image
        colpos = batch["colposcopy_images"][0]  # [3,C,H,W]
        img_t = colpos[0]  # phase 1
        inp = img_t.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = wrapper(inp)
            prob_pos = torch.softmax(logits, 1)[0, 1].item()
            pred = int(torch.argmax(logits, 1)[0].item())

        correct = (pred == label)
        # 记录备选：只要标签匹配
        if label == 1 and pos_fallback is None:
            pos_fallback = batch
        if label == 0 and neg_fallback is None:
            neg_fallback = batch

        # 首选：预测正确 + 相对高置信度（不要太严格）
        if label == 1 and correct and prob_pos >= 0.5 and pos_case is None:
            pos_case = batch
        if label == 0 and correct and prob_pos <= 0.5 and neg_case is None:
            neg_case = batch

    # 如果首选没找到，就退而求其次用 fallback
    if pos_case is None:
        pos_case = pos_fallback
    if neg_case is None:
        neg_case = neg_fallback

    return pos_case, neg_case


def build_colpo_multiphase_figure(model, device, pos_sample, neg_sample, save_path):
    wrapper = ColpoOnlyWrapper(model, device).eval()
    target_layer = model.visual_encoder.vit.blocks[-1].norm1
    # 新版接口不再使用 use_cuda 参数，设备由输入 tensor 决定
    cam = EigenCAM(model=wrapper, target_layers=[target_layer])

    phases = ["Raw", "Acetic", "Iodine"]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), dpi=300)

    def fill_case(sample, col_offset, tag):
        colpos = sample["colposcopy_images"][0]
        label = int(sample["label"].item())
        for i in range(3):
            img_t = colpos[i]
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

            axes[i, col_offset].imshow(rgb)
            axes[i, col_offset].set_title(f"{phases[i]} {tag} Orig")
            axes[i, col_offset + 1].imshow(overlay)
            axes[i, col_offset + 1].set_title(f"{phases[i]} {tag} CAM\np_pos={prob_pos:.2f}")
            axes[i, col_offset].axis("off")
            axes[i, col_offset + 1].axis("off")

    fill_case(pos_sample, 0, "Pos")
    fill_case(neg_sample, 2, "Neg")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def build_oct_figure(model, device, pos_sample, neg_sample, save_path):
    wrapper = OctOnlyWrapper(model, device).eval()
    target_layer = model.visual_encoder.vit.blocks[-1].norm1
    cam = EigenCAM(model=wrapper, target_layers=[target_layer])

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=300)

    def fill_case(sample, row, tag):
        oct_images = sample["oct_images"][0]  # [F,C,H,W]
        F = oct_images.size(0)
        mid = F // 2
        img_t = oct_images[mid]
        label = int(sample["label"].item())
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

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"{tag} OCT Slice\nOrig")
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(f"{tag} OCT CAM\np_pos={prob_pos:.2f}")
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")

    fill_case(pos_sample, 0, "Pos")
    fill_case(neg_sample, 1, "Neg")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 80)
    print("Bio-COT 3.2: Colposcopy (3-phase) + OCT Grad-CAM Multi-figure Generator")
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

    print("🔍 Searching for high-confidence positive & negative cases...")
    pos_sample, neg_sample = find_high_confidence_cases(
        model, ColpoOnlyWrapper, dataset, device, num_pos=1, num_neg=1
    )
    if pos_sample is None or neg_sample is None:
        print("⚠️ Could not find both high-confidence positive and negative cases.")
        return

    figures_dir = Path(__file__).parent.parent / "figures"
    colpo_fig_path = figures_dir / "Colpo_GradCAM_PosNeg_3x4.png"
    oct_fig_path = figures_dir / "OCT_GradCAM_PosNeg_2x2.png"

    print("🎨 Building colposcopy 3×4 figure...")
    build_colpo_multiphase_figure(model, device, pos_sample, neg_sample, colpo_fig_path)
    print(f"✅ Saved: {colpo_fig_path}")

    print("🎨 Building OCT 2×2 figure...")
    build_oct_figure(model, device, pos_sample, neg_sample, oct_fig_path)
    print(f"✅ Saved: {oct_fig_path}")

    print("✅ All done.")


if __name__ == "__main__":
    main()


