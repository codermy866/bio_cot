#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bio-COT 3.2: Multiphase (Colposcopy) + Volumetric (OCT) Explainability Generator

What this script does (paper-ready outputs):
- For each selected case:
  1) OCT volume slice importance curve (per-slice probability / logit)
  2) Top-K OCT slices CAM overlays (using ImprovedGradCAM: feature gradients + VisualNotes attention when available)
  3) Colposcopy multi-phase (up to 3 images) per-phase CAM overlays
  4) Fusion gating weights (w_oct, w_colpo) for the case
- For the whole dataset subset:
  5) Fusion gating weight distribution plots (by label / by center)

Notes / assumptions:
- Dataset returns:
  - oct_images: [B, F, 3, 224, 224] (F=config.oct_frames)
  - colposcopy_images: [B, N, 3, 224, 224] (N=config.colposcopy_images)
  - clinical_features: [B, 7]
  - label: [B]
  - center_idx: [B]
  - image_names: List[str] or str (for VLM retriever)
- Training uses ViT patch features and averages across frames/images; here we run *replacement inference*
  to recover per-frame/per-phase importance without changing the model.
"""

import argparse
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt
import logging

from torch.utils.data import DataLoader

# Matplotlib defaults (avoid noisy Calibri warnings on Linux)
plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["figure.dpi"] = 300
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Local imports (exp_bio3.2)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import BioCOT_v3_2_Config
from models.bio_cot_v3_2 import create_bio_cot_v3_2
from data.dataset_v3_2 import FiveCentersMultimodalDatasetV3_2

local_training_path = Path(__file__).resolve().parent.parent.parent / "training"
sys.path.insert(0, str(local_training_path))
from extract_vit_patches import extract_patch_features_with_vit

# Reuse CAM utilities from improved CAM generator
from generate_cam_improved import ImprovedGradCAM, overlay_cam_on_image, tensor_to_image


def _ensure_list(x, batch_size: int):
    if x is None:
        return None
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [x] * batch_size
    try:
        return list(x)
    except Exception:
        return [str(x)] * batch_size


@torch.no_grad()
def _forward_logits(model, f_oct, f_colpo, image_names, clinical_features, center_labels=None, clinical_info=None):
    out = model(
        f_oct=f_oct,
        f_colpo=f_colpo,
        image_names=image_names,
        clinical_info=clinical_info,
        center_labels=center_labels,
        clinical_features=clinical_features,
        return_loss_components=False,
        current_beta=0.1,  # explainability: stable "filtered" regime; consistent with later epochs
    )
    logits = out["logits"]
    return logits, out


def compute_oct_slice_importance(
    model,
    oct_images_1: torch.Tensor,   # [1, F, 3, 224, 224]
    colpo_images_1: torch.Tensor, # [1, N, 3, 224, 224]
    clinical_features_1: torch.Tensor, # [1, 7]
    image_names_1,
    center_labels_1=None,
    device=None,
    vit_batch_size: int = 8,
):
    """
    Replacement inference:
    - baseline: f_oct = mean over F slices
    - for each slice t: f_oct = slice_t features
    Returns per-slice positive probability and per-slice logit (class=1).
    """
    model.eval()
    device = device or oct_images_1.device

    # OCT features per slice: [F, 196, 768]
    F = oct_images_1.shape[1]
    oct_flat = oct_images_1.view(F, *oct_images_1.shape[2:])  # [F, 3, 224, 224]
    oct_feats = extract_patch_features_with_vit(oct_flat, device, batch_size=vit_batch_size)  # [F,196,768]

    # Colpo mean features: [1,196,768]
    N = colpo_images_1.shape[1]
    colpo_flat = colpo_images_1.view(N, *colpo_images_1.shape[2:])
    colpo_feats_all = extract_patch_features_with_vit(colpo_flat, device, batch_size=vit_batch_size)  # [N,196,768]
    colpo_feat_mean = colpo_feats_all.mean(dim=0, keepdim=True)  # [1,196,768]

    # Baseline: OCT mean
    oct_feat_mean = oct_feats.mean(dim=0, keepdim=True)  # [1,196,768]

    # Slice-wise forward
    image_names_1 = _ensure_list(image_names_1, 1)
    probs = []
    logits_1 = []
    for t in range(F):
        f_oct_t = oct_feats[t : t + 1]
        logits, _ = _forward_logits(
            model=model,
            f_oct=f_oct_t,
            f_colpo=colpo_feat_mean,
            image_names=image_names_1,
            clinical_features=clinical_features_1,
            center_labels=center_labels_1,
            clinical_info=None,
        )
        p = torch.softmax(logits, dim=1)[0, 1].item()
        l = logits[0, 1].item()
        probs.append(p)
        logits_1.append(l)

    # Baseline probability using mean features (for reference)
    logits_base, out_base = _forward_logits(
        model=model,
        f_oct=oct_feat_mean,
        f_colpo=colpo_feat_mean,
        image_names=image_names_1,
        clinical_features=clinical_features_1,
        center_labels=center_labels_1,
        clinical_info=None,
    )
    p_base = torch.softmax(logits_base, dim=1)[0, 1].item()

    return {
        "oct_feats_per_slice": oct_feats,          # [F,196,768]
        "oct_feat_mean": oct_feat_mean,            # [1,196,768]
        "colpo_feat_mean": colpo_feat_mean,        # [1,196,768]
        "slice_probs": np.array(probs, dtype=float),
        "slice_logits_pos": np.array(logits_1, dtype=float),
        "baseline_prob": float(p_base),
        "baseline_out": out_base,
    }


def compute_colpo_phase_importance(
    model,
    oct_images_1: torch.Tensor,   # [1, F, 3, 224, 224]
    colpo_images_1: torch.Tensor, # [1, N, 3, 224, 224]
    clinical_features_1: torch.Tensor, # [1, 7]
    image_names_1,
    center_labels_1=None,
    device=None,
    vit_batch_size: int = 8,
):
    """
    Replacement inference across colposcopy phases (N images).
    - baseline: colpo mean across N images
    - per phase i: use that single image's features
    Returns per-phase positive probability and logits.
    """
    model.eval()
    device = device or oct_images_1.device

    # OCT mean features: [1,196,768]
    F = oct_images_1.shape[1]
    oct_flat = oct_images_1.view(F, *oct_images_1.shape[2:])
    oct_feats_all = extract_patch_features_with_vit(oct_flat, device, batch_size=vit_batch_size)  # [F,196,768]
    oct_feat_mean = oct_feats_all.mean(dim=0, keepdim=True)  # [1,196,768]

    # Colpo per-phase features: [N,196,768]
    N = colpo_images_1.shape[1]
    colpo_flat = colpo_images_1.view(N, *colpo_images_1.shape[2:])
    colpo_feats = extract_patch_features_with_vit(colpo_flat, device, batch_size=vit_batch_size)  # [N,196,768]
    colpo_feat_mean = colpo_feats.mean(dim=0, keepdim=True)  # [1,196,768]

    image_names_1 = _ensure_list(image_names_1, 1)

    probs = []
    logits_pos = []
    for i in range(N):
        logits, _ = _forward_logits(
            model=model,
            f_oct=oct_feat_mean,
            f_colpo=colpo_feats[i : i + 1],
            image_names=image_names_1,
            clinical_features=clinical_features_1,
            center_labels=center_labels_1,
            clinical_info=None,
        )
        probs.append(torch.softmax(logits, dim=1)[0, 1].item())
        logits_pos.append(logits[0, 1].item())

    logits_base, _ = _forward_logits(
        model=model,
        f_oct=oct_feat_mean,
        f_colpo=colpo_feat_mean,
        image_names=image_names_1,
        clinical_features=clinical_features_1,
        center_labels=center_labels_1,
        clinical_info=None,
    )
    p_base = torch.softmax(logits_base, dim=1)[0, 1].item()

    return {
        "oct_feat_mean": oct_feat_mean,
        "colpo_feats_per_phase": colpo_feats,      # [N,196,768]
        "colpo_feat_mean": colpo_feat_mean,        # [1,196,768]
        "phase_probs": np.array(probs, dtype=float),
        "phase_logits_pos": np.array(logits_pos, dtype=float),
        "baseline_prob": float(p_base),
    }


def _reshape_patch_cam(cam_1d: np.ndarray) -> np.ndarray:
    cam_flat = cam_1d.flatten()
    n = cam_flat.shape[0]
    side = int(np.sqrt(n))
    if side * side == n:
        return cam_flat.reshape(side, side)
    # fallback: truncate to closest square
    return cam_flat[: side * side].reshape(side, side)


def plot_case_figure(
    save_path: Path,
    case_id: str,
    label: int,
    center_idx: int,
    oct_images_1: torch.Tensor,     # [1,F,3,224,224]
    colpo_images_1: torch.Tensor,   # [1,N,3,224,224]
    oct_info: dict,
    colpo_info: dict,
    cams: dict,
):
    """
    Paper-friendly single-case panel.
    Layout:
    Row 1: OCT importance curve + baseline prob
    Row 2: Top-3 OCT slice overlays
    Row 3: Colpo phase overlays (up to 3)
    Row 4: Fusion weights (w_oct/w_colpo)
    """
    label_name = "Positive" if int(label) == 1 else "Negative"

    slice_probs = oct_info["slice_probs"]
    baseline_prob = oct_info["baseline_prob"]

    # Choose top-K slices by prob (pos-class)
    top_k = min(3, len(slice_probs))
    top_idx = list(np.argsort(-slice_probs)[:top_k])
    top_idx.sort()

    # Prepare images
    F = oct_images_1.shape[1]
    N = colpo_images_1.shape[1]

    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(4, 6, hspace=0.6, wspace=0.35)

    # Row 1: OCT slice importance curve
    ax_curve = fig.add_subplot(gs[0, :4])
    ax_curve.plot(np.arange(F), slice_probs, linewidth=2)
    ax_curve.axhline(baseline_prob, linestyle="--", linewidth=2, alpha=0.7)
    for t in top_idx:
        ax_curve.axvline(t, linestyle=":", alpha=0.6)
    ax_curve.set_title(f"OCT slice importance (replacement inference)\nBaseline p(pos)={baseline_prob:.3f}")
    ax_curve.set_xlabel("Slice index")
    ax_curve.set_ylabel("p(positive)")
    ax_curve.set_ylim(0, 1)
    ax_curve.grid(True, alpha=0.25)

    # Row 1: fusion weights
    w_oct = float(cams["fusion_weights"]["oct"])
    w_colpo = float(cams["fusion_weights"]["colpo"])
    ax_w = fig.add_subplot(gs[0, 4:])
    ax_w.bar(["OCT", "Colpo"], [w_oct, w_colpo], color=["#4C72B0", "#DD8452"])
    ax_w.set_ylim(0, 1)
    ax_w.set_title("Adaptive modality gating weights")
    ax_w.set_ylabel("Weight")
    ax_w.grid(True, axis="y", alpha=0.25)

    # Row 2: Top-K OCT overlays
    for j, t in enumerate(top_idx):
        oct_tensor = oct_images_1[0, t]  # [3,224,224]
        oct_img = tensor_to_image(oct_tensor.unsqueeze(0))  # reuse helper expects batch-ish
        cam_2d = cams["oct_cam_2d_by_slice"][t]
        overlay = overlay_cam_on_image(oct_img, cam_2d, alpha=0.6, threshold=0.2)
        ax = fig.add_subplot(gs[1, j * 2 : (j + 1) * 2])
        ax.imshow(overlay)
        ax.set_title(f"OCT slice {t} (p={slice_probs[t]:.3f})")
        ax.axis("off")

    # Fill any remaining OCT slots
    for j in range(len(top_idx), 3):
        ax = fig.add_subplot(gs[1, j * 2 : (j + 1) * 2])
        ax.axis("off")

    # Row 3: Colpo phase overlays (up to 3)
    phase_probs = colpo_info["phase_probs"]
    for i in range(min(3, N)):
        col_tensor = colpo_images_1[0, i]  # [3,224,224]
        col_img = tensor_to_image(col_tensor.unsqueeze(0))
        cam_2d = cams["colpo_cam_2d_by_phase"][i]
        overlay = overlay_cam_on_image(col_img, cam_2d, alpha=0.6, threshold=0.2)
        ax = fig.add_subplot(gs[2, i * 2 : (i + 1) * 2])
        ax.imshow(overlay)
        ax.set_title(f"Colpo phase {i+1} (p={phase_probs[i]:.3f})")
        ax.axis("off")

    for i in range(min(3, N), 3):
        ax = fig.add_subplot(gs[2, i * 2 : (i + 1) * 2])
        ax.axis("off")

    # Row 4: heatmap thumbnails (optional) – show CAM grids for top OCT slice + phase 1
    ax_h1 = fig.add_subplot(gs[3, :3])
    if top_idx:
        ax_h1.imshow(cams["oct_cam_2d_by_slice"][top_idx[-1]], cmap="RdYlBu_r", vmin=0, vmax=1)
        ax_h1.set_title("OCT CAM (patch grid)")
    ax_h1.axis("off")

    ax_h2 = fig.add_subplot(gs[3, 3:])
    ax_h2.imshow(cams["colpo_cam_2d_by_phase"][0], cmap="RdYlBu_r", vmin=0, vmax=1)
    ax_h2.set_title("Colpo CAM (patch grid) – phase 1")
    ax_h2.axis("off")

    fig.suptitle(f"Case={case_id} | Label={label_name} | Center={center_idx}", fontsize=16, fontweight="bold", y=0.99)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_fusion_weight_distributions(records, save_dir: Path):
    """
    records: list of dict {label, center_idx, w_oct, w_colpo}
    """
    import pandas as pd

    df = pd.DataFrame.from_records(records)
    save_dir.mkdir(parents=True, exist_ok=True)

    # By label
    fig = plt.figure(figsize=(10, 4), dpi=300)
    ax1 = fig.add_subplot(1, 2, 1)
    for lab in sorted(df["label"].unique()):
        sub = df[df["label"] == lab]
        ax1.hist(sub["w_oct"], bins=20, alpha=0.6, label=f"label={lab} (w_oct)")
    ax1.set_title("Gating weight w_oct by label")
    ax1.set_xlabel("w_oct")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(1, 2, 2)
    for lab in sorted(df["label"].unique()):
        sub = df[df["label"] == lab]
        ax2.hist(sub["w_colpo"], bins=20, alpha=0.6, label=f"label={lab} (w_colpo)")
    ax2.set_title("Gating weight w_colpo by label")
    ax2.set_xlabel("w_colpo")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    fig.savefig(save_dir / "fusion_weights_by_label.png", bbox_inches="tight")
    plt.close(fig)

    # By center (boxplot)
    centers = sorted(df["center_idx"].unique())
    fig = plt.figure(figsize=(10, 4), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    data = [df[df["center_idx"] == c]["w_oct"].values for c in centers]
    ax.boxplot(data, labels=[str(c) for c in centers], showfliers=False)
    ax.set_title("w_oct by center")
    ax.set_xlabel("center_idx")
    ax.set_ylabel("w_oct")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(save_dir / "fusion_weights_w_oct_by_center.png", bbox_inches="tight")
    plt.close(fig)


def _auto_find_checkpoint(ckpt_dir: Path) -> Path | None:
    """
    Find a reasonable checkpoint automatically.
    Preference:
    - best_model*.pth under ckpt_dir (newest mtime)
    - otherwise: any *.pth (newest mtime)
    """
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.glob("best_model*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    candidates = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="CSV path (default: temp_val_labels.csv under data_root)")
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root (5centers_multi_leave_centers_out)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint .pth (default: config.checkpoint_dir/best_model.pth)")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: exp_bio3.2/visualization/figures)")
    parser.add_argument("--num_cases", type=int, default=12, help="Number of cases to generate single-case panels")
    parser.add_argument("--max_batches_stats", type=int, default=200, help="Max batches to collect gating weight stats")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu / cuda:0 ...")
    args = parser.parse_args()

    config = BioCOT_v3_2_Config()
    if args.data_root is not None:
        config.data_root = args.data_root

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    csv_path = Path(args.csv) if args.csv else Path(config.data_root) / "temp_val_labels.csv"
    outdir = Path(args.outdir) if args.outdir else (Path(__file__).parent.parent / "figures")
    outdir.mkdir(parents=True, exist_ok=True)
    case_dir = outdir / "cases_multiphase_oct"
    case_dir.mkdir(parents=True, exist_ok=True)

    # Model
    print("📦 Loading model...")
    model = create_bio_cot_v3_2(config).to(device).eval()

    # Resolve checkpoint directory robustly (config.checkpoint_dir is often relative)
    exp_root = Path(__file__).resolve().parents[2]  # .../experiments/exp_bio3.2
    ckpt_dir = Path(config.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = exp_root / ckpt_dir

    ckpt_path = Path(args.checkpoint) if args.checkpoint else _auto_find_checkpoint(ckpt_dir)
    if ckpt_path is not None and ckpt_path.exists():
        print(f"📥 Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print("✅ Checkpoint loaded")
    else:
        print(f"⚠️ Checkpoint not found under {ckpt_dir}. Please pass --checkpoint. (will run with current weights)")

    # Data
    print("📂 Loading dataset...")
    dataset = FiveCentersMultimodalDatasetV3_2(
        csv_path=str(csv_path),
        data_root=str(config.data_root),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    grad_cam = ImprovedGradCAM(model)

    gating_records = []
    cases_done = 0

    for batch_idx, batch in enumerate(loader):
        oct_images = batch["oct_images"].to(device)
        colpo_images = batch["colposcopy_images"].to(device)
        clinical_features = batch.get("clinical_features", batch.get("clinical", None)).to(device)
        label = int(batch["label"].item())
        center_idx = int(batch.get("center_idx", torch.tensor([-1])).item())
        image_names = batch.get("image_names", batch.get("image_name", "unknown.jpg"))
        image_names = _ensure_list(image_names, 1)

        # --- collect gating stats (cheap: use mean features) ---
        if batch_idx < args.max_batches_stats:
            # mean features (same as training)
            F = oct_images.shape[1]
            oct_feats = extract_patch_features_with_vit(
                oct_images.view(F, *oct_images.shape[2:]),
                device,
                batch_size=max(4, int(config.vit_batch_size // 2)),
            ).mean(dim=0, keepdim=True)

            N = colpo_images.shape[1]
            colpo_feats = extract_patch_features_with_vit(
                colpo_images.view(N, *colpo_images.shape[2:]),
                device,
                batch_size=max(4, int(config.vit_batch_size // 2)),
            ).mean(dim=0, keepdim=True)

            logits, out = _forward_logits(
                model=model,
                f_oct=oct_feats,
                f_colpo=colpo_feats,
                image_names=image_names,
                clinical_features=clinical_features,
                center_labels=batch.get("center_idx", None).to(device) if "center_idx" in batch else None,
                clinical_info=None,
            )
            fw = out.get("fusion_weights", {})
            w_oct = float(fw.get("oct", torch.tensor([[np.nan]])).view(-1)[0].item())
            w_colpo = float(fw.get("colpo", torch.tensor([[np.nan]])).view(-1)[0].item())
            gating_records.append(
                {"label": label, "center_idx": center_idx, "w_oct": w_oct, "w_colpo": w_colpo}
            )

        # --- generate case panels ---
        if cases_done < args.num_cases:
            # Per-slice importance + caches
            oct_info = compute_oct_slice_importance(
                model=model,
                oct_images_1=oct_images,
                colpo_images_1=colpo_images,
                clinical_features_1=clinical_features,
                image_names_1=image_names,
                center_labels_1=batch.get("center_idx", None).to(device) if "center_idx" in batch else None,
                device=device,
                vit_batch_size=max(4, int(config.vit_batch_size // 2)),
            )
            colpo_info = compute_colpo_phase_importance(
                model=model,
                oct_images_1=oct_images,
                colpo_images_1=colpo_images,
                clinical_features_1=clinical_features,
                image_names_1=image_names,
                center_labels_1=batch.get("center_idx", None).to(device) if "center_idx" in batch else None,
                device=device,
                vit_batch_size=max(4, int(config.vit_batch_size // 2)),
            )

            # CAMs for each OCT slice (only compute 2D grid, cheap)
            F = oct_images.shape[1]
            oct_cam_2d_by_slice = []
            for t in range(F):
                # CAM for single-slice features, with colpo mean
                oct_cam_1d, _ = grad_cam.generate_cam(
                    oct_features=oct_info["oct_feats_per_slice"][t : t + 1],
                    colpo_features=oct_info["colpo_feat_mean"],
                    image_names=image_names,
                    clinical_features=clinical_features,
                    target_class=label,
                )
                oct_cam_2d_by_slice.append(_reshape_patch_cam(oct_cam_1d))

            # CAMs for each colpo phase
            N = colpo_images.shape[1]
            colpo_cam_2d_by_phase = []
            for i in range(N):
                _, col_cam_1d = grad_cam.generate_cam(
                    oct_features=colpo_info["oct_feat_mean"],
                    colpo_features=colpo_info["colpo_feats_per_phase"][i : i + 1],
                    image_names=image_names,
                    clinical_features=clinical_features,
                    target_class=label,
                )
                colpo_cam_2d_by_phase.append(_reshape_patch_cam(col_cam_1d))

            # Get fusion weights for the baseline of this case
            fw = oct_info["baseline_out"].get("fusion_weights", {})
            w_oct = float(fw.get("oct", torch.tensor([[0.5]])).view(-1)[0].item())
            w_colpo = float(fw.get("colpo", torch.tensor([[0.5]])).view(-1)[0].item())

            cams = {
                "oct_cam_2d_by_slice": oct_cam_2d_by_slice,
                "colpo_cam_2d_by_phase": colpo_cam_2d_by_phase,
                "fusion_weights": {"oct": w_oct, "colpo": w_colpo},
            }

            case_id = f"case_{batch_idx:04d}"
            fig_path = case_dir / f"{case_id}_label{label}_center{center_idx}.png"
            plot_case_figure(
                save_path=fig_path,
                case_id=case_id,
                label=label,
                center_idx=center_idx,
                oct_images_1=oct_images.detach().cpu(),
                colpo_images_1=colpo_images.detach().cpu(),
                oct_info=oct_info,
                colpo_info=colpo_info,
                cams=cams,
            )
            cases_done += 1
            print(f"✅ Saved case panel: {fig_path.name}")

        # early stop when both are satisfied
        if cases_done >= args.num_cases and batch_idx >= args.max_batches_stats:
            break

    # Save gating stats
    if gating_records:
        plot_fusion_weight_distributions(gating_records, outdir)
        print(f"✅ Saved fusion weight distributions to: {outdir}")

    print("✅ Done.")


if __name__ == "__main__":
    main()


