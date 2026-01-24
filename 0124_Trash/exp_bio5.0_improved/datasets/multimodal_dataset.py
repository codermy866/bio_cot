from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Dict, Any

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# 复用 exp_bio5.0 已跑通的Dataset实现（含 clinical_features / image_name / clinical_info_str）
# 注意：exp_bio5.0 目录名包含 '.'，无法用标准包名导入；这里通过 sys.path 插入实现复用。
import sys

EXP_DIR = Path(__file__).resolve().parents[2]  # .../experiments
EXP_BIO5_DIR = EXP_DIR / "exp_bio5.0"
sys.path.insert(0, str(EXP_BIO5_DIR))
from data.dataset_v5 import FiveCentersMultimodalDatasetV5  # type: ignore


ImageSource = Literal["colposcopy", "oct_first"]


@dataclass
class DatasetArgs:
    csv_path: str
    vlm_json_path: Optional[str]
    image_source: ImageSource = "colposcopy"
    oct_num_frames: int = 20
    max_col_images: int = 3


class BioCOT_MultimodalDataset(Dataset):
    """
    HM-VR 训练用数据集封装：
    - 输出单张主图像 `image`：[3,224,224]
    - 输出临床向量 `clinical`：[C]
    - 输出标签 `label`：long
    - 输出 `image_name`（用于 VLM cache 检索）
    - 输出 `clinical_info_str`（用于 VLM prompt）
    """

    def __init__(self, args: DatasetArgs, transform=None):
        self.args = args
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.base = FiveCentersMultimodalDatasetV5(
            csv_path=str(args.csv_path),
            vlm_json_path=str(args.vlm_json_path) if args.vlm_json_path else None,
            transform=self.transform,
            oct_num_frames=args.oct_num_frames,
            max_col_images=args.max_col_images,
            balance_negative_frames=True,
        )

    def __len__(self) -> int:
        return len(self.base)

    def _pick_primary_image(self, sample: Dict[str, Any]) -> torch.Tensor:
        if self.args.image_source == "colposcopy":
            img = sample["colposcopy_images"][0]  # [3,224,224]
        else:
            # 使用OCT第一帧
            img = sample["oct_images"][0]
        return img

    def _pick_image_name(self, sample: Dict[str, Any]) -> str:
        # 默认用 colposcopy 的文件名更契合 VLM 描述（通常为 .jpg/.png）
        if self.args.image_source == "colposcopy":
            return sample.get("col_image_name", sample.get("image_name", "unknown.jpg"))
        return sample.get("oct_image_name", sample.get("image_name", "unknown.jpg"))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base[idx]
        return {
            "image": self._pick_primary_image(sample),
            "clinical": sample["clinical_features"],
            "label": sample["label"],
            "center_idx": sample["center_idx"],
            "image_name": self._pick_image_name(sample),
            "clinical_info_str": sample.get("clinical_info_str", ""),
        }


