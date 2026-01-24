#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_ROOT = PROJECT_ROOT / 'models'
for p in [PROJECT_ROOT, MODELS_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from training.optimized_2class_training import train_optimized_2class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='5centers_multi')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--output_root', type=str, default='models/MedicalViT/_results')
    parser.add_argument('--no_pretrained', action='store_true')
    # performance args
    parser.add_argument('--oct_frames', type=int, default=48)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--persistent_workers', action='store_true', default=True)
    parser.add_argument('--oct_cache_dir', type=str, default='oct_cache_optimized')
    parser.add_argument('--vit_model', type=str, default='vit_base_patch16_224',
                       choices=['vit_base_patch16_224', 'vit_large_patch16_224', 'vit_small_patch16_224'])
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    output_dir = os.path.join(args.output_root, 'multimodal')
    os.makedirs(output_dir, exist_ok=True)

    train_optimized_2class(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        data_path=args.data_path,
        output_dir=output_dir,
        backbone='medical_vit',
        vit_model=args.vit_model,
        vit_pretrained=(not args.no_pretrained),
        oct_frames=args.oct_frames,
        input_size=args.input_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        oct_cache_dir=args.oct_cache_dir,
    )


if __name__ == '__main__':
    main()

