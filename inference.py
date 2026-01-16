#!/usr/bin/env python3
"""
SPARK 2024 - Inference Script
==============================
Generate predictions on test set for Codabench submission.

Usage:
    python inference.py --checkpoint outputs/best_model.pth --test-dir data/test --output predictions/
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from transformers import SegformerForSemanticSegmentation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test-dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='predictions')
    parser.add_argument('--model', type=str, default='nvidia/segformer-b5-finetuned-ade-640-640')
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def get_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def labels_to_rgb(labels):
    """Convert label map to RGB mask for submission.
    
    Class 0 (background) → Black (0, 0, 0)
    Class 1 (body) → Red (255, 0, 0)
    Class 2 (panels) → Blue (0, 0, 255)
    """
    H, W = labels.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    
    rgb[labels == 1] = [255, 0, 0]  # Body - Red
    rgb[labels == 2] = [0, 0, 255]  # Panels - Blue
    # Background stays black (0, 0, 0)
    
    return rgb


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model: {args.model}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model,
        num_labels=args.num_classes,
        ignore_mismatched_sizes=True
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    if 'miou' in checkpoint:
        print(f"Checkpoint mIoU: {checkpoint['miou']*100:.2f}%")
    
    # Find test images
    test_dir = Path(args.test_dir)
    image_files = list(test_dir.glob('**/*_img.jpg')) + list(test_dir.glob('**/*.jpg'))
    print(f"Found {len(image_files)} test images")
    
    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Transforms
    transform = get_transforms(args.img_size)
    
    # Run inference
    print("Generating predictions...")
    
    with torch.no_grad():
        for img_path in tqdm(image_files):
            # Load image
            image = np.array(Image.open(img_path).convert('RGB'))
            orig_h, orig_w = image.shape[:2]
            
            # Transform
            transformed = transform(image=image)
            pixel_values = transformed['image'].unsqueeze(0).to(device)
            
            # Forward
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            
            # Upsample to original size
            logits = F.interpolate(
                logits,
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=False
            )
            
            # Get prediction
            pred = logits.argmax(dim=1).squeeze().cpu().numpy()
            
            # Convert to RGB mask
            rgb_mask = labels_to_rgb(pred)
            
            # Save prediction
            # Output filename: same as input but with _pred suffix
            output_name = img_path.stem.replace('_img', '_pred') + '.png'
            output_path = output_dir / output_name
            
            Image.fromarray(rgb_mask).save(output_path)
    
    print(f"\nPredictions saved to: {output_dir}")
    print(f"Total: {len(image_files)} images")


if __name__ == '__main__':
    main()
