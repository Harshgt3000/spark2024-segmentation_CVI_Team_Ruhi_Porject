#!/usr/bin/env python3
"""
SPARK 2024 - SegFormer Training (Fixed for NaN issues)
======================================================
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import SegformerForSemanticSegmentation

from tqdm import tqdm
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================================
# Dataset
# ============================================================================

class SparkDataset(Dataset):
    """SPARK 2024 Dataset - Fixed version."""
    
    CLASSES = ('background', 'spacecraft_body', 'solar_panel')
    NUM_CLASSES = 3
    
    SPACECRAFT = [
        'Cheops', 'LisaPathfinder', 'ObservationSat1', 'Proba2',
        'Proba3', 'Proba3ocs', 'Smart1', 'Soho', 'VenusExpress', 'XMM Newton'
    ]
    
    def __init__(self, data_root, split='train', img_size=512, max_samples=None, augment=True):
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        
        # Simple transforms - NO GaussNoise issues
        if self.augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        
        self.samples = self._load_samples(max_samples)
        print(f"Loaded {len(self.samples)} {split} samples")
    
    def _load_samples(self, max_samples):
        samples = []
        img_base = os.path.join(self.data_root, 'images')
        mask_base = os.path.join(self.data_root, 'mask')
        
        for spacecraft in self.SPACECRAFT:
            img_dir = os.path.join(img_base, spacecraft, self.split)
            mask_dir = os.path.join(mask_base, spacecraft, self.split)
            
            if not os.path.exists(img_dir):
                continue
            
            count = 0
            for fname in sorted(os.listdir(img_dir)):
                if not fname.endswith('_img.jpg') or fname.startswith('.'):
                    continue
                
                if max_samples and count >= max_samples:
                    break
                
                mask_fname = fname.replace('_img.jpg', '_layer.jpg')
                img_path = os.path.join(img_dir, fname)
                mask_path = os.path.join(mask_dir, mask_fname)
                
                if os.path.exists(mask_path):
                    samples.append((img_path, mask_path))
                    count += 1
        
        return samples
    
    def _rgb_to_label(self, rgb):
        """Convert RGB mask to labels."""
        label = np.zeros(rgb.shape[:2], dtype=np.int64)
        
        # Red -> Body (class 1)
        red_mask = (rgb[:,:,0] > 200) & (rgb[:,:,1] < 100) & (rgb[:,:,2] < 100)
        label[red_mask] = 1
        
        # Blue -> Panel (class 2)
        blue_mask = (rgb[:,:,0] < 100) & (rgb[:,:,1] < 100) & (rgb[:,:,2] > 200)
        label[blue_mask] = 2
        
        return label
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # Load
        image = np.array(Image.open(img_path).convert('RGB'))
        mask_rgb = np.array(Image.open(mask_path).convert('RGB'))
        label = self._rgb_to_label(mask_rgb)
        
        # Transform
        transformed = self.transform(image=image, mask=label)
        image = transformed['image']  # (3, H, W) float tensor
        label = transformed['mask']   # (H, W) int tensor
        
        return {
            'pixel_values': image,
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# Training
# ============================================================================

def compute_iou(pred, target, num_classes):
    """Compute per-class IoU."""
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
        else:
            ious.append(1.0 if intersection == 0 else 0.0)
    return ious


@torch.no_grad()
def evaluate(model, loader, device, num_classes, img_size):
    model.eval()
    all_ious = {c: [] for c in range(num_classes)}
    
    for batch in tqdm(loader, desc="Eval", leave=False):
        images = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(pixel_values=images).logits
        logits = F.interpolate(logits, size=(img_size, img_size), mode='bilinear', align_corners=False)
        preds = logits.argmax(dim=1)
        
        for pred, label in zip(preds, labels):
            ious = compute_iou(pred.cpu(), label.cpu(), num_classes)
            for c, iou in enumerate(ious):
                all_ious[c].append(iou)
    
    class_ious = {c: np.mean(v) for c, v in all_ious.items()}
    miou = np.mean(list(class_ious.values()))
    return miou, class_ious


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Output dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'segformer_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data
    train_ds = SparkDataset(args.data_root, 'train', args.img_size, args.max_samples, augment=True)
    val_ds = SparkDataset(args.data_root, 'val', args.img_size, args.max_samples, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    # Model
    print(f"Loading: {args.model}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model,
        num_labels=3,
        ignore_mismatched_sizes=True
    ).to(device)
    
    # Optimizer - LOWER learning rate to prevent NaN
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Loss - NO class weights initially
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # Training
    best_miou = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward
            outputs = model(pixel_values=images)
            logits = outputs.logits
            
            # Upsample logits to label size
            logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            
            # Loss
            loss = criterion(logits, labels)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"WARNING: NaN loss detected, skipping batch")
                optimizer.zero_grad()
                continue
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Evaluate
        if (epoch + 1) % args.eval_interval == 0:
            miou, class_ious = evaluate(model, val_loader, device, 3, args.img_size)
            
            print(f"  mIoU: {miou*100:.2f}%")
            for c, name in enumerate(SparkDataset.CLASSES):
                print(f"    {name}: {class_ious[c]*100:.2f}%")
            
            if miou > best_miou:
                best_miou = miou
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'miou': miou,
                    'class_ious': class_ious
                }, output_dir / 'best_model.pth')
                print(f"  Saved best model!")
        
        # Checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f'checkpoint_epoch{epoch+1}.pth')
    
    print(f"\nTraining complete! Best mIoU: {best_miou*100:.2f}%")
    print(f"Models saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/scratch/users/hchaurasia/spark2024/data')
    parser.add_argument('--model', default='nvidia/segformer-b0-finetuned-ade-512-512')
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-5)  # Lower LR
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--eval-interval', type=int, default=5)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--output-dir', default='/scratch/users/hchaurasia/spark2024/outputs')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
