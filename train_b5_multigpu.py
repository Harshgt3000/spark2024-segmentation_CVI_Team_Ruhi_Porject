#!/usr/bin/env python3
"""
SPARK 2024 - SegFormer-B5 Multi-GPU Training
=============================================
Features:
- Multi-GPU with DataParallel
- Resume from checkpoint
- Frequent checkpoints
- Progress bars
- NaN protection
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

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
    """SPARK 2024 Dataset."""
    
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
        
        if self.augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5),
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
        print(f"[{split.upper()}] Loaded {len(self.samples)} samples from {len(self.SPACECRAFT)} spacecraft classes")
    
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
        
        image = np.array(Image.open(img_path).convert('RGB'))
        mask_rgb = np.array(Image.open(mask_path).convert('RGB'))
        label = self._rgb_to_label(mask_rgb)
        
        transformed = self.transform(image=image, mask=label)
        image = transformed['image']
        label = transformed['mask']
        
        return {
            'pixel_values': image,
            'labels': label.clone().detach().long() if isinstance(label, torch.Tensor) else torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# Metrics
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
    
    pbar = tqdm(loader, desc="Evaluating", leave=False, ncols=100)
    for batch in pbar:
        images = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        # Handle DataParallel
        if isinstance(model, nn.DataParallel):
            logits = model.module(pixel_values=images).logits
        else:
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


# ============================================================================
# Training
# ============================================================================

def train(args):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    
    print("=" * 60)
    print(" SPARK 2024 - SegFormer Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Number of GPUs: {num_gpus}")
    if torch.cuda.is_available():
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"         Memory: {mem:.1f} GB")
    print("=" * 60)
    
    # Output directory
    if args.resume:
        # Extract output dir from checkpoint path
        output_dir = Path(args.resume).parent
        print(f"Resuming training to: {output_dir}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(args.output_dir) / f'segformer_b5_{timestamp}'
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # Save config
    config = vars(args).copy()
    config['num_gpus'] = num_gpus
    config['output_dir'] = str(output_dir)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Data
    print("\nLoading datasets...")
    train_ds = SparkDataset(args.data_root, 'train', args.img_size, args.max_samples, augment=True)
    val_ds = SparkDataset(args.data_root, 'val', args.img_size, args.max_samples, augment=False)
    
    # Adjust batch size for multi-GPU
    effective_batch_size = args.batch_size * max(num_gpus, 1)
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Effective batch size: {effective_batch_size}")
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=effective_batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=effective_batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    # Model
    print(f"\nLoading model: {args.model}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model,
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    
    # Multi-GPU
    if num_gpus > 1:
        print(f"Using DataParallel with {num_gpus} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Count parameters
    if isinstance(model, nn.DataParallel):
        num_params = sum(p.numel() for p in model.module.parameters())
    else:
        num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer
    if isinstance(model, nn.DataParallel):
        optimizer = AdamW(model.module.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Scheduler with warmup
    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    
    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # Resume from checkpoint
    start_epoch = 0
    best_miou = 0.0
    
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'best_miou' in checkpoint:
            best_miou = checkpoint['best_miou']
        
        print(f"Resumed from epoch {start_epoch}, best mIoU: {best_miou*100:.2f}%")
    
    # Training loop
    print("\n" + "=" * 60)
    print(" Starting Training")
    print("=" * 60)
    print(f"Epochs: {start_epoch + 1} -> {args.epochs}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print("=" * 60 + "\n")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{args.epochs}", 
            ncols=100,
            leave=True
        )
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward
            if isinstance(model, nn.DataParallel):
                outputs = model.module(pixel_values=images)
            else:
                outputs = model(pixel_values=images)
            
            logits = outputs.logits
            logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            
            loss = criterion(logits, labels)
            
            # NaN check
            if torch.isnan(loss):
                print(f"\n[WARNING] NaN loss at batch {batch_idx}, skipping...")
                optimizer.zero_grad()
                continue
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.1e}'
            })
        
        scheduler.step()
        
        avg_loss = total_loss / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n[Epoch {epoch+1}] Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
        
        # Evaluate
        if (epoch + 1) % args.eval_interval == 0 or (epoch + 1) == args.epochs:
            print("\nRunning evaluation...")
            miou, class_ious = evaluate(model, val_loader, device, 3, args.img_size)
            
            print(f"  mIoU: {miou*100:.2f}%")
            for c, name in enumerate(SparkDataset.CLASSES):
                print(f"    {name}: {class_ious[c]*100:.2f}%")
            
            # Save best model
            if miou > best_miou:
                best_miou = miou
                save_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'miou': miou,
                    'best_miou': best_miou,
                    'class_ious': class_ious
                }
                torch.save(save_dict, output_dir / 'best_model.pth')
                print(f"  ★ New best model saved! mIoU: {miou*100:.2f}%")
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
            }
            ckpt_path = output_dir / f'checkpoint_epoch{epoch+1}.pth'
            torch.save(save_dict, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path.name}")
        
        print()  # Empty line between epochs
    
    # Final summary
    print("=" * 60)
    print(" Training Complete!")
    print("=" * 60)
    print(f"Best mIoU: {best_miou*100:.2f}%")
    print(f"Models saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='SPARK 2024 SegFormer Training')
    
    # Data
    parser.add_argument('--data-root', default='/scratch/users/hchaurasia/spark2024/data')
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=8)
    
    # Model
    parser.add_argument('--model', default='nvidia/segformer-b5-finetuned-ade-640-640')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=2e-5)
    
    # Checkpointing
    parser.add_argument('--eval-interval', type=int, default=5)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    # Output
    parser.add_argument('--output-dir', default='/scratch/users/hchaurasia/spark2024/outputs')
    
    # Debug
    parser.add_argument('--max-samples', type=int, default=None)
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
