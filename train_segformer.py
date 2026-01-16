#!/usr/bin/env python3
"""
SPARK 2024 - SegFormer Training Script (Simpler Alternative)
=============================================================
SegFormer is simpler and more reliable than Mask2Former for semantic segmentation.
Use this if Mask2Former has issues.

Usage:
    python train_segformer.py
    python train_segformer.py --model nvidia/segformer-b5-finetuned-ade-640-640
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
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from tqdm import tqdm
import numpy as np

from spark_dataset import SparkDataset, get_train_transforms, get_val_transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Train SegFormer for SPARK 2024')
    
    # Data
    parser.add_argument('--data-root', type=str, 
                        default='/scratch/users/hchaurasia/spark2024/data')
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Model - SegFormer options:
    # nvidia/segformer-b0-finetuned-ade-512-512 (3.7M params)
    # nvidia/segformer-b1-finetuned-ade-512-512 (13.7M params)
    # nvidia/segformer-b2-finetuned-ade-512-512 (24.7M params)
    # nvidia/segformer-b3-finetuned-ade-512-512 (44.6M params)
    # nvidia/segformer-b4-finetuned-ade-512-512 (61.4M params)
    # nvidia/segformer-b5-finetuned-ade-640-640 (81.4M params) <- Best
    parser.add_argument('--model', type=str, 
                        default='nvidia/segformer-b5-finetuned-ade-640-640')
    parser.add_argument('--num-classes', type=int, default=3)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--grad-accum', type=int, default=2)
    parser.add_argument('--amp', action='store_true', default=True)
    
    # Output
    parser.add_argument('--output-dir', type=str, 
                        default='/scratch/users/hchaurasia/spark2024/outputs')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--eval-interval', type=int, default=5)
    
    # Debug
    parser.add_argument('--max-samples', type=int, default=None)
    
    return parser.parse_args()


def compute_iou(pred, target, num_classes):
    """Compute IoU per class."""
    ious = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = (intersection / union).item()
        
        ious.append(iou)
    
    return ious


def evaluate(model, dataloader, device, num_classes, img_size):
    """Evaluate model."""
    model.eval()
    
    all_ious = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            
            # Upsample to original size
            logits = F.interpolate(
                logits,
                size=(img_size, img_size),
                mode='bilinear',
                align_corners=False
            )
            
            preds = logits.argmax(dim=1)
            
            # Compute IoU
            for pred, label in zip(preds, labels):
                ious = compute_iou(pred.cpu(), label.cpu(), num_classes)
                for cls, iou in enumerate(ious):
                    all_ious[cls].append(iou)
    
    class_ious = {cls: np.mean(ious) if ious else 0.0 for cls, ious in all_ious.items()}
    mean_iou = np.mean(list(class_ious.values()))
    
    return mean_iou, class_ious


def train(args):
    """Main training function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'segformer_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Datasets
    print("\nLoading datasets...")
    train_dataset = SparkDataset(
        data_root=args.data_root,
        split='train',
        transform=get_train_transforms((args.img_size, args.img_size)),
        max_samples_per_class=args.max_samples
    )
    
    val_dataset = SparkDataset(
        data_root=args.data_root,
        split='val',
        transform=get_val_transforms((args.img_size, args.img_size)),
        max_samples_per_class=args.max_samples
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model
    print(f"\nLoading model: {args.model}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model,
        num_labels=args.num_classes,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    warmup_iters = len(train_loader) * args.warmup_epochs
    total_iters = len(train_loader) * args.epochs
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], [warmup_iters])
    
    # Loss with class weights (solar panels are harder)
    class_weights = torch.tensor([1.0, 1.0, 2.0], device=device)  # More weight on panels
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # AMP
    scaler = GradScaler() if args.amp else None
    
    # Training
    print("\nStarting training...")
    best_miou = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            if args.amp:
                with autocast():
                    outputs = model(pixel_values=pixel_values)
                    logits = outputs.logits
                    
                    # Upsample logits to label size
                    logits = F.interpolate(
                        logits,
                        size=labels.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    loss = criterion(logits, labels)
                    loss = loss / args.grad_accum
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % args.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                outputs = model(pixel_values=pixel_values)
                logits = F.interpolate(outputs.logits, size=labels.shape[-2:], 
                                       mode='bilinear', align_corners=False)
                loss = criterion(logits, labels) / args.grad_accum
                loss.backward()
                
                if (batch_idx + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            
            epoch_loss += loss.item() * args.grad_accum
            pbar.set_postfix({'loss': f'{loss.item() * args.grad_accum:.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
        
        # Evaluate
        if (epoch + 1) % args.eval_interval == 0:
            miou, class_ious = evaluate(model, val_loader, device, args.num_classes, args.img_size)
            
            print(f"mIoU: {miou*100:.2f}%")
            for cls, iou in class_ious.items():
                cls_name = SparkDataset.CLASSES[cls]
                print(f"  {cls_name}: {iou*100:.2f}%")
            
            if miou > best_miou:
                best_miou = miou
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'miou': miou,
                    'class_ious': class_ious,
                }, output_dir / 'best_model.pth')
                print(f"Best model saved! mIoU: {miou*100:.2f}%")
        
        # Checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
            }, output_dir / f'checkpoint_epoch{epoch+1}.pth')
    
    print(f"\nTraining complete! Best mIoU: {best_miou*100:.2f}%")
    print(f"Models saved to: {output_dir}")


if __name__ == '__main__':
    args = parse_args()
    train(args)
