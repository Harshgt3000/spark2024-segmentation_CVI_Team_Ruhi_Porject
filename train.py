#!/usr/bin/env python3
"""
SPARK 2024 - HuggingFace Mask2Former Training Script
=====================================================
Train Mask2Former with Swin-Large backbone for spacecraft segmentation.

Usage:
    # Single GPU
    python train.py
    
    # Multi-GPU with accelerate
    accelerate launch --num_processes=4 train.py
    
    # With custom config
    python train.py --epochs 100 --batch-size 4 --lr 1e-4
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

from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
    Mask2FormerConfig,
)

from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix

# Local imports
from spark_dataset import SparkDataset, get_train_transforms, get_val_transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Train Mask2Former for SPARK 2024')
    
    # Data
    parser.add_argument('--data-root', type=str, 
                        default='/scratch/users/hchaurasia/spark2024/data')
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Model
    parser.add_argument('--model', type=str, 
                        default='facebook/mask2former-swin-large-ade-semantic',
                        help='Pretrained model name')
    parser.add_argument('--num-classes', type=int, default=3)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--amp', action='store_true', default=True)
    
    # Output
    parser.add_argument('--output-dir', type=str, 
                        default='/scratch/users/hchaurasia/spark2024/outputs')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--eval-interval', type=int, default=5)
    
    # Debug
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit samples per class for debugging')
    
    return parser.parse_args()


class Mask2FormerSegmentation(nn.Module):
    """Wrapper for Mask2Former semantic segmentation."""
    
    def __init__(self, model_name: str, num_classes: int, img_size: int = 512):
        super().__init__()
        
        # Load pretrained model
        print(f"Loading pretrained model: {model_name}")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        
        # Image processor for post-processing
        self.processor = Mask2FormerImageProcessor.from_pretrained(model_name)
        self.processor.size = {"height": img_size, "width": img_size}
        
        self.num_classes = num_classes
        self.img_size = img_size
        
    def forward(self, pixel_values, labels=None):
        """Forward pass.
        
        Args:
            pixel_values: (B, 3, H, W) normalized images
            labels: (B, H, W) semantic labels
            
        Returns:
            loss (if training) or logits (if inference)
        """
        if labels is not None:
            # Training mode - need to convert labels to mask format
            # Mask2Former expects class_labels and mask_labels
            outputs = self.model(
                pixel_values=pixel_values,
                mask_labels=self._prepare_mask_labels(labels),
                class_labels=self._prepare_class_labels(labels),
            )
            return outputs.loss
        else:
            # Inference mode
            outputs = self.model(pixel_values=pixel_values)
            return outputs
    
    def _prepare_mask_labels(self, labels):
        """Convert semantic labels to instance masks for Mask2Former."""
        batch_mask_labels = []
        
        for label in labels:
            masks = []
            for class_id in range(self.num_classes):
                mask = (label == class_id).float()
                if mask.sum() > 0:  # Only include non-empty masks
                    masks.append(mask)
            
            if len(masks) == 0:
                # Add dummy background mask
                masks.append(torch.zeros_like(label).float())
            
            batch_mask_labels.append(torch.stack(masks))
        
        return batch_mask_labels
    
    def _prepare_class_labels(self, labels):
        """Prepare class labels for each mask."""
        batch_class_labels = []
        
        for label in labels:
            class_ids = []
            for class_id in range(self.num_classes):
                mask = (label == class_id)
                if mask.sum() > 0:
                    class_ids.append(class_id)
            
            if len(class_ids) == 0:
                class_ids.append(0)  # Background
            
            batch_class_labels.append(torch.tensor(class_ids, device=label.device))
        
        return batch_class_labels
    
    def predict(self, pixel_values):
        """Get semantic segmentation predictions."""
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            
        # Post-process to get semantic maps
        # Get class queries results
        masks_queries_logits = outputs.masks_queries_logits  # (B, num_queries, H, W)
        class_queries_logits = outputs.class_queries_logits  # (B, num_queries, num_classes+1)
        
        # Simple argmax approach for semantic segmentation
        B = pixel_values.shape[0]
        H, W = masks_queries_logits.shape[-2:]
        
        pred_masks = []
        for b in range(B):
            # Get mask logits and class logits for this sample
            mask_logits = masks_queries_logits[b]  # (num_queries, H, W)
            class_logits = class_queries_logits[b]  # (num_queries, num_classes+1)
            
            # Get predicted class for each query (excluding "no object" class)
            pred_classes = class_logits[:, :-1].argmax(dim=-1)  # (num_queries,)
            class_scores = class_logits[:, :-1].softmax(dim=-1).max(dim=-1)[0]  # (num_queries,)
            
            # Upsample masks to original size
            mask_logits = F.interpolate(
                mask_logits.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # (num_queries, H, W)
            
            # Create semantic map
            semantic_map = torch.zeros((self.img_size, self.img_size), 
                                       device=pixel_values.device, dtype=torch.long)
            
            # For each class, find the best query and assign
            for class_id in range(self.num_classes):
                # Find queries predicting this class
                class_mask = pred_classes == class_id
                if class_mask.sum() == 0:
                    continue
                
                # Get the query with highest confidence for this class
                class_query_scores = class_scores * class_mask.float()
                best_query = class_query_scores.argmax()
                
                # Get the mask for this query
                query_mask = mask_logits[best_query] > 0
                semantic_map[query_mask] = class_id
            
            pred_masks.append(semantic_map)
        
        return torch.stack(pred_masks)


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


def evaluate(model, dataloader, device, num_classes):
    """Evaluate model on validation set."""
    model.eval()
    
    all_ious = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Get predictions
            preds = model.predict(pixel_values)
            
            # Compute IoU for each sample
            for pred, label in zip(preds, labels):
                ious = compute_iou(pred.cpu(), label.cpu(), num_classes)
                for cls, iou in enumerate(ious):
                    all_ious[cls].append(iou)
    
    # Average IoU per class
    class_ious = {cls: np.mean(ious) if ious else 0.0 for cls, ious in all_ious.items()}
    mean_iou = np.mean(list(class_ious.values()))
    
    return mean_iou, class_ious


def train(args):
    """Main training function."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'mask2former_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create datasets
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
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = Mask2FormerSegmentation(
        model_name=args.model,
        num_classes=args.num_classes,
        img_size=args.img_size
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler with warmup
    warmup_iters = len(train_loader) * args.warmup_epochs
    total_iters = len(train_loader) * args.epochs
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_iters
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_iters - warmup_iters,
        eta_min=1e-6
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_iters]
    )
    
    # Mixed precision
    scaler = GradScaler() if args.amp else None
    
    # Training loop
    print("\nStarting training...")
    best_miou = 0.0
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass with AMP
            if args.amp:
                with autocast():
                    loss = model(pixel_values, labels)
                    loss = loss / args.grad_accum
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % args.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                loss = model(pixel_values, labels)
                loss = loss / args.grad_accum
                loss.backward()
                
                if (batch_idx + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            
            epoch_loss += loss.item() * args.grad_accum
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item() * args.grad_accum:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Evaluation
        if (epoch + 1) % args.eval_interval == 0:
            print("\nEvaluating...")
            miou, class_ious = evaluate(model, val_loader, device, args.num_classes)
            
            print(f"mIoU: {miou*100:.2f}%")
            for cls, iou in class_ious.items():
                cls_name = SparkDataset.CLASSES[cls]
                print(f"  {cls_name}: {iou*100:.2f}%")
            
            # Save best model
            if miou > best_miou:
                best_miou = miou
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'miou': miou,
                    'class_ious': class_ious,
                }, output_dir / 'best_model.pth')
                print(f"New best model saved! mIoU: {miou*100:.2f}%")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f'checkpoint_epoch{epoch+1}.pth')
    
    # Final evaluation
    print("\nFinal Evaluation...")
    miou, class_ious = evaluate(model, val_loader, device, args.num_classes)
    
    print(f"\nFinal Results:")
    print(f"  Best mIoU: {best_miou*100:.2f}%")
    print(f"  Final mIoU: {miou*100:.2f}%")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'miou': miou,
        'class_ious': class_ious,
    }, output_dir / 'final_model.pth')
    
    print(f"\nTraining complete! Models saved to {output_dir}")


if __name__ == '__main__':
    args = parse_args()
    train(args)
