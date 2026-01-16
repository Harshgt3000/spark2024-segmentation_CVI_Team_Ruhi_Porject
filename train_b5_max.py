#!/usr/bin/env python3
"""
SegFormer-B5 with MAXIMUM resources + all optimizations
- 4x V100-32GB GPUs
- FP16 training + inference
- Streaming mIoU
- Frequent checkpoints
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation
import numpy as np
from PIL import Image
from datetime import datetime
from tqdm import tqdm
import albumentations as A
import gc

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.6'

class SparkDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=512, augment=True):
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        self.samples = []
        
        images_dir = os.path.join(root_dir, 'images')
        mask_dir = os.path.join(root_dir, 'mask')
        
        for spacecraft in os.listdir(images_dir):
            img_dir = os.path.join(images_dir, spacecraft, split)
            msk_dir = os.path.join(mask_dir, spacecraft, split)
            if not os.path.isdir(img_dir):
                continue
            for f in os.listdir(img_dir):
                if f.endswith('_img.jpg'):
                    img_path = os.path.join(img_dir, f)
                    msk_path = os.path.join(msk_dir, f.replace('_img.jpg', '_layer.jpg'))
                    if os.path.exists(msk_path):
                        self.samples.append((img_path, msk_path))
        
        print(f"[{split.upper()}] {len(self.samples)} samples")
        
        if self.augment:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, border_mode=0, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.Resize(img_size, img_size),
            ])
        else:
            self.transform = A.Compose([A.Resize(img_size, img_size)])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, msk_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(msk_path).convert('RGB'))
        
        label = np.zeros(mask.shape[:2], dtype=np.int64)
        label[(mask[:,:,0] > 200) & (mask[:,:,1] < 50) & (mask[:,:,2] < 50)] = 1
        label[(mask[:,:,2] > 200) & (mask[:,:,0] < 50) & (mask[:,:,1] < 50)] = 2
        
        t = self.transform(image=image, mask=label)
        image = torch.from_numpy(t['image']).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(t['mask']).long()
        return {'image': image, 'mask': label}


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        pred_soft = F.softmax(pred, dim=1)
        target_oh = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
        inter = (pred_soft * target_oh).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
        dice = (2 * inter + 1) / (union + 1)
        dice_loss = 1 - dice.mean()
        return 0.5 * ce_loss + 0.5 * dice_loss


def evaluate(model, loader, device):
    """Streaming mIoU with FP16 - memory safe"""
    model.eval()
    conf_matrix = torch.zeros(3, 3, dtype=torch.int64)
    
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(loader, desc="Eval", leave=False)):
            imgs = batch['image'].to(device, non_blocking=True)
            lbls = batch['mask']
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                if isinstance(model, nn.DataParallel):
                    out = model.module(pixel_values=imgs).logits
                else:
                    out = model(pixel_values=imgs).logits
            
            preds = F.interpolate(out, size=lbls.shape[-2:], mode='bilinear', align_corners=False).argmax(1).cpu()
            
            # Update confusion matrix
            for p, l in zip(preds.view(-1), lbls.view(-1)):
                if 0 <= l < 3:
                    conf_matrix[l, p] += 1
            
            del imgs, out, preds
            
            # Clear cache periodically
            if i % 200 == 0:
                torch.cuda.empty_cache()
    
    # Compute IoU
    intersection = torch.diag(conf_matrix).float()
    union = conf_matrix.sum(1) + conf_matrix.sum(0) - intersection
    iou = intersection / (union + 1e-6)
    
    class_names = ['background', 'body', 'solar']
    class_ious = {n: iou[i].item() * 100 for i, n in enumerate(class_names)}
    return iou.mean().item() * 100, class_ious


def train(args):
    device = torch.device('cuda')
    num_gpus = torch.cuda.device_count()
    print(f"=" * 60)
    print(f" SegFormer-B5 MAX RESOURCES")
    print(f"=" * 60)
    print(f"GPUs: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
    print(f"=" * 60)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}/segformer_b5max_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")
    
    # Data
    train_ds = SparkDataset(args.data_dir, 'train', args.img_size, augment=True)
    val_ds = SparkDataset(args.data_dir, 'val', args.img_size, augment=False)
    
    # Larger batch with 4 GPUs
    batch_size = args.batch_size * num_gpus
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train: {len(train_loader)} batches (batch={batch_size})")
    print(f"Val: {len(val_loader)} batches (batch=1)")
    
    # Model - B5
    print(f"\nLoading: {args.model}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model, num_labels=3, ignore_mismatched_sizes=True
    )
    
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.1f}M")
    
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    
    # Warmup + cosine
    warmup = 5
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (args.epochs - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    start_epoch = 0
    best_miou = 0.0
    
    # Resume
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)
        start_epoch = ckpt.get("epoch", 0)
        best_miou = ckpt.get("best_miou", 0.0)
        print(f"Resumed epoch {start_epoch}, best mIoU: {best_miou:.2f}%")
    
    print(f"\n{'='*60}")
    print(f" Training: Epochs {start_epoch+1} -> {args.epochs}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            imgs = batch['image'].to(device)
            lbls = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                if isinstance(model, nn.DataParallel):
                    out = model.module(pixel_values=imgs).logits
                else:
                    out = model(pixel_values=imgs).logits
                out = F.interpolate(out, size=lbls.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(out, lbls)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Evaluate every 5 epochs
        if (epoch + 1) % args.eval_interval == 0:
            gc.collect()
            torch.cuda.empty_cache()
            
            miou, class_ious = evaluate(model, val_loader, device)
            print(f"  mIoU: {miou:.2f}% | body: {class_ious['body']:.1f}% | solar: {class_ious['solar']:.1f}%")
            
            # Save best
            if miou > best_miou:
                best_miou = miou
                state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save({"epoch": epoch+1, "model_state_dict": state, "best_miou": best_miou}, 
                          f"{output_dir}/best_model.pth")
                print(f"  ★ New best: {miou:.2f}%")
            
            # Always checkpoint at eval
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({"epoch": epoch+1, "model_state_dict": state, "optimizer_state_dict": optimizer.state_dict(), 
                       "best_miou": best_miou}, f"{output_dir}/checkpoint_epoch{epoch+1}.pth")
            print(f"  Checkpoint saved: epoch {epoch+1}")
            
            gc.collect()
            torch.cuda.empty_cache()
    
    print(f"\nDone! Best mIoU: {best_miou:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='nvidia/segformer-b5-finetuned-ade-640-640')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--eval-interval', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    train(args)
