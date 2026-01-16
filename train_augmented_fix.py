#!/usr/bin/env python3
import os
import sys
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

class SparkDatasetAug(Dataset):
    def __init__(self, root_dir, split='train', img_size=512, augment=True):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        
        self.samples = []
        images_dir = os.path.join(root_dir, 'images')
        mask_dir = os.path.join(root_dir, 'mask')
        
        for spacecraft in os.listdir(images_dir):
            spacecraft_img_dir = os.path.join(images_dir, spacecraft, split)
            spacecraft_mask_dir = os.path.join(mask_dir, spacecraft, split)
            
            if not os.path.isdir(spacecraft_img_dir):
                continue
            
            for img_file in os.listdir(spacecraft_img_dir):
                if img_file.endswith('_img.jpg'):
                    img_path = os.path.join(spacecraft_img_dir, img_file)
                    mask_file = img_file.replace('_img.jpg', '_layer.jpg')
                    mask_path = os.path.join(spacecraft_mask_dir, mask_file)
                    
                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))
        
        print(f"[{split.upper()}] Loaded {len(self.samples)} samples")
        
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
        img_path, mask_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('RGB'))
        label = np.zeros(mask.shape[:2], dtype=np.int64)
        label[(mask[:,:,0] > 200) & (mask[:,:,1] < 50) & (mask[:,:,2] < 50)] = 1
        label[(mask[:,:,2] > 200) & (mask[:,:,0] < 50) & (mask[:,:,1] < 50)] = 2
        transformed = self.transform(image=image, mask=label)
        image = transformed['image']
        label = transformed['mask']
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(label).long()
        return {'image': image, 'mask': label}

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target, num_classes=3):
        pred = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        intersection = (pred * target_onehot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        return 0.5 * self.ce(pred, target) + 0.5 * self.dice(pred, target)

def compute_miou_streaming(model, val_loader, device, num_classes=3):
    model.eval()
    class_intersection = torch.zeros(num_classes)
    class_union = torch.zeros(num_classes)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['mask'].to(device)
            
            if isinstance(model, nn.DataParallel):
                outputs = model.module(pixel_values=images)
            else:
                outputs = model(pixel_values=images)
            
            logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            preds = logits.argmax(dim=1)
            
            for c in range(num_classes):
                pred_c = (preds == c)
                label_c = (labels == c)
                class_intersection[c] += (pred_c & label_c).sum().cpu()
                class_union[c] += (pred_c | label_c).sum().cpu()
            
            del images, labels, outputs, logits, preds
            torch.cuda.empty_cache()
    
    ious = []
    class_names = ['background', 'spacecraft_body', 'solar_panel']
    class_ious = {}
    
    for c in range(num_classes):
        if class_union[c] > 0:
            iou = (class_intersection[c] / class_union[c]).item()
            ious.append(iou)
            class_ious[class_names[c]] = iou * 100
        else:
            class_ious[class_names[c]] = 100.0
    
    return np.mean(ious) * 100, class_ious

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"GPUs: {torch.cuda.device_count()}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'segformer_aug_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")
    
    train_dataset = SparkDatasetAug(args.data_dir, split='train', img_size=args.img_size, augment=True)
    val_dataset = SparkDatasetAug(args.data_dir, split='val', img_size=args.img_size, augment=False)
    
    batch_size = args.batch_size * max(1, torch.cuda.device_count())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    model = SegformerForSemanticSegmentation.from_pretrained(args.model, num_labels=3, ignore_mismatched_sizes=True)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    start_epoch = 0
    best_miou = 0.0
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        start_epoch = checkpoint.get("epoch", 0)
        best_miou = checkpoint.get("best_miou", 0.0)
        print(f"Resumed epoch {start_epoch}, best mIoU: {best_miou:.2f}%")
    
    print(f"\nTraining: Epochs {start_epoch + 1} -> {args.epochs}")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['mask'].to(device)
            
            optimizer.zero_grad()
            if isinstance(model, nn.DataParallel):
                outputs = model.module(pixel_values=images)
            else:
                outputs = model(pixel_values=images)
            
            logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        
        scheduler.step()
        print(f"[Epoch {epoch+1}] Loss: {total_loss/len(train_loader):.4f}")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        if (epoch + 1) % args.eval_interval == 0:
            print("Evaluating...")
            miou, class_ious = compute_miou_streaming(model, val_loader, device)
            print(f"  mIoU: {miou:.2f}%")
            for name, iou in class_ious.items():
                print(f"    {name}: {iou:.2f}%")
            
            if miou > best_miou:
                best_miou = miou
                torch.save({"epoch": epoch+1, "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), "best_miou": best_miou}, os.path.join(output_dir, "best_model.pth"))
                print(f"  ★ New best! {miou:.2f}%")
            
            gc.collect()
            torch.cuda.empty_cache()
        
        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save({"epoch": epoch+1, "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_miou": best_miou}, os.path.join(output_dir, f"checkpoint_epoch{epoch+1}.pth"))
            print(f"  Checkpoint saved: epoch {epoch+1}")
    
    print(f"\nDone! Best mIoU: {best_miou:.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='nvidia/segformer-b5-finetuned-ade-640-640')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--eval-interval', type=int, default=5)
    parser.add_argument('--checkpoint-interval', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
