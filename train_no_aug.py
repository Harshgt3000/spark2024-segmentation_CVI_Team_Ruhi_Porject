#!/usr/bin/env python3
"""
Training WITHOUT augmentation - for better test generalization
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
import gc

class SparkDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=512):
        self.img_size = img_size
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, msk_path = self.samples[idx]
        
        # Load and resize
        image = Image.open(img_path).convert('RGB').resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = Image.open(msk_path).convert('RGB').resize((self.img_size, self.img_size), Image.NEAREST)
        
        image = np.array(image)
        mask = np.array(mask)
        
        # Convert mask to labels
        label = np.zeros(mask.shape[:2], dtype=np.int64)
        label[(mask[:,:,0] > 200) & (mask[:,:,1] < 50) & (mask[:,:,2] < 50)] = 1
        label[(mask[:,:,2] > 200) & (mask[:,:,0] < 50) & (mask[:,:,1] < 50)] = 2
        
        # Normalize same as inference: /255.0 only
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(label).long()
        
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
        return 0.5 * ce_loss + 0.5 * (1 - dice.mean())


def evaluate(model, loader, device):
    model.eval()
    conf_matrix = torch.zeros(3, 3, dtype=torch.int64)
    
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Eval", leave=False):
            imgs = batch['image'].to(device)
            lbls = batch['mask']
            
            with torch.cuda.amp.autocast(dtype=torch.float16):
                if isinstance(model, nn.DataParallel):
                    out = model.module(pixel_values=imgs).logits
                else:
                    out = model(pixel_values=imgs).logits
            
            preds = F.interpolate(out, size=lbls.shape[-2:], mode='bilinear', align_corners=False).argmax(1).cpu()
            
            for p, l in zip(preds.view(-1), lbls.view(-1)):
                if 0 <= l < 3:
                    conf_matrix[l, p] += 1
            
            del imgs, out, preds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    intersection = torch.diag(conf_matrix).float()
    union = conf_matrix.sum(1) + conf_matrix.sum(0) - intersection
    iou = intersection / (union + 1e-6)
    return iou.mean().item() * 100, {n: iou[i].item()*100 for i, n in enumerate(['bg', 'body', 'solar'])}


def train(args):
    device = torch.device('cuda')
    print(f"GPUs: {torch.cuda.device_count()}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}/segformer_noaug_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    train_ds = SparkDataset(args.data_dir, 'train', args.img_size)
    val_ds = SparkDataset(args.data_dir, 'val', args.img_size)
    
    batch_size = args.batch_size * torch.cuda.device_count()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/segformer-b5-finetuned-ade-640-640', num_labels=3, ignore_mismatched_sizes=True
    )
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    
    warmup = 5
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (args.epochs - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_miou = 0.0
    
    print(f"\n{'='*50}")
    print(f" NO AUGMENTATION Training: {args.epochs} epochs")
    print(f"{'='*50}\n")
    
    for epoch in range(args.epochs):
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
        print(f"[Epoch {epoch+1}] Loss: {total_loss/len(train_loader):.4f}")
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            
            miou, ious = evaluate(model, val_loader, device)
            print(f"  mIoU: {miou:.2f}% | body: {ious['body']:.1f}% | solar: {ious['solar']:.1f}%")
            
            if miou > best_miou:
                best_miou = miou
                state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save({"epoch": epoch+1, "model_state_dict": state, "best_miou": best_miou}, 
                          f"{output_dir}/best_model.pth")
                print(f"  ★ New best: {miou:.2f}%")
            
            # Save checkpoint
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({"epoch": epoch+1, "model_state_dict": state, "best_miou": best_miou}, 
                      f"{output_dir}/checkpoint_epoch{epoch+1}.pth")
            
            gc.collect()
            torch.cuda.empty_cache()
    
    print(f"\nDone! Best mIoU: {best_miou:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--img-size', type=int, default=512)
    args = parser.parse_args()
    train(args)
