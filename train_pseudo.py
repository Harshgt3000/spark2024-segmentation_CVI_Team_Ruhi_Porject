#!/usr/bin/env python3
"""Pseudo-labeling: Train on test predictions"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
        image = Image.open(img_path).convert('RGB').resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = Image.open(msk_path).convert('RGB').resize((self.img_size, self.img_size), Image.NEAREST)
        image = np.array(image)
        mask = np.array(mask)
        label = np.zeros(mask.shape[:2], dtype=np.int64)
        label[(mask[:,:,0] > 200) & (mask[:,:,1] < 50) & (mask[:,:,2] < 50)] = 1
        label[(mask[:,:,2] > 200) & (mask[:,:,0] < 50) & (mask[:,:,1] < 50)] = 2
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(label).long()
        return {'image': image, 'mask': label}

class PseudoDataset(Dataset):
    def __init__(self, test_dir, pred_dir, img_size=512):
        self.img_size = img_size
        self.samples = []
        test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])
        for f in test_files:
            img_path = os.path.join(test_dir, f)
            pred_name = f.replace('_img.jpg', '_pred.png').replace('.jpg', '_pred.png')
            pred_path = os.path.join(pred_dir, pred_name)
            if os.path.exists(pred_path):
                self.samples.append((img_path, pred_path))
        print(f"[PSEUDO] {len(self.samples)} test samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, pred_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB').resize((self.img_size, self.img_size), Image.BILINEAR)
        pred = Image.open(pred_path).convert('RGB').resize((self.img_size, self.img_size), Image.NEAREST)
        image = np.array(image)
        pred = np.array(pred)
        label = np.zeros(pred.shape[:2], dtype=np.int64)
        label[(pred[:,:,0] > 200) & (pred[:,:,1] < 50) & (pred[:,:,2] < 50)] = 1
        label[(pred[:,:,2] > 200) & (pred[:,:,0] < 50) & (pred[:,:,1] < 50)] = 2
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(label).long()
        return {'image': image, 'mask': label}

def evaluate(model, loader, device):
    model.eval()
    conf_matrix = torch.zeros(3, 3, dtype=torch.int64)
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Eval", leave=False):
            imgs = batch['image'].to(device)
            lbls = batch['mask']
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(pixel_values=imgs).logits
            preds = F.interpolate(out, lbls.shape[-2:], mode='bilinear', align_corners=False).argmax(1).cpu()
            for c in range(3):
                for c2 in range(3):
                    conf_matrix[c, c2] += ((lbls == c) & (preds == c2)).sum()
            del imgs, out, preds
    torch.cuda.empty_cache()
    intersection = torch.diag(conf_matrix).float()
    union = conf_matrix.sum(1) + conf_matrix.sum(0) - intersection
    iou = intersection / (union + 1e-6)
    return iou.mean().item() * 100

def main(args):
    device = torch.device('cuda')
    print("=" * 60)
    print(" PSEUDO-LABELING TRAINING")
    print("=" * 60)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"outputs/segformer_pseudo_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    train_ds = SparkDataset(args.data_dir, 'train', args.img_size)
    val_ds = SparkDataset(args.data_dir, 'val', args.img_size)
    pseudo_ds = PseudoDataset(args.test_dir, args.pred_dir, args.img_size)
    combined_ds = ConcatDataset([train_ds, pseudo_ds])
    
    train_loader = DataLoader(combined_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)
    
    print(f"Combined: {len(train_ds)} train + {len(pseudo_ds)} pseudo = {len(combined_ds)}")
    
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b5-finetuned-ade-640-640', num_labels=3, ignore_mismatched_sizes=True)
    print(f"Loading: {args.resume}")
    ckpt = torch.load(args.resume, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in ckpt['model_state_dict'].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    
    best_miou = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            imgs = batch['image'].to(device)
            lbls = batch['mask'].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(pixel_values=imgs).logits
                out = F.interpolate(out, lbls.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(out, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"[Epoch {epoch+1}] Loss: {total_loss/len(train_loader):.4f}")
        
        if (epoch + 1) % 2 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            miou = evaluate(model, val_loader, device)
            print(f"  mIoU: {miou:.2f}%")
            if miou > best_miou:
                best_miou = miou
                torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(), "best_miou": best_miou}, f"{output_dir}/best_model.pth")
                print(f"  ★ New best: {miou:.2f}%")
    
    print(f"\nDone! Best mIoU: {best_miou:.2f}%")
    print(f"Output: {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--test-dir', default='data/stream-1-test')
    parser.add_argument('--pred-dir', default='predictions_tta_97')
    parser.add_argument('--resume', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--img-size', type=int, default=512)
    args = parser.parse_args()
    main(args)
