#!/usr/bin/env python3
"""
Multi-Node Distributed Training for SegFormer-B5
"""
import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from spark_dataset import SparkDataset

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def compute_miou(preds, labels, num_classes=3):
    ious = []
    for c in range(num_classes):
        pred_c = (preds == c)
        label_c = (labels == c)
        intersection = (pred_c & label_c).sum().item()
        union = (pred_c | label_c).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) * 100, ious

def train(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    setup(rank, world_size)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"World size: {world_size}")
        print(f"Training with {world_size} GPUs")
    
    # Load datasets
    train_dataset = SparkDataset(
        root_dir=args.data_dir,
        split='train',
        img_size=args.img_size
    )
    val_dataset = SparkDataset(
        root_dir=args.data_dir,
        split='val',
        img_size=args.img_size
    )
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=4, pin_memory=True)
    
    if rank == 0:
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        print(f"Batch size per GPU: {args.batch_size}, Effective batch: {args.batch_size * world_size}")
    
    # Load model
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model,
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
    
    # Resume if checkpoint exists
    start_epoch = 0
    best_miou = 0.0
    
    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.module.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()})
        start_epoch = checkpoint.get("epoch", 0)
        best_miou = checkpoint.get("best_miou", 0.0)
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}, best mIoU: {best_miou:.2f}%")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        total_loss = 0.0
        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            pbar = train_loader
        
        for batch in pbar:
            images = batch["image"].to(device)
            labels = batch["mask"].to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values=images, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if rank == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        
        if rank == 0:
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Evaluation
        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            all_preds, all_labels = [], []
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    labels = batch["mask"].to(device)
                    
                    outputs = model(pixel_values=images)
                    logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear")
                    preds = logits.argmax(dim=1)
                    
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())
            
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)
            
            miou, class_ious = compute_miou(all_preds.numpy(), all_labels.numpy())
            
            if rank == 0:
                print(f"Running evaluation...")
                print(f"  mIoU: {miou:.2f}%")
                print(f"    background: {class_ious[0]*100:.2f}%")
                print(f"    spacecraft_body: {class_ious[1]*100:.2f}%")
                print(f"    solar_panel: {class_ious[2]*100:.2f}%")
                
                if miou > best_miou:
                    best_miou = miou
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": model.module.state_dict(),
                        "best_miou": best_miou
                    }, os.path.join(args.output_dir, "best_model.pth"))
                    print(f"  ★ New best model saved! mIoU: {miou:.2f}%")
        
        # Checkpoint
        if rank == 0 and (epoch + 1) % args.checkpoint_interval == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_miou": best_miou
            }, os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pth"))
            print(f"  Checkpoint saved: checkpoint_epoch{epoch+1}.pth")
    
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/segformer-b5-finetuned-ade-640-640")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/segformer_b5_20251220_174956")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()
