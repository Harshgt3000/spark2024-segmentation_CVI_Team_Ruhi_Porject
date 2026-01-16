#!/usr/bin/env python3
"""Generate sample prediction visualizations for report"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration - UPDATE THESE PATHS
DATA_DIR = "/scratch/users/hchaurasia/spark2024/data"
CHECKPOINT = "/scratch/users/hchaurasia/spark2024/outputs/segformer_pseudo_20260109_123826/best_model.pth"
OUTPUT_DIR = "/scratch/users/hchaurasia/spark2024/report_figures"

# If checkpoint doesn't exist, find the best one
if not os.path.exists(CHECKPOINT):
    import glob
    checkpoints = glob.glob("/scratch/users/hchaurasia/spark2024/outputs/*/best_model.pth")
    if checkpoints:
        CHECKPOINT = sorted(checkpoints)[-1]
        print(f"Using checkpoint: {CHECKPOINT}")
    else:
        print("No checkpoint found!")
        sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette
palette = np.array([
    [0, 0, 0],       # Background - black
    [255, 0, 0],     # Spacecraft body - red  
    [0, 0, 255],     # Solar panel - blue
], dtype=np.uint8)

# Load model
print("Loading model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640",
    num_labels=3,
    ignore_mismatched_sizes=True
)

checkpoint = torch.load(CHECKPOINT, map_location=device)
state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()
print("Model loaded!")

# Find sample images (one per spacecraft)
spacecraft_list = ['Cheops', 'VenusExpress', 'Proba2', 'Smart1', 'XMM-Newton', 'Soho']
samples = []

images_dir = os.path.join(DATA_DIR, 'images')
for spacecraft in spacecraft_list:
    val_dir = os.path.join(images_dir, spacecraft, 'val')
    if os.path.exists(val_dir):
        imgs = [f for f in os.listdir(val_dir) if f.endswith('_img.jpg')]
        if imgs:
            samples.append((spacecraft, os.path.join(val_dir, imgs[0])))

print(f"Found {len(samples)} sample images")

# Process each sample
for idx, (spacecraft, img_path) in enumerate(samples):
    print(f"\nProcessing: {spacecraft}")
    
    # Load image
    image = Image.open(img_path).convert('RGB')
    img_array = np.array(image)
    
    # Load ground truth
    mask_path = img_path.replace('/images/', '/mask/').replace('_img.jpg', '_layer.jpg')
    if os.path.exists(mask_path):
        gt_mask = np.array(Image.open(mask_path))
        gt_classes = np.zeros(gt_mask.shape[:2], dtype=np.uint8)
        gt_classes[(gt_mask[:,:,0] > 200) & (gt_mask[:,:,2] < 50)] = 1
        gt_classes[(gt_mask[:,:,2] > 200) & (gt_mask[:,:,0] < 50)] = 2
    else:
        gt_classes = None
    
    # Inference
    with torch.no_grad():
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        outputs = model(img_tensor)
        logits = outputs.logits
        logits = F.interpolate(logits, size=(1024, 1024), mode='bilinear', align_corners=False)
        pred = logits.argmax(dim=1).squeeze().cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_array)
    axes[0].set_title('Input Image', fontsize=12)
    axes[0].axis('off')
    
    if gt_classes is not None:
        gt_colored = palette[gt_classes]
        axes[1].imshow(gt_colored)
        axes[1].set_title('Ground Truth', fontsize=12)
        axes[1].axis('off')
    
    pred_colored = palette[pred]
    axes[2].imshow(pred_colored)
    axes[2].set_title('Prediction', fontsize=12)
    axes[2].axis('off')
    
    plt.suptitle(f'{spacecraft}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, f'qualitative_{spacecraft}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

# Create combined figure
print("\nCreating combined figure...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (spacecraft, _) in enumerate(samples[:6]):
    img_path = os.path.join(OUTPUT_DIR, f'qualitative_{spacecraft}.png')
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        axes[idx].imshow(img)
        axes[idx].axis('off')

plt.tight_layout()
combined_path = os.path.join(OUTPUT_DIR, 'qualitative_combined.png')
plt.savefig(combined_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Combined figure saved: {combined_path}")

print("\n✓ Done! Check", OUTPUT_DIR)
