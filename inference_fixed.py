#!/usr/bin/env python3
"""
Fixed inference - matches training preprocessing (no ImageNet normalization)
"""
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test-dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='predictions')
    args = parser.parse_args()
    
    device = torch.device('cuda')
    
    # Load model
    print("Loading model...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/segformer-b5-finetuned-ade-640-640',
        num_labels=3, ignore_mismatched_sizes=True
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"Checkpoint mIoU: {ckpt.get('best_miou', 'N/A')}")
    
    # Find test images
    test_files = sorted([f for f in os.listdir(args.test_dir) if f.endswith('.jpg')])
    print(f"Found {len(test_files)} test images")
    
    os.makedirs(args.output, exist_ok=True)
    
    with torch.inference_mode():
        for f in tqdm(test_files, desc="Inference"):
            # Load image
            img_path = os.path.join(args.test_dir, f)
            image = np.array(Image.open(img_path).convert('RGB'))
            orig_h, orig_w = image.shape[:2]
            
            # Resize to 512x512 (match training)
            image_resized = np.array(Image.fromarray(image).resize((512, 512), Image.BILINEAR))
            
            # MATCH TRAINING: just divide by 255, NO ImageNet normalization!
            image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            # Forward pass
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(pixel_values=image_tensor)
            
            # Upsample to original size
            logits = F.interpolate(outputs.logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            pred = logits.argmax(dim=1).squeeze().cpu().numpy()
            
            # Convert to RGB
            rgb = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            rgb[pred == 1] = [255, 0, 0]  # Body - Red
            rgb[pred == 2] = [0, 0, 255]  # Panels - Blue
            
            # Save
            out_name = f.replace('.jpg', '_pred.png').replace('_img', '')
            Image.fromarray(rgb).save(os.path.join(args.output, out_name))
    
    print(f"Done! Saved to {args.output}")

if __name__ == '__main__':
    main()
