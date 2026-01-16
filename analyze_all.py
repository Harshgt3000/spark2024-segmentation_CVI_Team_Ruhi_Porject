import os
import sys
import torch
import numpy as np
from PIL import Image
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F

def compute_iou(pred, gt, cls):
    intersection = ((pred == cls) & (gt == cls)).sum()
    union = ((pred == cls) | (gt == cls)).sum()
    return intersection / union if union > 0 else 1.0

def find_all_samples(data_dir, split='val'):
    samples = []
    images_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'mask')
    
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
                    samples.append((f"{spacecraft}/{img_file}", img_path, mask_path, spacecraft))
    
    return samples

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("Loading model...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640",
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    checkpoint = torch.load("outputs/segformer_b5_20251220_174956/best_model.pth", map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded!")
    
    val_samples = find_all_samples("data", split='val')
    print(f"Found {len(val_samples)} total validation samples")
    
    # Sample 50 from each spacecraft type
    spacecraft_samples = {}
    for sample in val_samples:
        sc = sample[3]
        if sc not in spacecraft_samples:
            spacecraft_samples[sc] = []
        spacecraft_samples[sc].append(sample)
    
    selected_samples = []
    for sc, samples in spacecraft_samples.items():
        selected_samples.extend(samples[:50])
    
    print(f"Analyzing {len(selected_samples)} samples (50 per spacecraft)...")
    
    results = []
    for i, (name, img_path, mask_path, spacecraft) in enumerate(selected_samples):
        img = np.array(Image.open(img_path).convert('RGB'))
        h, w = img.shape[:2]
        
        gt_mask = np.array(Image.open(mask_path).convert('RGB'))
        gt = np.zeros(gt_mask.shape[:2], dtype=np.int64)
        gt[(gt_mask[:,:,0] > 200) & (gt_mask[:,:,1] < 50) & (gt_mask[:,:,2] < 50)] = 1
        gt[(gt_mask[:,:,2] > 200) & (gt_mask[:,:,0] < 50) & (gt_mask[:,:,1] < 50)] = 2
        
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(pixel_values=img_tensor)
            logits = F.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
            pred = logits.argmax(dim=1)[0].cpu().numpy()
        
        iou_body = compute_iou(pred, gt, 1)
        iou_solar = compute_iou(pred, gt, 2)
        
        has_body = (gt == 1).sum() > 0
        has_solar = (gt == 2).sum() > 0
        
        if has_body and has_solar:
            miou = (iou_body + iou_solar) / 2
        elif has_body:
            miou = iou_body
        elif has_solar:
            miou = iou_solar
        else:
            miou = 1.0
        
        results.append({
            'name': name,
            'spacecraft': spacecraft,
            'miou': miou,
            'body_iou': iou_body if has_body else None,
            'solar_iou': iou_solar if has_solar else None,
        })
        
        if (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{len(selected_samples)}")
            sys.stdout.flush()
    
    print("\n" + "=" * 80)
    print(" PERFORMANCE BY SPACECRAFT TYPE (Sorted Worst to Best)")
    print("=" * 80)
    
    spacecraft_results = {}
    for r in results:
        sc = r['spacecraft']
        if sc not in spacecraft_results:
            spacecraft_results[sc] = {'miou': [], 'body': [], 'solar': []}
        spacecraft_results[sc]['miou'].append(r['miou'])
        if r['body_iou'] is not None:
            spacecraft_results[sc]['body'].append(r['body_iou'])
        if r['solar_iou'] is not None:
            spacecraft_results[sc]['solar'].append(r['solar_iou'])
    
    print(f"{'Spacecraft':<20} {'mIoU':>10} {'Body':>10} {'Solar':>10} {'Count':>8}")
    print("-" * 60)
    
    for sc, data in sorted(spacecraft_results.items(), key=lambda x: np.mean(x[1]['miou'])):
        miou = np.mean(data['miou']) * 100
        body = np.mean(data['body']) * 100 if data['body'] else 0
        solar = np.mean(data['solar']) * 100 if data['solar'] else 0
        print(f"{sc:<20} {miou:>9.2f}% {body:>9.2f}% {solar:>9.2f}% {len(data['miou']):>8}")
    
    print("\n" + "=" * 80)
    print(" OVERALL STATISTICS")
    print("=" * 80)
    all_miou = [r['miou'] for r in results]
    all_body = [r['body_iou'] for r in results if r['body_iou'] is not None]
    all_solar = [r['solar_iou'] for r in results if r['solar_iou'] is not None]
    
    print(f"Overall mIoU: {np.mean(all_miou)*100:.2f}%")
    print(f"Overall Body IoU: {np.mean(all_body)*100:.2f}%")
    print(f"Overall Solar IoU: {np.mean(all_solar)*100:.2f}%")

if __name__ == "__main__":
    main()
