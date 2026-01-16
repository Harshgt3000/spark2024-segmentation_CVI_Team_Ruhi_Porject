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
                # FIX: mask is _layer.jpg not _mask.png
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
    print(f"Found {len(val_samples)} validation samples")
    
    if len(val_samples) == 0:
        print("ERROR: No samples found!")
        return
    
    results = []
    total = min(500, len(val_samples))
    
    print(f"Analyzing {total} samples...")
    for i, (name, img_path, mask_path, spacecraft) in enumerate(val_samples[:total]):
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
            'img_path': img_path,
            'mask_path': mask_path
        })
        
        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{total}")
            sys.stdout.flush()
    
    results.sort(key=lambda x: x['miou'])
    
    print("\n" + "=" * 80)
    print(" WORST 20 PREDICTIONS")
    print("=" * 80)
    print(f"{'Name':<45} {'mIoU':>8} {'Body':>8} {'Solar':>8}")
    print("-" * 80)
    
    for r in results[:20]:
        body_str = f"{r['body_iou']*100:>7.2f}%" if r['body_iou'] is not None else "    N/A"
        solar_str = f"{r['solar_iou']*100:>7.2f}%" if r['solar_iou'] is not None else "    N/A"
        print(f"{r['name']:<45} {r['miou']*100:>7.2f}% {body_str} {solar_str}")
    
    print("\n" + "=" * 80)
    print(" STATISTICS")
    print("=" * 80)
    mious = [r['miou'] for r in results]
    body_ious = [r['body_iou'] for r in results if r['body_iou'] is not None]
    solar_ious = [r['solar_iou'] for r in results if r['solar_iou'] is not None]
    
    print(f"Average mIoU: {np.mean(mious)*100:.2f}%")
    print(f"Average Body IoU: {np.mean(body_ious)*100:.2f}%")
    print(f"Average Solar IoU: {np.mean(solar_ious)*100:.2f}%")
    print(f"\nWorst mIoU: {np.min(mious)*100:.2f}%")
    print(f"Best mIoU: {np.max(mious)*100:.2f}%")
    
    print("\n" + "=" * 80)
    print(" PERFORMANCE BY SPACECRAFT TYPE")
    print("=" * 80)
    
    spacecraft_results = {}
    for r in results:
        spacecraft = r['spacecraft']
        if spacecraft not in spacecraft_results:
            spacecraft_results[spacecraft] = []
        spacecraft_results[spacecraft].append(r['miou'])
    
    for spacecraft, mious_list in sorted(spacecraft_results.items(), key=lambda x: np.mean(x[1])):
        print(f"{spacecraft:<20} mIoU: {np.mean(mious_list)*100:.2f}% (n={len(mious_list)})")
    
    with open("worst_predictions.txt", "w") as f:
        f.write("name,miou,body_iou,solar_iou,img_path,mask_path\n")
        for r in results[:50]:
            body = r['body_iou'] if r['body_iou'] is not None else -1
            solar = r['solar_iou'] if r['solar_iou'] is not None else -1
            f.write(f"{r['name']},{r['miou']:.4f},{body:.4f},{solar:.4f},{r['img_path']},{r['mask_path']}\n")
    
    print("\nSaved worst 50 to worst_predictions.txt")

if __name__ == "__main__":
    main()
