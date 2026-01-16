#!/usr/bin/env python3
"""
SPARK 2024 - Advanced Inference for Maximum IoU
Combines: TTA + Snapshot Ensemble + Multi-Scale + Post-Processing
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
import cv2


class AdvancedTTA:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
    def _predict_single(self, image):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = self.model(pixel_values=image)
            if hasattr(out, 'logits'):
                out = out.logits
        return out
    
    @torch.no_grad()
    def predict(self, image, original_size, use_multiscale=True):
        self.model.eval()
        scales = [1.0] if not use_multiscale else [0.75, 1.0, 1.25]
        accumulated = torch.zeros(1, 3, original_size[0], original_size[1], device=self.device)
        count = 0
        
        for scale in scales:
            if scale != 1.0:
                h, w = int(image.shape[2] * scale), int(image.shape[3] * scale)
                scaled = F.interpolate(image, (h, w), mode='bilinear', align_corners=False)
            else:
                scaled = image
            
            for flip_h in [False, True]:
                for flip_v in [False, True]:
                    aug = scaled
                    if flip_h:
                        aug = torch.flip(aug, [-1])
                    if flip_v:
                        aug = torch.flip(aug, [-2])
                    
                    pred = self._predict_single(aug.to(self.device))
                    
                    if flip_v:
                        pred = torch.flip(pred, [-2])
                    if flip_h:
                        pred = torch.flip(pred, [-1])
                    
                    pred = F.interpolate(pred, original_size, mode='bilinear', align_corners=False)
                    accumulated += F.softmax(pred, dim=1)
                    count += 1
        
        return accumulated / count


class SnapshotEnsemble:
    def __init__(self, checkpoint_paths, device='cuda'):
        self.device = device
        self.models = []
        self.tta_predictors = []
        
        print(f"Loading {len(checkpoint_paths)} checkpoints...")
        for path in checkpoint_paths:
            if not os.path.exists(path):
                print(f"  WARNING: {path} not found")
                continue
                
            model = SegformerForSemanticSegmentation.from_pretrained(
                'nvidia/segformer-b5-finetuned-ade-640-640',
                num_labels=3, ignore_mismatched_sizes=True
            )
            
            ckpt = torch.load(path, map_location=device)
            state_dict = ckpt.get('model_state_dict', ckpt)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model.to(device).eval()
            
            self.models.append(model)
            self.tta_predictors.append(AdvancedTTA(model, device))
            print(f"  ✓ {os.path.basename(path)}")
    
    @torch.no_grad()
    def predict(self, image, original_size, use_tta=True, use_multiscale=False):
        accumulated = torch.zeros(1, 3, original_size[0], original_size[1], device=self.device)
        
        for model, tta in zip(self.models, self.tta_predictors):
            if use_tta:
                pred = tta.predict(image, original_size, use_multiscale)
            else:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = model(pixel_values=image.to(self.device))
                    if hasattr(out, 'logits'):
                        out = out.logits
                pred = F.interpolate(out, original_size, mode='bilinear', align_corners=False)
                pred = F.softmax(pred, dim=1)
            accumulated += pred
        
        return accumulated / len(self.models)


def refine_boundaries(prediction, kernel_size=3, min_area=50):
    refined = prediction.copy()
    for class_id in [1, 2]:
        mask = (prediction == class_id).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                cv2.drawContours(mask, [cnt], -1, 0, -1)
        refined[mask == 1] = class_id
    return refined


def labels_to_rgb(labels):
    H, W = labels.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[labels == 1] = [255, 0, 0]
    rgb[labels == 2] = [0, 0, 255]
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', type=str, required=True)
    parser.add_argument('--test-dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='predictions_advanced')
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--no-tta', action='store_true')
    parser.add_argument('--multiscale', action='store_true')
    parser.add_argument('--no-refine', action='store_true')
    args = parser.parse_args()
    
    device = torch.device('cuda')
    checkpoint_paths = [p.strip() for p in args.checkpoints.split(',')]
    ensemble = SnapshotEnsemble(checkpoint_paths, device)
    
    test_files = sorted([f for f in os.listdir(args.test_dir) if f.endswith('.jpg')])
    print(f"Found {len(test_files)} test images")
    os.makedirs(args.output, exist_ok=True)
    
    use_tta = not args.no_tta
    use_refine = not args.no_refine
    print(f"TTA={use_tta}, MultiScale={args.multiscale}, Refine={use_refine}")
    
    for f in tqdm(test_files, desc="Inference"):
        image = np.array(Image.open(os.path.join(args.test_dir, f)).convert('RGB'))
        orig_h, orig_w = image.shape[:2]
        
        image_resized = np.array(Image.fromarray(image).resize((args.img_size, args.img_size), Image.BILINEAR))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)
        
        probs = ensemble.predict(image_tensor, (orig_h, orig_w), use_tta, args.multiscale)
        pred = probs.argmax(dim=1).squeeze().cpu().numpy()
        
        if use_refine:
            pred = refine_boundaries(pred)
        
        rgb = labels_to_rgb(pred)
        out_name = f.replace('.jpg', '_pred.png').replace('_img', '')
        Image.fromarray(rgb).save(os.path.join(args.output, out_name))
    
    print(f"Done! Saved to {args.output}")


if __name__ == '__main__':
    main()
