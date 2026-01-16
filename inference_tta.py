import os
import torch
import numpy as np
from PIL import Image
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
import sys

def load_model(checkpoint_path, device):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640",
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def tta_predict(model, image, device):
    h, w = image.shape[:2]
    
    transforms = [
        lambda x: x,
        lambda x: np.fliplr(x),
        lambda x: np.flipud(x),
        lambda x: np.flipud(np.fliplr(x)),
    ]
    
    inverse_transforms = [
        lambda x: x,
        lambda x: np.fliplr(x),
        lambda x: np.flipud(x),
        lambda x: np.flipud(np.fliplr(x)),
    ]
    
    all_preds = []
    
    for transform, inv_transform in zip(transforms, inverse_transforms):
        img_t = transform(image.copy())
        img_tensor = torch.from_numpy(img_t.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(pixel_values=img_tensor)
            logits = F.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        
        probs_inv = np.stack([inv_transform(probs[c]) for c in range(3)])
        all_preds.append(probs_inv)
    
    avg_probs = np.mean(all_preds, axis=0)
    final_pred = np.argmax(avg_probs, axis=0)
    return final_pred

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    checkpoint = "outputs/segformer_b5_20251220_174956/best_model.pth"
    model = load_model(checkpoint, device)
    print("Model loaded!")
    
    test_dir = "data/stream-1-test"
    output_dir = "predictions_tta"
    os.makedirs(output_dir, exist_ok=True)
    
    colors = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 0, 255]}
    
    files = sorted([f for f in os.listdir(test_dir) if f.endswith("_img.jpg")])
    total = len(files)
    print(f"Processing {total} images with TTA...")
    
    for i, f in enumerate(files):
        img = np.array(Image.open(os.path.join(test_dir, f)))
        pred = tta_predict(model, img, device)
        
        mask_rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for c, color in colors.items():
            mask_rgb[pred == c] = color
        
        out_name = f.replace("_img.jpg", "_pred.png")
        Image.fromarray(mask_rgb).save(os.path.join(output_dir, out_name))
        
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"Progress: {i+1}/{total} ({100*(i+1)/total:.1f}%)")
            sys.stdout.flush()
    
    print(f"Done! Saved to {output_dir}/")

if __name__ == "__main__":
    main()
