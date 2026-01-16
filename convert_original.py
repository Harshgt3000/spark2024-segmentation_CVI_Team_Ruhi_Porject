import os
import numpy as np
from PIL import Image

input_dir = "predictions_original"
output_dir = "submission_original"
os.makedirs(output_dir, exist_ok=True)

files = sorted([f for f in os.listdir(input_dir) if f.endswith("_pred.png")])
print(f"Converting {len(files)} files...")

for i, f in enumerate(files):
    img = np.array(Image.open(os.path.join(input_dir, f)))
    
    # Ensure RGB format
    if len(img.shape) == 2:
        rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        rgb[img == 1] = [255, 0, 0]
        rgb[img == 2] = [0, 0, 255]
        img = rgb
    
    name = f.replace("_pred.png", "_layer.npz")
    np.savez_compressed(os.path.join(output_dir, name), data=img)
    
    if (i + 1) % 500 == 0:
        print(f"Progress: {i+1}/{len(files)}")

print(f"Done! {len(os.listdir(output_dir))} files")

# Verify
sample = np.load(f"{output_dir}/{sorted(os.listdir(output_dir))[0]}")
print(f"Keys: {list(sample.keys())}, Shape: {sample['data'].shape}")
