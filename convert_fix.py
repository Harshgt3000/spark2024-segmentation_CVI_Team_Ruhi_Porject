import os
import sys
import numpy as np
from PIL import Image

input_dir = "predictions"
output_dir = "submission_fix"

os.makedirs(output_dir, exist_ok=True)

files = sorted([f for f in os.listdir(input_dir) if f.endswith("_pred.png")])
total = len(files)
print(f"Converting {total} files with key='data'...")
print("-" * 50)

for i, f in enumerate(files):
    img = np.array(Image.open(os.path.join(input_dir, f)))
    
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[(img[:,:,0] > 200) & (img[:,:,1] < 50) & (img[:,:,2] < 50)] = 1
    mask[(img[:,:,2] > 200) & (img[:,:,0] < 50) & (img[:,:,1] < 50)] = 2
    
    name = f.replace("_pred.png", "_layer.npz")
    
    # KEY FIX: Use 'data' as key name
    np.savez_compressed(os.path.join(output_dir, name), data=mask)
    
    if (i + 1) % 200 == 0 or (i + 1) == total:
        print(f"Progress: {i+1}/{total} ({100*(i+1)/total:.1f}%)")
        sys.stdout.flush()

print("-" * 50)
print(f"Done! Converted {len(os.listdir(output_dir))} files")

# Verify
f = np.load(f"{output_dir}/test_00000_layer.npz")
print(f"\n=== Verification ===")
print(f"Keys: {list(f.keys())}")
print(f"Shape: {f['data'].shape}")
print(f"Unique: {np.unique(f['data'])}")
