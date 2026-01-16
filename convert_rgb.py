import os
import sys
import numpy as np
from PIL import Image

input_dir = "predictions"
output_dir = "submission_rgb"

os.makedirs(output_dir, exist_ok=True)

files = sorted([f for f in os.listdir(input_dir) if f.endswith("_pred.png")])
total = len(files)
print(f"Converting {total} files to RGB format...")
print("-" * 50)

for i, f in enumerate(files):
    # Load prediction PNG (already RGB)
    img = np.array(Image.open(os.path.join(input_dir, f)))
    
    # Keep as RGB (1024, 1024, 3)
    name = f.replace("_pred.png", "_layer.npz")
    
    # Save RGB array with key 'data'
    np.savez_compressed(os.path.join(output_dir, name), data=img)
    
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
print(f"Dtype: {f['data'].dtype}")
