import os
import sys
import numpy as np
from PIL import Image

input_dir = "predictions_aug"
output_dir = "submission_aug"

os.makedirs(output_dir, exist_ok=True)

files = sorted([f for f in os.listdir(input_dir) if f.endswith("_pred.png")])
total = len(files)
print(f"Converting {total} files to RGB format...")

for i, f in enumerate(files):
    img = np.array(Image.open(os.path.join(input_dir, f)))
    name = f.replace("_pred.png", "_layer.npz")
    np.savez_compressed(os.path.join(output_dir, name), data=img)
    
    if (i + 1) % 500 == 0 or (i + 1) == total:
        print(f"Progress: {i+1}/{total}")
        sys.stdout.flush()

print(f"Done! {len(os.listdir(output_dir))} files")

# Verify
f = np.load(f"{output_dir}/test_00000_layer.npz")
print(f"Shape: {f['data'].shape}, Keys: {list(f.keys())}")
