import os
import sys
import numpy as np
from PIL import Image

input_dir = "predictions_tta"
output_dir = "submission_tta"

os.makedirs(output_dir, exist_ok=True)

files = sorted([f for f in os.listdir(input_dir) if f.endswith("_pred.png")])
total = len(files)
print(f"Converting {total} files...")

for i, f in enumerate(files):
    img = np.array(Image.open(os.path.join(input_dir, f)))
    name = f.replace("_pred.png", "_layer.npz")
    np.savez_compressed(os.path.join(output_dir, name), data=img)
    
    if (i + 1) % 500 == 0 or (i + 1) == total:
        print(f"Progress: {i+1}/{total}")

print("Done!")
