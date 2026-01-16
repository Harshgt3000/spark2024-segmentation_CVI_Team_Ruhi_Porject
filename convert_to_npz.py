import os
import numpy as np
from PIL import Image
from tqdm import tqdm

input_dir = "predictions"
output_dir = "submission_npz"
os.makedirs(output_dir, exist_ok=True)

for f in tqdm(sorted(os.listdir(input_dir))):
    if f.endswith("_pred.png"):
        img = np.array(Image.open(os.path.join(input_dir, f)))
        
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[(img[:,:,0] > 200) & (img[:,:,1] < 50) & (img[:,:,2] < 50)] = 1  # Red = body
        mask[(img[:,:,2] > 200) & (img[:,:,0] < 50) & (img[:,:,1] < 50)] = 2  # Blue = solar
        
        name = f.replace("_pred.png", "_layer.npz")
        np.savez_compressed(os.path.join(output_dir, name), arr_0=mask)

print(f"Converted {len(os.listdir(output_dir))} files")
