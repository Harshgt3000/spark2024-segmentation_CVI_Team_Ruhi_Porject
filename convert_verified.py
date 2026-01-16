"""
Verified conversion script for Codabench submission
- Key: 'data'
- Shape: (1024, 1024, 3)
- Colors: black=background, red=body, blue=panels
- Filename: test_XXXXX_layer.npz
"""
import os
import numpy as np
from PIL import Image

input_dir = "predictions_opt"
output_dir = "submission_opt"
os.makedirs(output_dir, exist_ok=True)

files = sorted([f for f in os.listdir(input_dir) if f.endswith("_pred.png")])
print(f"Converting {len(files)} files...")
print("=" * 50)

errors = []
for i, f in enumerate(files):
    img = np.array(Image.open(os.path.join(input_dir, f)))
    
    # Verify shape is (1024, 1024, 3)
    if img.shape != (1024, 1024, 3):
        errors.append(f"{f}: shape {img.shape} != (1024, 1024, 3)")
    
    # Output filename: test_00000_pred.png -> test_00000_layer.npz
    name = f.replace("_pred.png", "_layer.npz")
    
    # Save with key='data'
    np.savez_compressed(os.path.join(output_dir, name), data=img)
    
    if (i + 1) % 500 == 0 or (i + 1) == len(files):
        print(f"Progress: {i+1}/{len(files)}")

print("=" * 50)

if errors:
    print(f"ERRORS: {len(errors)}")
    for e in errors[:5]:
        print(f"  {e}")
else:
    print("✓ No errors!")

# === VERIFICATION ===
print("\n" + "=" * 50)
print("VERIFICATION")
print("=" * 50)

# Check first file
first_file = sorted(os.listdir(output_dir))[0]
data = np.load(os.path.join(output_dir, first_file))

print(f"Sample file: {first_file}")
print(f"Keys: {list(data.keys())} (should be ['data'])")
print(f"Shape: {data['data'].shape} (should be (1024, 1024, 3))")
print(f"Dtype: {data['data'].dtype} (should be uint8)")

# Check colors
img = data['data']
unique_colors = set()
for color in [[0,0,0], [255,0,0], [0,0,255]]:
    mask = np.all(img == color, axis=-1)
    if mask.any():
        unique_colors.add(tuple(color))

print(f"Colors found: {unique_colors}")
print(f"  Black (0,0,0) = background: {'✓' if (0,0,0) in unique_colors else '✗'}")
print(f"  Red (255,0,0) = body: {'✓' if (255,0,0) in unique_colors else '✗'}")
print(f"  Blue (0,0,255) = panels: {'✓' if (0,0,255) in unique_colors else '✗'}")

# Count total files
total = len(os.listdir(output_dir))
print(f"\nTotal files: {total} (should be 4000)")
print("=" * 50)

if total == 4000 and list(data.keys()) == ['data'] and data['data'].shape == (1024, 1024, 3):
    print("✓ READY FOR SUBMISSION!")
else:
    print("✗ CHECK ERRORS ABOVE")
