#!/bin/bash -l
#SBATCH --job-name=spark_tta97
#SBATCH --output=logs/tta97_%j.out
#SBATCH --error=logs/tta97_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module purge
module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0
source /scratch/users/hchaurasia/spark2024/venv/bin/activate
cd /scratch/users/hchaurasia/spark2024

BEST_MODEL=$(ls -t outputs/segformer_resume_*/best_model.pth | head -1)
echo "Using: $BEST_MODEL"

python inference_tta_97.py \
    --checkpoint "$BEST_MODEL" \
    --test-dir data/stream-1-test \
    --output predictions_tta_97

# Convert and zip
python -c "
import os, numpy as np
from PIL import Image
input_dir = 'predictions_tta_97'
output_dir = 'submission_tta_97'
os.makedirs(output_dir, exist_ok=True)
files = sorted([f for f in os.listdir(input_dir) if f.endswith('_pred.png')])
for i, f in enumerate(files):
    img = np.array(Image.open(os.path.join(input_dir, f)))
    name = f.replace('_pred.png', '_layer.npz')
    np.savez_compressed(os.path.join(output_dir, name), data=img)
print(f'Done! {len(files)} files')
"

cd submission_tta_97 && zip -r ../submission_tta_97.zip *.npz && cd ..
ls -lh submission_tta_97.zip
