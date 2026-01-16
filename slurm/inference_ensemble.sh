#!/bin/bash -l
#SBATCH --job-name=spark_ensemble
#SBATCH --output=logs/ensemble_%j.out
#SBATCH --error=logs/ensemble_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

echo "============================================================"
echo " SPARK 2024 - Ensemble + TTA Inference"
echo "============================================================"

module purge
module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

source /scratch/users/hchaurasia/spark2024/venv/bin/activate
cd /scratch/users/hchaurasia/spark2024

CHECKPOINTS="outputs/segformer_b5_20251220_174956/checkpoint_epoch20.pth"
CHECKPOINTS="$CHECKPOINTS,outputs/segformer_b5_20251220_174956/checkpoint_epoch30.pth"
CHECKPOINTS="$CHECKPOINTS,outputs/segformer_b5_20251220_174956/checkpoint_epoch40.pth"
CHECKPOINTS="$CHECKPOINTS,outputs/segformer_b5_20251220_174956/best_model.pth"

python inference_advanced.py \
    --checkpoints "$CHECKPOINTS" \
    --test-dir data/stream-1-test \
    --output predictions_ensemble \
    --multiscale

python -c "
import os, numpy as np
from PIL import Image

input_dir = 'predictions_ensemble'
output_dir = 'submission_ensemble'
os.makedirs(output_dir, exist_ok=True)

files = sorted([f for f in os.listdir(input_dir) if f.endswith('_pred.png')])
print(f'Converting {len(files)} files...')

for i, f in enumerate(files):
    img = np.array(Image.open(os.path.join(input_dir, f)))
    name = f.replace('_pred.png', '_layer.npz')
    np.savez_compressed(os.path.join(output_dir, name), data=img)
    if (i + 1) % 500 == 0: print(f'{i+1}/{len(files)}')

print('Done!')
"

cd submission_ensemble && zip -r ../submission_ensemble.zip *.npz && cd ..
ls -lh submission_ensemble.zip
echo "Ready: submission_ensemble.zip"
