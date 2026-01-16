#!/bin/bash -l
#SBATCH --job-name=spark_simple_ens
#SBATCH --output=logs/simple_ens_%j.out
#SBATCH --error=logs/simple_ens_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module purge
module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0
source /scratch/users/hchaurasia/spark2024/venv/bin/activate
cd /scratch/users/hchaurasia/spark2024

python inference_simple_ensemble.py

python -c "
import os, numpy as np
from PIL import Image
input_dir = 'predictions_simple_ensemble'
output_dir = 'submission_simple_ensemble'
os.makedirs(output_dir, exist_ok=True)
files = sorted([f for f in os.listdir(input_dir) if f.endswith('_pred.png')])
for i, f in enumerate(files):
    img = np.array(Image.open(os.path.join(input_dir, f)))
    name = f.replace('_pred.png', '_layer.npz')
    np.savez_compressed(os.path.join(output_dir, name), data=img)
print(f'Done! {len(files)} files')
"

cd submission_simple_ensemble && zip -r ../submission_simple_ensemble.zip *.npz && cd ..
ls -lh submission_simple_ensemble.zip
echo "READY: submission_simple_ensemble.zip"
