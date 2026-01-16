#!/bin/bash -l
#SBATCH --job-name=vis_results
#SBATCH --output=logs/vis_%j.out
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

module purge
module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0
source /scratch/users/hchaurasia/spark2024/venv/bin/activate
cd /scratch/users/hchaurasia/spark2024

# Using best_model.pth for visualization
CHECKPOINT="outputs/segformer_b5_20251220_174956/best_model.pth"

echo "Generating visualization from: $CHECKPOINT"

# We test on Proba3 validation images
python inference.py \
    --checkpoint $CHECKPOINT \
    --test-dir data/images/Proba3/val \
    --output visualization_results

echo "Visualization complete."
