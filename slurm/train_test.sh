#!/bin/bash
# =============================================================================
# SPARK 2024 - Quick Test Run (5 epochs, limited data)
# =============================================================================
# Submit: sbatch slurm/train_test.sh
# =============================================================================

#SBATCH --job-name=spark_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

echo "=============================================="
echo " SPARK 2024 - Quick Test Run"
echo "=============================================="

# Environment
module purge
module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0
source /scratch/users/hchaurasia/spark2024/venv/bin/activate

cd /scratch/users/hchaurasia/spark2024
mkdir -p logs outputs

nvidia-smi

# Quick test with limited data
python train_segformer.py \
    --model nvidia/segformer-b0-finetuned-ade-512-512 \
    --epochs 5 \
    --batch-size 4 \
    --max-samples 50 \
    --eval-interval 1 \
    --amp

echo "Test complete!"
