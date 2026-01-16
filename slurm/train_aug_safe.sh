#!/bin/bash -l
#SBATCH --job-name=spark_safe
#SBATCH --output=logs/safe_%j.out
#SBATCH --error=logs/safe_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

echo "============================================================"
echo " SPARK 2024 - Safe Training (batch=1, less memory)"
echo "============================================================"

module purge
module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

source /scratch/users/hchaurasia/spark2024/venv/bin/activate
cd /scratch/users/hchaurasia/spark2024

CHECKPOINT="outputs/segformer_aug_20251231_192335/best_model.pth"
echo "Resuming from: $CHECKPOINT"

python train_augmented.py \
    --epochs 100 \
    --batch-size 1 \
    --resume "$CHECKPOINT"

echo "Done: $(date)"
