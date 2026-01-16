#!/bin/bash -l
#SBATCH --job-name=spark_aug4
#SBATCH --output=logs/aug4_%j.out
#SBATCH --error=logs/aug4_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo "============================================================"
echo " SPARK 2024 - Continue Training from Last Checkpoint"
echo "============================================================"

module purge
module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

source /scratch/users/hchaurasia/spark2024/venv/bin/activate
cd /scratch/users/hchaurasia/spark2024

# Find latest checkpoint (prefer checkpoint_epoch > best_model)
CHECKPOINT=$(ls -t outputs/segformer_aug_*/checkpoint_epoch*.pth 2>/dev/null | head -1)
if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(ls -t outputs/segformer_aug_*/best_model.pth 2>/dev/null | head -1)
fi

echo "Found checkpoint: $CHECKPOINT"
echo "Starting at: $(date)"

python train_augmented.py \
    --epochs 100 \
    --batch-size 2 \
    --resume "$CHECKPOINT"

echo "Done: $(date)"
