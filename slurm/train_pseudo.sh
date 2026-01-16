#!/bin/bash -l
#SBATCH --job-name=spark_pseudo
#SBATCH --output=logs/pseudo_%j.out
#SBATCH --error=logs/pseudo_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

module purge
module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0
source /scratch/users/hchaurasia/spark2024/venv/bin/activate
cd /scratch/users/hchaurasia/spark2024

BEST_MODEL=$(ls -t outputs/segformer_resume_*/best_model.pth | head -1)
echo "Using model: $BEST_MODEL"

python train_pseudo.py \
    --resume "$BEST_MODEL" \
    --pred-dir predictions_tta_97 \
    --epochs 10 \
    --batch-size 4 \
    --lr 1e-6

echo "Done!"
