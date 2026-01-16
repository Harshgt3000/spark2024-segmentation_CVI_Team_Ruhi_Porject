#!/bin/bash -l
#SBATCH --job-name=spark_noaug
#SBATCH --output=logs/noaug_%j.out
#SBATCH --error=logs/noaug_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo "============================================"
echo " NO AUGMENTATION - Better Test Generalization"
echo "============================================"

module purge
module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

source /scratch/users/hchaurasia/spark2024/venv/bin/activate
cd /scratch/users/hchaurasia/spark2024

python train_no_aug.py --epochs 50 --batch-size 2

echo "Done: $(date)"
