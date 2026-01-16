#!/bin/bash -l
#SBATCH --job-name=spark_fast
#SBATCH --output=logs/fast_%j.out
#SBATCH --error=logs/fast_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-user=harsh.chaurasia.001@student.uni.lu
#SBATCH --mail-type=BEGIN,END,FAIL

echo "SPARK 2024 - Fast Training (384px)"
nvidia-smi

module purge
module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

source /scratch/users/hchaurasia/spark2024/venv/bin/activate
cd /scratch/users/hchaurasia/spark2024

python train_b5_multigpu.py \
    --model nvidia/segformer-b5-finetuned-ade-640-640 \
    --epochs 100 \
    --batch-size 6 \
    --lr 2e-5 \
    --img-size 384 \
    --eval-interval 5 \
    --checkpoint-interval 10

echo "Done: $(date)"
