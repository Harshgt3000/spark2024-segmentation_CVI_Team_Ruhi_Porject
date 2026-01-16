#!/bin/bash -l
#SBATCH --job-name=spark_aug
#SBATCH --output=logs/aug_%j.out
#SBATCH --error=logs/aug_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --constraint=volta32
#SBATCH --mail-user=harsh.chaurasia.001@student.uni.lu
#SBATCH --mail-type=BEGIN,END,FAIL

echo "============================================================"
echo " SPARK 2024 - Training WITH AUGMENTATION"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "============================================================"

module purge
module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

source /scratch/users/hchaurasia/spark2024/venv/bin/activate
cd /scratch/users/hchaurasia/spark2024

python train_augmented.py \
    --model nvidia/segformer-b5-finetuned-ade-640-640 \
    --epochs 100 \
    --batch-size 2 \
    --lr 2e-5 \
    --img-size 512 \
    --eval-interval 5 \
    --checkpoint-interval 10

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"
