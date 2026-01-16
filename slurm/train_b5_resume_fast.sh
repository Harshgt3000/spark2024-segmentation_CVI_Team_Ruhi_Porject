#!/bin/bash -l
#SBATCH --job-name=spark_resume
#SBATCH --output=logs/resume_%j.out
#SBATCH --error=logs/resume_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=volta32
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-user=harsh.chaurasia.001@student.uni.lu
#SBATCH --mail-type=BEGIN,END,FAIL

# Confirmed path based on your ls output
RESUME_PATH="outputs/segformer_b5_20251220_174956/checkpoint_epoch20.pth"

echo "Resuming training from: $RESUME_PATH"
nvidia-smi

module purge
module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

source /scratch/users/hchaurasia/spark2024/venv/bin/activate
cd /scratch/users/hchaurasia/spark2024

# Fast mode: Batch size 4 on 32GB cards
python train_b5_multigpu.py \
    --model nvidia/segformer-b5-finetuned-ade-640-640 \
    --epochs 100 \
    --batch-size 4 \
    --lr 2e-5 \
    --img-size 512 \
    --eval-interval 5 \
    --checkpoint-interval 10 \
    --resume $RESUME_PATH

echo "Done: $(date)"
