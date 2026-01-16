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
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --constraint=volta32
#SBATCH --mail-user=harsh.chaurasia.001@student.uni.lu
#SBATCH --mail-type=BEGIN,END,FAIL

echo "============================================================"
echo " SPARK 2024 - Resume Training (32GB GPUs, Batch 4)"
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

# Find latest checkpoint
CHECKPOINT=$(ls -t outputs/segformer_b5_*/checkpoint_epoch*.pth 2>/dev/null | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "No checkpoint found! Starting fresh..."
    RESUME_FLAG=""
else
    echo "Resuming from: $CHECKPOINT"
    RESUME_FLAG="--resume $CHECKPOINT"
fi

python train_b5_multigpu.py \
    --model nvidia/segformer-b5-finetuned-ade-640-640 \
    --epochs 100 \
    --batch-size 2 \
    --lr 2e-5 \
    --img-size 512 \
    --eval-interval 5 \
    --checkpoint-interval 10 \
    $RESUME_FLAG

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"
