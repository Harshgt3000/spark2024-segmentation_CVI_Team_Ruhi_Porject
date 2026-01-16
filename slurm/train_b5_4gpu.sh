#!/bin/bash -l
# =============================================================================
# SPARK 2024 - SegFormer-B5 Multi-GPU Training
# =============================================================================
# Features:
#   - 4 GPUs (Tesla V100)
#   - Email notifications on start/end/fail
#   - 48 hour time limit
#   - Checkpoints every 10 epochs (can resume if fails)
# 
# Submit: sbatch slurm/train_b5_4gpu.sh
# Monitor: tail -f logs/b5_*.out
# Cancel: scancel <JOBID>
# =============================================================================

#SBATCH --job-name=spark_b5
#SBATCH --output=logs/b5_%j.out
#SBATCH --error=logs/b5_%j.err

# Time: 48 hours (enough for 100 epochs)
#SBATCH --time=48:00:00


# Resources: 4 GPUs
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# Email notifications - CHANGE THIS TO YOUR EMAIL
#SBATCH --mail-user=harsh.chaurasia@uni.lu
#SBATCH --mail-type=BEGIN,END,FAIL

echo "============================================================"
echo " SPARK 2024 - SegFormer-B5 Multi-GPU Training"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "GPUs requested: 4"
echo "============================================================"

# Setup environment
module purge
module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

source /scratch/users/hchaurasia/spark2024/venv/bin/activate
cd /scratch/users/hchaurasia/spark2024

# Create directories
mkdir -p logs outputs

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Run training
# - B5 model (81M params)
# - 4 GPUs with batch size 4 each = 16 effective batch size
# - 100 epochs
# - Checkpoint every 10 epochs
# - Evaluate every 5 epochs

python train_b5_multigpu.py \
    --model nvidia/segformer-b5-finetuned-ade-640-640 \
    --epochs 100 \
    --batch-size 2 \
    --lr 2e-5 \
    --img-size 512 \
    --eval-interval 5 \
    --checkpoint-interval 10 \
    --num-workers 8

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "============================================================"
