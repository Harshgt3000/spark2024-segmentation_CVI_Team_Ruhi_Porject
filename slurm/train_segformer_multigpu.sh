#!/bin/bash
# =============================================================================
# SPARK 2024 - Multi-GPU SegFormer Training (RECOMMENDED)
# =============================================================================
# Submit: sbatch slurm/train_segformer_multigpu.sh
# Monitor: tail -f logs/segformer_*.out
# =============================================================================

#SBATCH --job-name=spark_seg
#SBATCH --output=logs/segformer_%j.out
#SBATCH --error=logs/segformer_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

echo "=============================================="
echo " SPARK 2024 - SegFormer Multi-GPU Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=============================================="

# Environment
module purge
module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0
source /scratch/users/hchaurasia/spark2024/venv/bin/activate

cd /scratch/users/hchaurasia/spark2024
mkdir -p logs outputs

# GPU info
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Distributed setup
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4

# Train with SegFormer-B5 (81M params, best performance)
# Batch size 4 per GPU x 4 GPUs = 16 effective batch size
# Gradient accumulation 2 = 32 effective batch size

python train_segformer.py \
    --model nvidia/segformer-b5-finetuned-ade-640-640 \
    --epochs 100 \
    --batch-size 4 \
    --lr 6e-5 \
    --grad-accum 2 \
    --img-size 512 \
    --amp

echo ""
echo "=============================================="
echo "Completed: $(date)"
echo "=============================================="
