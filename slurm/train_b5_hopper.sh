#!/bin/bash -l
# =============================================================================
# SPARK 2024 - SegFormer-B5 on H100 Hopper GPU
# =============================================================================

#SBATCH --job-name=spark_h100
#SBATCH --output=logs/h100_%j.out
#SBATCH --error=logs/h100_%j.err

#SBATCH --time=72:00:00
#SBATCH --partition=hopper
#SBATCH --qos=iris-hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

#SBATCH --mail-user=harsh.chaurasia.001@student.uni.lu
#SBATCH --mail-type=BEGIN,END,FAIL

echo "============================================================"
echo " SPARK 2024 - SegFormer-B5 on H100 Hopper"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "============================================================"

# Setup environment
module purge
module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

source /scratch/users/hchaurasia/spark2024/venv/bin/activate
cd /scratch/users/hchaurasia/spark2024

mkdir -p logs outputs

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# H100 can handle larger batch size!
python train_b5_multigpu.py \
    --model nvidia/segformer-b5-finetuned-ade-640-640 \
    --epochs 100 \
    --batch-size 8 \
    --lr 2e-5 \
    --img-size 512 \
    --eval-interval 5 \
    --checkpoint-interval 10 \
    --num-workers 8

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "============================================================"
