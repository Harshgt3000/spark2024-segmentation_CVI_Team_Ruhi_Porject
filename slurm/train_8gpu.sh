#!/bin/bash -l
#SBATCH --job-name=spark_8gpu
#SBATCH --output=logs/8gpu_%j.out
#SBATCH --error=logs/8gpu_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --constraint=volta32
#SBATCH --mail-user=harsh.chaurasia.001@student.uni.lu
#SBATCH --mail-type=BEGIN,END,FAIL

echo "============================================================"
echo " SPARK 2024 - 8 GPU Training (2 Nodes)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "============================================================"

module purge
module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

source /scratch/users/hchaurasia/spark2024/venv/bin/activate
cd /scratch/users/hchaurasia/spark2024

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

srun torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_ddp.py \
    --epochs 100 \
    --batch-size 2 \
    --resume outputs/segformer_b5_20251220_174956/checkpoint_epoch20.pth

echo "Done: $(date)"
