#!/bin/bash -l
#SBATCH --job-name=spark_b5max
#SBATCH --output=logs/b5max_%j.out
#SBATCH --error=logs/b5max_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --constraint=volta32

echo "============================================================"
echo " SPARK 2024 - SegFormer-B5 MAXIMUM RESOURCES"
echo "============================================================"
echo " GPUs: 4x V100-32GB (128GB total VRAM)"
echo " RAM: 180GB system memory"
echo " CPUs: 16 cores"
echo " Model: SegFormer-B5 (84.6M params)"
echo "============================================================"

module purge
module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

source /scratch/users/hchaurasia/spark2024/venv/bin/activate
cd /scratch/users/hchaurasia/spark2024

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "Starting: $(date)"

CHECKPOINT="outputs/segformer_aug_20251231_192335/best_model.pth"
echo "Resuming from: $CHECKPOINT"

python train_b5_max.py \
    --model nvidia/segformer-b5-finetuned-ade-640-640 \
    --epochs 100 \
    --batch-size 2 \
    --lr 2e-5 \
    --eval-interval 5 \
    --resume "$CHECKPOINT"

echo "Done: $(date)"
