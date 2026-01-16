#!/bin/bash -l
#SBATCH --job-name=spark_opt
#SBATCH --output=logs/opt_%j.out
#SBATCH --error=logs/opt_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo "============================================================"
echo " SPARK 2024 - OPTIMIZED Training (Expert Recommendations)"
echo "============================================================"
echo "Features:"
echo "  - torch.inference_mode() for faster eval"
echo "  - FP16 inference (50% memory reduction)"
echo "  - Streaming mIoU via confusion matrix"
echo "  - Mixed precision training with GradScaler"
echo "  - Proper gc.collect() + empty_cache() placement"
echo "============================================================"

module purge
module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

source /scratch/users/hchaurasia/spark2024/venv/bin/activate
cd /scratch/users/hchaurasia/spark2024

# Set CUDA memory allocator config
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.8

CHECKPOINT="outputs/segformer_aug_20251231_192335/best_model.pth"
echo "Resuming from: $CHECKPOINT"
echo "Starting at: $(date)"

python train_optimized.py \
    --epochs 100 \
    --batch-size 2 \
    --resume "$CHECKPOINT"

echo "Done: $(date)"
