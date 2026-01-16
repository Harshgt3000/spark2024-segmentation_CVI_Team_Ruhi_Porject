#!/bin/bash -l
#SBATCH --job-name=spark_resume
#SBATCH --output=logs/resume_%j.out
#SBATCH --error=logs/resume_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo "============================================"
echo " RESUME Original Model (85.97% test)"
echo " NO augmentation, lower LR, faster eval"
echo "============================================"

module purge
module load env/release/2023b
module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

source /scratch/users/hchaurasia/spark2024/venv/bin/activate
cd /scratch/users/hchaurasia/spark2024

# Resume from epoch 40 checkpoint (latest)
python train_resume_original.py \
    --epochs 30 \
    --batch-size 2 \
    --lr 5e-6 \
    --resume outputs/segformer_b5_20251220_174956/checkpoint_epoch40.pth

echo "Done: $(date)"
