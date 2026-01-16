#!/bin/bash -l
#SBATCH --job-name=convert
#SBATCH --output=logs/convert_%j.out
#SBATCH --error=logs/convert_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

cd /scratch/users/hchaurasia/spark2024
source venv/bin/activate

python convert_to_npz.py

cd submission_npz
zip -r ../submission.zip *.npz
echo "Done! Files: $(ls *.npz | wc -l)"
