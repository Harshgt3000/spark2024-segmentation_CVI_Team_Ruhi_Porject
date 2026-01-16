#!/usr/bin/env python3
"""
SPARK 2024 - Setup Verification
================================
Run this after setup_env.sh to verify everything works.
"""

import sys

def check(name, test_func):
    """Run a check and print result."""
    try:
        result = test_func()
        print(f"✓ {name}: {result}")
        return True
    except Exception as e:
        print(f"✗ {name}: {e}")
        return False

def main():
    print("=" * 50)
    print(" SPARK 2024 - Setup Verification")
    print("=" * 50)
    
    errors = []
    
    # Check PyTorch
    if not check("PyTorch", lambda: __import__('torch').__version__):
        errors.append("PyTorch")
    
    # Check CUDA
    if not check("CUDA available", lambda: __import__('torch').cuda.is_available()):
        errors.append("CUDA")
    
    # Check GPU
    def get_gpu():
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "No GPU"
    check("GPU", get_gpu)
    
    # Check transformers
    if not check("Transformers", lambda: __import__('transformers').__version__):
        errors.append("Transformers")
    
    # Check SegFormer
    def check_segformer():
        from transformers import SegformerForSemanticSegmentation
        return "Available"
    if not check("SegFormer", check_segformer):
        errors.append("SegFormer")
    
    # Check albumentations
    if not check("Albumentations", lambda: __import__('albumentations').__version__):
        errors.append("Albumentations")
    
    # Check dataset
    def check_dataset():
        import os
        data_root = '/scratch/users/hchaurasia/spark2024/data'
        if os.path.exists(data_root):
            imgs = os.path.join(data_root, 'images')
            masks = os.path.join(data_root, 'mask')
            if os.path.exists(imgs) and os.path.exists(masks):
                classes = len([d for d in os.listdir(imgs) if os.path.isdir(os.path.join(imgs, d))])
                return f"{classes} spacecraft classes"
        return "NOT FOUND"
    check("Dataset", check_dataset)
    
    # Check custom dataset loading
    def check_spark_dataset():
        from spark_dataset import SparkDataset
        ds = SparkDataset(
            data_root='/scratch/users/hchaurasia/spark2024/data',
            split='train',
            max_samples_per_class=2
        )
        return f"{len(ds)} samples loaded"
    if not check("SparkDataset", check_spark_dataset):
        errors.append("SparkDataset")
    
    # Summary
    print("\n" + "=" * 50)
    if errors:
        print(f"✗ ISSUES FOUND: {', '.join(errors)}")
        print("Please fix before training.")
        return False
    else:
        print("✓ ALL CHECKS PASSED!")
        print("\nNext steps:")
        print("  1. Quick test: sbatch slurm/train_test.sh")
        print("  2. Full training: sbatch slurm/train_segformer_multigpu.sh")
        return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
