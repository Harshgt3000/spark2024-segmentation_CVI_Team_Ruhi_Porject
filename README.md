# SPARK 2024 Spacecraft Segmentation

Semantic segmentation of spacecraft bodies and solar panels for the SPARK 2024 competition.

## Results
| Metric | Score |
|--------|-------|
| **Codabench Test** | **88.5%** |
| Validation mIoU | 97.39% |

## Model
- **Architecture**: SegFormer-B5
- **Loss**: 0.5 × CrossEntropy + 0.5 × DiceLoss
- **Input Resolution**: 1024×1024

## Key Insight
Simple Test-Time Augmentation (4 geometric flips) at full resolution outperformed complex ensemble methods.

## Usage

### Inference
```bash
python inference_tta_97.py \
    --checkpoint path/to/best_model.pth \
    --test-dir data/stream-1-test \
    --output predictions
```

## Files
- `inference_tta_97.py` - Inference with TTA
- `train_resume_original.py` - Training script
- `train_pseudo.py` - Pseudo-labeling training
- `spark_dataset.py` - Dataset loader

## Team
CVI Team Ruhi - University of Luxembourg
