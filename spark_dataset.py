"""
SPARK 2024 - PyTorch Dataset for HuggingFace Mask2Former
=========================================================
Scans directory structure directly (NO CSV file reading).
Handles RGB mask → semantic label conversion.

Directory structure:
    data/
    ├── images/
    │   ├── Cheops/train/, val/
    │   └── ...
    ├── mask/
    │   └── [same structure]

Mask RGB values:
    - Red (255, 0, 0)   → Class 1: Spacecraft body
    - Blue (0, 0, 255)  → Class 2: Solar panels  
    - Black (0, 0, 0)   → Class 0: Background
"""

import os
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SparkDataset(Dataset):
    """SPARK 2024 Spacecraft Segmentation Dataset."""
    
    # Class definitions
    CLASSES = ('background', 'spacecraft_body', 'solar_panel')
    NUM_CLASSES = 3
    
    # Color palette for visualization
    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 0, 255]]
    
    # Spacecraft folders (10 types)
    SPACECRAFT = [
        'Cheops', 'LisaPathfinder', 'ObservationSat1', 'Proba2',
        'Proba3', 'Proba3ocs', 'Smart1', 'Soho', 'VenusExpress', 'XMM Newton'
    ]
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        img_size: Tuple[int, int] = (512, 512),
        max_samples_per_class: Optional[int] = None,
    ):
        """Initialize SparkDataset.
        
        Args:
            data_root: Path to dataset root (contains images/ and mask/)
            split: 'train' or 'val'
            transform: Albumentations transform
            img_size: Target image size (H, W)
            max_samples_per_class: Limit samples per spacecraft (for debugging)
        """
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.max_samples_per_class = max_samples_per_class
        
        # Default transforms
        if transform is None:
            if split == 'train':
                self.transform = A.Compose([
                    A.Resize(img_size[0], img_size[1]),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(img_size[0], img_size[1]),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
        else:
            self.transform = transform
        
        # Load data list by scanning directories
        self.data_list = self._load_data_list()
        
    def _load_data_list(self) -> List[Dict]:
        """Load dataset by scanning directory structure."""
        data_list = []
        
        img_base = os.path.join(self.data_root, 'images')
        mask_base = os.path.join(self.data_root, 'mask')
        
        print(f"\n{'='*60}")
        print(f"Loading SparkDataset: {self.split}")
        print(f"Data root: {self.data_root}")
        print(f"{'='*60}")
        
        total_count = 0
        
        for spacecraft in self.SPACECRAFT:
            img_dir = os.path.join(img_base, spacecraft, self.split)
            mask_dir = os.path.join(mask_base, spacecraft, self.split)
            
            if not os.path.exists(img_dir):
                print(f"  [WARN] {spacecraft}/{self.split} not found, skipping")
                continue
            
            count = 0
            for filename in sorted(os.listdir(img_dir)):
                # Skip hidden files
                if filename.startswith('.') or filename.startswith('._'):
                    continue
                if not filename.endswith('_img.jpg'):
                    continue
                
                # Apply sample limit if set
                if self.max_samples_per_class and count >= self.max_samples_per_class:
                    break
                
                # Map image to mask filename
                base_name = filename.replace('_img.jpg', '')
                mask_filename = base_name + '_layer.jpg'
                
                img_path = os.path.join(img_dir, filename)
                mask_path = os.path.join(mask_dir, mask_filename)
                
                if not os.path.exists(mask_path):
                    continue
                
                data_list.append({
                    'img_path': img_path,
                    'mask_path': mask_path,
                    'spacecraft': spacecraft
                })
                count += 1
                total_count += 1
            
            print(f"  {spacecraft}: {count} samples")
        
        print(f"{'='*60}")
        print(f"Total {self.split} samples: {total_count}")
        print(f"{'='*60}\n")
        
        return data_list
    
    def _convert_rgb_mask_to_labels(self, rgb_mask: np.ndarray) -> np.ndarray:
        """Convert RGB mask to semantic labels.
        
        Args:
            rgb_mask: (H, W, 3) RGB mask array
            
        Returns:
            (H, W) label array with values 0, 1, 2
        """
        H, W = rgb_mask.shape[:2]
        labels = np.zeros((H, W), dtype=np.int64)
        
        R = rgb_mask[:, :, 0]
        G = rgb_mask[:, :, 1]
        B = rgb_mask[:, :, 2]
        
        # Class 1: Spacecraft body (red pixels)
        body_mask = (R > 200) & (G < 100) & (B < 100)
        labels[body_mask] = 1
        
        # Class 2: Solar panels (blue pixels)
        panel_mask = (R < 100) & (G < 100) & (B > 200)
        labels[panel_mask] = 2
        
        # Class 0: Background (black pixels) - already 0
        
        return labels
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        info = self.data_list[idx]
        
        # Load image
        image = np.array(Image.open(info['img_path']).convert('RGB'))
        
        # Load and convert mask
        rgb_mask = np.array(Image.open(info['mask_path']).convert('RGB'))
        mask = self._convert_rgb_mask_to_labels(rgb_mask)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return {
            'pixel_values': image,
            'labels': mask.long() if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.long),
        }


def get_train_transforms(img_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """Get training transforms with heavy augmentation."""
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """Get validation transforms (no augmentation)."""
    return A.Compose([
        A.Resize(img_size[0], img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def test_dataset():
    """Test dataset loading."""
    data_root = '/scratch/users/hchaurasia/spark2024/data'
    
    if not os.path.exists(data_root):
        print(f"Data root not found: {data_root}")
        return False
    
    # Test train
    train_ds = SparkDataset(
        data_root=data_root,
        split='train',
        max_samples_per_class=5
    )
    print(f"Train samples: {len(train_ds)}")
    
    # Test val
    val_ds = SparkDataset(
        data_root=data_root,
        split='val',
        max_samples_per_class=5
    )
    print(f"Val samples: {len(val_ds)}")
    
    # Check a sample
    if len(train_ds) > 0:
        sample = train_ds[0]
        print(f"\nSample:")
        print(f"  pixel_values shape: {sample['pixel_values'].shape}")
        print(f"  labels shape: {sample['labels'].shape}")
        print(f"  labels unique: {torch.unique(sample['labels']).tolist()}")
    
    return True


if __name__ == '__main__':
    test_dataset()
