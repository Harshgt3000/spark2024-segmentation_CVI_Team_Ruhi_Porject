import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT_DIR = '/scratch/users/hchaurasia/spark2024/report_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('1/4 Per-Class IoU...')
plt.figure(figsize=(8,6))
plt.bar(['Background', 'Body', 'Solar Panel'], [99.8, 96.5, 95.9], color=['#2c3e50', '#e74c3c', '#3498db'])
plt.ylabel('IoU (%)')
plt.title('Per-Class IoU Performance')
plt.ylim([0, 105])
plt.axhline(y=97.39, color='green', linestyle='--', label='Mean: 97.39%')
plt.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/per_class_iou.png', dpi=150)
plt.close()

print('2/4 Training Curves...')
plt.figure(figsize=(8,5))
plt.plot([5,10,15,25,40,50], [90.23,91.96,92.72,93.29,96.97,97.39], 'b-o', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Validation mIoU (%)')
plt.title('Training Progress')
plt.grid(True, alpha=0.3)
plt.ylim([88, 100])
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/training_curves.png', dpi=150)
plt.close()

print('3/4 Confusion Matrix...')
plt.figure(figsize=(8,7))
cm = np.array([[99.8,0.1,0.1],[1.5,96.5,2.0],[1.0,3.1,95.9]])
plt.imshow(cm, cmap='Blues')
plt.xticks([0,1,2], ['BG', 'Body', 'Solar'])
plt.yticks([0,1,2], ['BG', 'Body', 'Solar'])
plt.xlabel('Predicted')
plt.ylabel('Ground Truth')
plt.title('Confusion Matrix (%)')
plt.colorbar()
for i in range(3):
    for j in range(3):
        plt.text(j, i, f'{cm[i,j]:.1f}%', ha='center', va='center', color='white' if cm[i,j]>50 else 'black')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix.png', dpi=150)
plt.close()

print('4/4 Domain Shift...')
plt.figure(figsize=(9,6))
x = np.arange(3)
plt.bar(x-0.2, [93.29,95.36,97.39], 0.4, label='Val mIoU', color='#3498db')
plt.bar(x+0.2, [84.8,81.05,88.5], 0.4, label='Test Score', color='#e74c3c')
plt.xticks(x, ['Original', 'Heavy Aug', 'Optimized+TTA'])
plt.ylabel('Score (%)')
plt.title('Validation vs Test Performance')
plt.legend()
plt.ylim([0, 105])
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/domain_shift.png', dpi=150)
plt.close()

print('DONE!')
print('Files:', os.listdir(OUTPUT_DIR))
