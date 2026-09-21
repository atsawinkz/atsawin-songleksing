"""
Training script for RT-DETR-L (Version 5 - Data-Centric AI on Cleaned BLB Annotations)
Fine-tuning from Champion Model v4 weights on NVIDIA GeForce RTX 4090 (24GB VRAM)
"""
import os
import glob
import warnings
import torch
from ultralytics import RTDETR

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*grid_sampler_2d_backward_cuda.*")

torch.backends.cudnn.benchmark = True

def find_best_starting_weights():
    candidate_paths = [
        "runs/detect/runs/rtdetr_rice_v4/weights/best.pt",
        "runs/rtdetr_rice_v4/weights/best.pt",
        "runs/detect/runs/rtdetr_rice_v3/weights/best.pt",
        "runs/rtdetr_rice_v3/weights/best.pt",
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    
    v4_files = glob.glob("**/rtdetr_rice_v4/**/best.pt", recursive=True)
    if v4_files:
        return v4_files[0]
        
    return 'rtdetr-l.pt'

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    base_weights = find_best_starting_weights()
    print(f"Loading Base Weights (v4 Champion): {base_weights}")
    model = RTDETR(base_weights)

    print(f"Starting RT-DETR-L Fine-Tuning (Version 5 - Data-Centric BLB) on RTX 4090...")
    print(f"Working Directory: {current_dir}")
    print(f"Dataset configuration: dataset_v5.yaml")

    results = model.train(
        data='dataset_v5.yaml',
        epochs=100,             # 100 epochs with early stopping
        imgsz=800,              # High-resolution 800x800 for optimal spatial details
        batch=16,               # Batch size for RTX 4090 24GB
        lr0=0.0001,             # Gentle fine-tuning learning rate
        lrf=0.01,
        optimizer='AdamW',
        warmup_epochs=3,
        cos_lr=True,
        patience=20,            # Early stopping patience
        mosaic=0.0,             # Keep disabled to preserve natural continuous lesions
        mixup=0.0,
        fliplr=0.5,
        flipud=0.5,
        degrees=15.0,
        scale=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        box=8.5,
        augment=True,
        device=0,
        project='runs',
        name='rtdetr_rice_v5',
        amp=True,
        workers=16,
        cache='disk',
        deterministic=False,
        val=True,
        save=True
    )

    print("Retraining v5 Completed! Best weights saved to runs/rtdetr_rice_v5/weights/best.pt")

if __name__ == '__main__':
    main()
