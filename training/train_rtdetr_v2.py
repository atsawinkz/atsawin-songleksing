"""
Training script for RT-DETR-L (Version 2 - Optimized Hyperparameters)
Optimized for NVIDIA GeForce RTX 4090 (24GB VRAM)
"""
import os
from ultralytics import RTDETR

def main():
    # กำหนด Working Directory ให้อยู่ในโฟลเดอร์ training เสมอ
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    # Load COCO pre-trained RT-DETR-L
    model = RTDETR('rtdetr-l.pt')

    print(f"Starting RT-DETR-L Retraining (Version 2) with imgsz=640 on RTX 4090...")
    print(f"Working Directory: {current_dir}")
    print(f"Dataset configuration: dataset.yaml")

    results = model.train(
        data='dataset.yaml',
        epochs=100,             # Full 100 epochs
        imgsz=640,              # Standard 640x640 matching dataset resolution
        batch=16,               # Batch size for 24GB VRAM
        lr0=0.0005,             # Optimized starting learning rate
        lrf=0.01,               # Final learning rate ratio
        optimizer='AdamW',
        warmup_epochs=5,
        cos_lr=True,            # Cosine learning rate scheduler
        patience=25,            # Increased patience for early stopping
        mosaic=0.5,             # Augmentation to handle overlapping lesions
        mixup=0.1,              # Augmentation
        augment=True,
        device=0,
        project='runs',
        name='rtdetr_rice_v2',
        amp=True,               # Mixed precision FP16
        workers=16,
        cache=False,
        val=True,
        save=True
    )

    print("Retraining Completed! Best weights saved to runs/rtdetr_rice_v2/weights/best.pt")

if __name__ == '__main__':
    main()
