"""
Training Script for Segmented Rice Leaf Dataset (Ablation Study)
Controlled Experiment: Identical hyperparameters to Champion Model v5
To be executed on NVIDIA GeForce RTX 4090 (24GB VRAM)
"""
import os
import glob
import warnings
import torch
from pathlib import Path
from ultralytics import RTDETR

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*grid_sampler_2d_backward_cuda.*")

torch.backends.cudnn.benchmark = True

def find_starting_weights():
    candidate_paths = [
        "../../training/runs/detect/runs/rtdetr_rice_v4/weights/best.pt",
        "../../training/runs/rtdetr_rice_v4/weights/best.pt",
        "../../runs/rtdetr_rice_v4/weights/best.pt",
        "../../runs/detect/runs/rtdetr_rice_v4/weights/best.pt",
        "rtdetr-l.pt"
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
            
    v4_files = glob.glob("../../**/rtdetr_rice_v4/**/best.pt", recursive=True)
    if v4_files:
        return v4_files[0]
        
    return 'rtdetr-l.pt'

def main():
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    dataset_yaml = script_dir / "dataset_segmented" / "data.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Segmented dataset config not found at: {dataset_yaml}")

    base_weights = find_starting_weights()
    print(f"Loading Base Weights: {base_weights}")
    model = RTDETR(base_weights)

    print("=" * 60)
    print("Starting RT-DETR-L Training on Segmented Rice Leaves (Ablation Study)")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Dataset: {dataset_yaml}")
    print("=" * 60)

    results = model.train(
        data=str(dataset_yaml),
        epochs=100,             # 100 epochs with early stopping
        imgsz=800,              # Identical to v5 (High-resolution 800x800)
        batch=16,               # Batch size for RTX 4090 24GB
        lr0=0.0001,             # Identical to v5
        lrf=0.01,
        optimizer='AdamW',
        warmup_epochs=3,
        cos_lr=True,
        patience=20,            # Early stopping
        mosaic=0.0,             # Preserved as 0.0 (identical to v5)
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
        device=0 if torch.cuda.is_available() else 'cpu',
        project=str(script_dir / 'runs'),
        name='rtdetr_segmented_ablation',
        amp=True,
        workers=8,
        cache='disk',
        deterministic=False,
        val=True,
        save=True
    )

    print("=" * 60)
    print("Ablation Study Training Completed!")
    print(f"Weights saved in: {script_dir / 'runs' / 'rtdetr_segmented_ablation' / 'weights'}")
    print("=" * 60)

if __name__ == '__main__':
    main()
