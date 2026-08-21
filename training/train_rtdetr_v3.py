"""
Training script for RT-DETR-L (Version 3 - Morphological Preservation & Fine-tuning)
Optimized for NVIDIA GeForce RTX 4090 (24GB VRAM)
"""
import os
import glob
from ultralytics import RTDETR

def find_best_starting_weights():
    # ลำดับความสำคัญ: 1. v2 weights -> 2. v1 weights -> 3. COCO pre-trained rtdetr-l.pt
    candidate_paths = [
        "runs/detect/runs/rtdetr_rice_v2/weights/best.pt",
        "runs/rtdetr_rice_v2/weights/best.pt",
        "runs/detect/runs/rtdetr_rice_v1-4/weights/best.pt",
        "runs/rtdetr_rice_v1-4/weights/best.pt"
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    
    # ค้นหาแบบ recursive เผื่ออยู่โฟลเดอร์อื่น
    v2_files = glob.glob("**/rtdetr_rice_v2/weights/best.pt", recursive=True)
    if v2_files:
        return v2_files[0]
        
    return 'rtdetr-l.pt'

def main():
    # กำหนด Working Directory ให้อยู่ในโฟลเดอร์ training เสมอ
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    base_weights = find_best_starting_weights()
    print(f"Loading Base Weights: {base_weights}")
    model = RTDETR(base_weights)

    print(f"Starting RT-DETR-L Retraining (Version 3 - Morphological Preservation) on RTX 4090...")
    print(f"Working Directory: {current_dir}")
    print(f"Dataset configuration: dataset.yaml")

    results = model.train(
        data='dataset.yaml',
        epochs=100,             # Full 100 epochs with early stopping
        imgsz=640,              # Standard 640x640 matching dataset resolution
        batch=16,               # Batch size for 24GB VRAM
        lr0=0.0002,             # Fine-tuning learning rate (milder for pre-trained rice weights)
        lrf=0.01,               # Final learning rate ratio
        optimizer='AdamW',
        warmup_epochs=3,        # Mild warmup
        cos_lr=True,            # Cosine learning rate scheduler
        patience=25,            # Patience for early stopping
        mosaic=0.0,             # DISABLED: ปิด Mosaic เพื่อรักษาความหนาแน่นพิกเซลของรอยโรคแคบยาว
        mixup=0.0,              # DISABLED: ปิด Mixup เพื่อรักษาขอบเขตรอยโรคให้คมชัด
        fliplr=0.5,             # พลิกภาพแนวนอน
        flipud=0.5,             # พลิกภาพแนวตั้ง
        degrees=15.0,           # หมุนภาพสุ่มไม่เกิน 15 องศา
        scale=0.2,              # สุ่มปรับสเกลขนาดภาพเบาๆ
        hsv_h=0.015,            # ปรับเฉดสีจำลองแสงแดดแปลงนา
        hsv_s=0.7,
        hsv_v=0.4,
        augment=True,
        device=0,
        project='runs',
        name='rtdetr_rice_v3',
        amp=True,               # Mixed precision FP16
        workers=16,
        cache=False,
        val=True,
        save=True
    )

    print("Retraining v3 Completed! Best weights saved to runs/rtdetr_rice_v3/weights/best.pt")

if __name__ == '__main__':
    main()
