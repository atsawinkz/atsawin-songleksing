"""
Training script for RT-DETR-L (Version 4 - High-Resolution & Hardware-Accelerated SOTA)
Optimized for NVIDIA GeForce RTX 4090 (24GB VRAM) & Intel Core i9-14900K (64GB RAM)
"""
import os
import glob
import torch
from ultralytics import RTDETR

# บังคับเปิด cuDNN Benchmark สำหรับรูปขนาดคงที่ 800x800 เพื่อเร่งความเร็ว CUDA Kernels
torch.backends.cudnn.benchmark = True

def find_best_starting_weights():
    # ลำดับความสำคัญ: 1. v3 weights (Champion Model) -> 2. v2 weights -> 3. v1 weights -> 4. COCO
    candidate_paths = [
        "runs/detect/runs/rtdetr_rice_v3/weights/best.pt",
        "runs/rtdetr_rice_v3/weights/best.pt",
        "runs/detect/runs/rtdetr_rice_v2/weights/best.pt",
        "runs/rtdetr_rice_v2/weights/best.pt",
        "runs/detect/runs/rtdetr_rice_v1-4/weights/best.pt",
        "runs/rtdetr_rice_v1-4/weights/best.pt"
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    
    # ค้นหาแบบ recursive เผื่ออยู่โฟลเดอร์อื่น
    v3_files = glob.glob("**/rtdetr_rice_v3/weights/best.pt", recursive=True)
    if v3_files:
        return v3_files[0]
        
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

    print(f"Starting RT-DETR-L Retraining (Version 4 - High-Resolution 800px) on RTX 4090...")
    print(f"Working Directory: {current_dir}")
    print(f"Dataset configuration: dataset.yaml")

    results = model.train(
        data='dataset.yaml',
        epochs=100,             # Full 100 epochs with early stopping
        imgsz=800,              # High-resolution 800x800 (+56% pixel density for narrow lesions)
        batch=16,               # Batch size for 800x800 on 24GB VRAM
        lr0=0.0001,             # Fine-tuning learning rate on new resolution scale
        lrf=0.01,               # Final learning rate ratio
        optimizer='AdamW',
        warmup_epochs=3,        # Mild warmup
        cos_lr=True,            # Cosine learning rate scheduler
        patience=25,            # Patience for early stopping
        mosaic=0.0,             # DISABLED: ปิด Mosaic เพื่อรักษาสัณฐานวิทยาแผลริ้วยาว
        mixup=0.0,              # DISABLED: ปิด Mixup เพื่อรักษาขอบเขตรอยโรคให้คมชัด
        fliplr=0.5,             # พลิกภาพแนวนอน
        flipud=0.5,             # พลิกภาพแนวตั้ง
        degrees=15.0,           # หมุนภาพสุ่มไม่เกิน 15 องศา
        scale=0.2,              # สุ่มปรับสเกลขนาดภาพเบาๆ
        hsv_h=0.015,            # ปรับเฉดสีจำลองแสงแดดแปลงนา
        hsv_s=0.7,
        hsv_v=0.4,
        box=8.5,                # ปรับเพิ่มค่าน้ำหนัก Box Loss เพื่อกระชับกรอบรอยโรค
        augment=True,
        device=0,
        project='runs',
        name='rtdetr_rice_v4',
        amp=True,               # Mixed precision FP16
        workers=8,              # 8 Worker threads สำหรับ 8 P-Cores ของ i9-14900K
        cache='ram',            # แคชภาพลง 64GB RAM ตัดคอขวดดิสก์ 100%
        val=True,
        save=True
    )

    print("Retraining v4 Completed! Best weights saved to runs/rtdetr_rice_v4/weights/best.pt")

if __name__ == '__main__':
    main()
