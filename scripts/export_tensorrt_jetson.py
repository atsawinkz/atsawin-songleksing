#!/usr/bin/env python3
"""
scripts/export_tensorrt_jetson.py
สคริปต์คอมไพล์โมเดล RT-DETR-L v5 เป็น NVIDIA TensorRT Engine (.engine) บน Jetson AGX Orin
โครงการ: ระบบวินิจฉัยโรคข้าวและให้คำแนะนำด้วย LINE Chatbot (RT-DETR + RAG)

ประโยชน์:
- เร่งความเร็วการประมวลผล (Inference) สูงสุดด้วย Tensor Cores ของชิป Tegra Orin
- ลด Latency เหลือต่ำกว่า 5-10 ms ต่อภาพ
- ลดการใช้พลังงาน (Power Consumption) บนบอร์ด Jetson
"""

import os
import sys
import glob
from ultralytics import RTDETR

def main():
    print("=" * 65)
    print("NVIDIA Jetson AGX Orin — TensorRT Engine Exporter (RT-DETR-L v5)")
    print("=" * 65)
    
    # 1. ค้นหาไฟล์ weights ของโมเดล v5 Champion
    pt_files = glob.glob("**/rtdetr_rice_v5/weights/best.pt", recursive=True)
    if not pt_files:
        pt_files = glob.glob("**/weights/best.pt", recursive=True)
        
    if not pt_files:
        print("[FAIL] ไม่พบไฟล์ best.pt ในระบบ! กรุณาตรวจสอบตำแหน่งไฟล์")
        sys.exit(1)
        
    model_path = pt_files[0]
    print(f"[INFO] พบไฟล์โมเดลต้นทาง: {model_path}")
    
    # 2. โหลดโมเดล
    print("[INFO] กำลังโหลดโมเดล RT-DETR...")
    model = RTDETR(model_path)
    
    # 3. ส่งออกเป็น TensorRT Engine
    print("[INFO] กำลังสร้าง TensorRT Engine (FP16 Half Precision, imgsz=800, device=0)...")
    print("   (ขั้นตอนนี้ใช้เวลาประมาณ 3-8 นาทีในการ Optimize Layer บน GPU Tegra Orin)")
    
    try:
        exported_path = model.export(
            format='engine',
            half=True,       # ใช้ FP16 Half Precision ประสิทธิภาพสูงสุดบน Tensor Cores
            device=0,        # รันบน Tegra GPU
            imgsz=800,       # ขนาดภาพเดียวกับโมเดล v5
            workspace=4,     # จัดสรร Workspace 4GB สำหรับการ Optimize
            verbose=True
        )
        print("\n" + "=" * 65)
        print(f"[SUCCESS] สร้าง TensorRT Engine สำเร็จเรียบร้อย!")
        print(f"บันทึกไฟล์ที่: {exported_path}")
        print("[NOTE] เมื่อรัน api/app.py ระบบจะตรวจจับและเรียกใช้ไฟล์ .engine นี้เป็นลำดับแรกอัตโนมัติ")
        print("=" * 65)
    except Exception as e:
        print(f"\n[FAIL] การคอมไพล์ TensorRT ล้มเหลว: {e}")
        print("[NOTE] หมายเหตุ: หากเครื่อง Jetson ยังไม่มี TensorRT สามารถใช้งาน best.onnx ต่อไปได้ตามปกติครับ")

if __name__ == '__main__':
    main()
