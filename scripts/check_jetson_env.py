#!/usr/bin/env python3
"""
scripts/check_jetson_env.py
สคริปต์ตรวจสอบความพร้อมของระบบและวัดประสิทธิภาพ (Benchmark) บน NVIDIA Jetson AGX Orin
โครงการ: ระบบวินิจฉัยโรคข้าวและให้คำแนะนำด้วย LINE Chatbot (RT-DETR + RAG)
"""

import os
import sys
import time
import platform
import glob
from dotenv import load_dotenv

# ป้องกัน UnicodeEncodeError บน Windows หรือ non-UTF8 terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

load_dotenv()

def print_section(title):
    print("\n" + "=" * 60)
    print(f"=== {title} ===")
    print("=" * 60)

def check_system_info():
    print_section("1. ข้อมูลฮาร์ดแวร์และระบบปฏิบัติการ (Hardware & OS)")
    print(f"• OS / Platform   : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"• Python Version  : {sys.version.split()[0]}")
    
    # ตรวจสอบ RAM
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        mem_total = int([l.split()[1] for l in lines if 'MemTotal' in l][0]) // 1024
        print(f"• System RAM Total: {mem_total:,} MB (~{mem_total / 1024:.1f} GB Unified Memory)")
    except Exception:
        print("• System RAM Total: N/A")

    # ตรวจสอบ Tegra / Jetson Model
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model_name = f.read().strip().replace('\x00', '')
            print(f"• Jetson Model    : {model_name}")
    except Exception:
        print("• Jetson Model    : Unknown / Non-Tegra Board")

def check_cuda_and_pytorch():
    print_section("2. ตรวจสอบ PyTorch และการเร่งความเร็ว CUDA")
    try:
        import torch
        print(f"• PyTorch Version : {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"• CUDA Available  : {'[PASS] YES' if cuda_available else '[FAIL] NO (CPU Mode)'}")
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"• GPU Device Name : {device_name} (Total: {device_count} GPU)")
            print(f"• Unified VRAM    : {vram_gb:.2f} GB")
            
            # ทดสอบ Tensor พื้นฐานบน CUDA
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.matmul(x, x)
            print("• CUDA Tensor Ops : [PASS] (ความสามารถในการคำนวณ GPU ปกติ)")
        else:
            print("[WARNING] คำเตือน: PyTorch ตรวจไม่พบ CUDA บนเครื่องนี้ หากรันบน Jetson ควรใช้ PyTorch ที่บิลด์มากับ JetPack")
    except ImportError:
        print("[FAIL] ไม่พบ PyTorch ในสภาพแวดล้อมนี้ กรุณาติดตั้ง PyTorch")

def check_model_benchmark():
    print_section("3. ตรวจสอบและวัดความเร็วโมเดล RT-DETR-L")
    try:
        from ultralytics import RTDETR
        import numpy as np
        
        # ค้นหาไฟล์โมเดลโดยตรง (หลีกเลี่ยงปัญหา wildcard)
        target_model = None
        model_type = None
        for p, desc in [
            ("training/runs/detect/runs/rtdetr_rice_v5/weights/best.engine", "TensorRT Engine"),
            ("training/runs/detect/runs/rtdetr_rice_v5/weights/best.pt", "PyTorch Weights"),
            ("training/runs/detect/runs/rtdetr_rice_v5/weights/best.onnx", "ONNX FP16"),
        ]:
            if os.path.exists(p):
                target_model = p
                model_type = desc
                break
                
        if not target_model:
            # Fallback ค้นหาในโฟลเดอร์ weights ทั่วไป
            for ext, desc in [("*.engine", "TensorRT Engine"), ("*.pt", "PyTorch Weights"), ("*.onnx", "ONNX FP16")]:
                found = glob.glob(f"**/{ext}", recursive=True)
                if found:
                    found.sort(key=os.path.getmtime, reverse=True)
                    target_model = found[0]
                    model_type = desc
                    break
                
        if not target_model:
            print("[FAIL] ไม่พบไฟล์โมเดล (.engine, .pt, หรือ .onnx)")
            return
            
        print(f"• กำลังโหลดโมเดล : {target_model} ({model_type})")
        
        start_load = time.time()
        model = RTDETR(target_model)
        load_time = (time.time() - start_load) * 1000
        print(f"• เวลาโหลดโมเดล : {load_time:.2f} ms")
        
        # Benchmark 10 ครั้ง
        print("• เริ่มต้นรัน Benchmark การวินิจฉัย (ขนาดภาพ 800x800)...")
        dummy_img = np.zeros((800, 800, 3), dtype=np.uint8)
        
        # Warm-up 2 รอบ
        for _ in range(2):
            _ = model.predict(dummy_img, verbose=False, imgsz=800)
            
        times = []
        for _ in range(10):
            t0 = time.time()
            _ = model.predict(dummy_img, verbose=False, imgsz=800)
            times.append((time.time() - t0) * 1000)
            
        avg_time = sum(times) / len(times)
        fps = 1000.0 / avg_time if avg_time > 0 else 0
        print(f"• ค่าเฉลี่ย Latency : {avg_time:.2f} ms ต่อภาพ")
        print(f"• Throughput (FPS) : {fps:.1f} FPS")
        if avg_time < 30.0:
            print("• สถานะประสิทธิภาพ : [OK] ยอดเยี่ยม (Real-time Deployment Ready)")
        else:
            print("• สถานะประสิทธิภาพ : [WARN] พอใช้ได้ (แนะนำแปลงเป็น .engine สำหรับความเร็วสูงสุด)")
            
    except Exception as e:
        print(f"[FAIL] เกิดข้อผิดพลาดในการรัน Benchmark โมเดล: {e}")

def check_cloud_rag_apis():
    print_section("4. ตรวจสอบการเชื่อมต่อระบบ RAG (Gemini & Pinecone)")
    
    # 4.1 ตรวจสอบ Gemini API
    gemini_key = os.getenv('GEMINI_API_KEY', '')
    if not gemini_key:
        print("• Gemini API Key  : [FAIL] ไม่พบใน .env (GEMINI_API_KEY)")
    else:
        print(f"• Gemini API Key  : [PASS] พร้อมใช้งาน (ขึ้นต้นด้วย {gemini_key[:6]}...)")
        try:
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={gemini_key}"
            payload = {"model": "models/gemini-embedding-001", "content": {"parts": [{"text": "ทดสอบระบบโรคข้าว"}]}}
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code == 200:
                emb_len = len(r.json().get('embedding', {}).get('values', []))
                print(f"• Gemini Embedding: [PASS] เชื่อมต่อสำเร็จ (ได้เวกเตอร์มิติ {emb_len})")
            else:
                print(f"• Gemini Embedding: [FAIL] HTTP Error {r.status_code}: {r.text}")
        except Exception as e:
            print(f"• Gemini Embedding: [FAIL] เกิดข้อผิดพลาด: {e}")
            
    # 4.2 ตรวจสอบ Pinecone
    pinecone_key = os.getenv('PINECONE_API_KEY', '')
    if not pinecone_key:
        print("• Pinecone API Key: [FAIL] ไม่พบใน .env (PINECONE_API_KEY)")
    else:
        print(f"• Pinecone API Key: [PASS] พร้อมใช้งาน (ขึ้นต้นด้วย {pinecone_key[:6]}...)")
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=pinecone_key)
            index_list = [i.name for i in pc.list_indexes()]
            if 'rice-knowledge' in index_list:
                print("• Pinecone Index  : [PASS] พบดัชนี 'rice-knowledge' บนระบบ Cloud พร้อมค้นหา")
            else:
                print(f"• Pinecone Index  : [WARN] ไม่พบดัชนี 'rice-knowledge' (พบดัชนี: {index_list})")
        except Exception as e:
            print(f"• Pinecone Client : [FAIL] เกิดข้อผิดพลาด: {e}")

    # 4.3 ตรวจสอบ PyThaiNLP
    try:
        from pythainlp.tokenize import word_tokenize
        tokens = word_tokenize("โรคขอบใบแห้งในนาข้าว", engine="newmm")
        print(f"• PyThaiNLP Token : [PASS] ตัดคำสำเร็จ: {tokens}")
    except Exception as e:
        print(f"• PyThaiNLP       : [FAIL] ไม่สามารถตัดคำได้: {e}")

def main():
    print("=" * 60)
    print("  Rice Disease LINE Chatbot — Jetson AGX Orin Deployment Validator")
    print("=" * 60)
    check_system_info()
    check_cuda_and_pytorch()
    check_model_benchmark()
    check_cloud_rag_apis()
    print_section("สรุปผลการตรวจสอบ")
    print("หากทุกหัวข้อขึ้นเครื่องหมาย [PASS] ท่านสามารถรัน FastAPI และ n8n บน Jetson AGX Orin ได้ทันทีครับ!")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    main()
