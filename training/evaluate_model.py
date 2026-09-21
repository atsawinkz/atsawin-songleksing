import os
import glob
import sys
import argparse
from datetime import datetime
from ultralytics import RTDETR

def get_target_model_path(specific_run=None):
    # กรณีผู้ใช้ระบุโฟลเดอร์หรือไฟล์ weights มาโดยตรง
    if specific_run:
        if os.path.isfile(specific_run):
            return specific_run
        target_pt = os.path.join(specific_run, 'weights', 'best.pt')
        if os.path.exists(target_pt):
            return target_pt

    # ค้นหาไฟล์ best.pt ทั้งหมดในระบบ
    best_pt_files = glob.glob("**/weights/best.pt", recursive=True)
    if not best_pt_files:
        return None

    # เรียงลำดับไฟล์ตามเวลาที่บันทึกล่าสุด (Timestamp ใหม่ที่สุด)
    best_pt_files.sort(key=os.path.getmtime, reverse=True)
    return best_pt_files[0]

def main():
    parser = argparse.ArgumentParser(description="Evaluate RT-DETR Model on Test Set and Export ONNX")
    parser.add_argument('--run', type=str, default=None, help="Path to run folder or weights (e.g. runs/rtdetr_rice_v2)")
    parser.add_argument('--data', type=str, default=None, help="Path to dataset yaml (e.g. dataset_v5.yaml)")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    weights_path = get_target_model_path(args.run)

    if not weights_path or not os.path.exists(weights_path):
        print("Error: ไม่พบไฟล์ best.pt ในระบบ (กรุณาตรวจสอบว่าการเทรนเสร็จสิ้นเรียบร้อยแล้ว)")
        return

    latest_run = os.path.dirname(os.path.dirname(os.path.abspath(weights_path)))
    
    if args.data:
        data_yaml = args.data if os.path.isabs(args.data) else os.path.join(current_dir, args.data)
    elif 'v5' in str(args.run or '') or 'v5' in str(weights_path):
        data_yaml = os.path.join(current_dir, 'dataset_v5.yaml')
    else:
        data_yaml = os.path.join(current_dir, 'dataset.yaml')
        
    mod_time = datetime.fromtimestamp(os.path.getmtime(weights_path)).strftime('%Y-%m-%d %H:%M:%S')

    print("=========================================================")
    print(f"Target Model:    {weights_path}")
    print(f"Dataset Config:  {data_yaml}")
    print(f"Saved Timestamp: {mod_time}")
    print(f"Run Folder:      {latest_run}")
    print("=========================================================")

    # โหลดโมเดลที่ดีที่สุดจากการเทรน
    model = RTDETR(weights_path)

    # 1. ประเมินผลบนชุดข้อมูลทดสอบ (Test Set)
    print("\n--- Evaluating on Test Set ---")
    metrics = model.val(data=data_yaml, split='test', project=latest_run, name='eval_test', exist_ok=True)

    # ปริ้นท์และบันทึกค่า Metric หลักลงไฟล์
    result_text = (
        f"--- Evaluation Results on Test Set ---\n"
        f"Model:     {weights_path}\n"
        f"Timestamp: {mod_time}\n"
        f"mAP50-95:  {metrics.box.map:.4f}\n"
        f"mAP50:     {metrics.box.map50:.4f}\n"
        f"Precision: {metrics.box.mp:.4f}\n"
        f"Recall:    {metrics.box.mr:.4f}\n"
        f"-----------------------------------------"
    )
    print("\n" + result_text)
    
    # เซฟไฟล์ไว้ในโฟลเดอร์เทรนรอบนั้นๆ
    result_file = os.path.join(latest_run, 'test_results.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(result_text)

    print(f"บันทึกตัวเลขคะแนนลงไฟล์ {result_file} เรียบร้อยแล้ว")

    # 2. Export โมเดลเป็น ONNX สำหรับนำไปรันบน Jetson / FastAPI
    print("\n--- Exporting to ONNX for Deployment ---")
    path = model.export(format='onnx', imgsz=640, half=True, simplify=True)
    print(f"Export เสร็จสมบูรณ์! ไฟล์ ONNX ถูกบันทึกที่: {path}")

if __name__ == '__main__':
    main()
