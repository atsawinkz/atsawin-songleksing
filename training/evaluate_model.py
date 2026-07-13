from ultralytics import RTDETR
import os
import glob

def get_latest_run_folder(prefix='rtdetr_rice_v1'):
    # ค้นหาไฟล์ best.pt ทุกที่ที่อยู่ในโฟลเดอร์รันของเราแบบเจาะลึก (เผื่อ Ultralytics เซฟซ้อนโฟลเดอร์)
    search_path = f"**/{prefix}*/weights/best.pt"
    best_pt_files = glob.glob(search_path, recursive=True)
    
    if not best_pt_files:
        return None
        
    # เรียงลำดับไฟล์ตามเวลาที่ถูกอัปเดตล่าสุด (ดึงรอบที่ใหม่ที่สุด)
    best_pt_files.sort(key=os.path.getmtime, reverse=True)
    
    # โฟลเดอร์รันจะอยู่ 2 ชั้นถัดขึ้นไปจาก best.pt
    latest_run = os.path.dirname(os.path.dirname(best_pt_files[0]))
    return latest_run

def main():
    latest_run = get_latest_run_folder()
    if not latest_run:
        print("Error: ไม่พบไฟล์ best.pt ในระบบเลย (การเทรนอาจจะยังไม่เสร็จ หรือยังไม่เคยเซฟโมเดลสำเร็จ)")
        return
        
    weights_path = os.path.join(latest_run, 'weights', 'best.pt')

    # โหลดโมเดลที่ดีที่สุดจากการเทรนรอบล่าสุด
    print(f"Loading latest model from: {weights_path}")
    model = RTDETR(weights_path)

    # 1. ประเมินผลบนชุดข้อมูลทดสอบ (Test Set)
    print("\n--- Evaluating on Test Set ---")
    metrics = model.val(data='dataset.yaml', split='test', project=latest_run, name='eval_test')

    # ปริ้นท์และบันทึกค่า Metric หลักลงไฟล์
    result_text = (
        f"--- Evaluation Results on Test Set ---\n"
        f"Model:     {weights_path}\n"
        f"mAP50-95:  {metrics.box.map:.4f}\n"
        f"mAP50:     {metrics.box.map50:.4f}\n"
        f"Precision: {metrics.box.mp:.4f}\n"
        f"Recall:    {metrics.box.mr:.4f}\n"
        f"-----------------------------------------"
    )
    print("\n" + result_text)
    
    # เซฟไฟล์ไว้ในโฟลเดอร์เทรนรอบนั้นๆ เลย จะได้ไม่ทับกัน
    result_file = os.path.join(latest_run, 'test_results.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(result_text)

    print(f"หมายเหตุ: บันทึกตัวเลขคะแนนลงไฟล์ {result_file} เรียบร้อยแล้ว")

    # 2. Export โมเดลเป็น ONNX สำหรับนำไปรันบน Jetson
    print("\n--- Exporting to ONNX for Jetson ---")
    # ใช้ dynamic=False และขนาดคงที่ 640x640 เพื่อประสิทธิภาพที่ดีที่สุดบน Jetson
    path = model.export(format='onnx', imgsz=640, half=True, simplify=True)
    print(f"Export เสร็จสมบูรณ์! ไฟล์ ONNX ถูกบันทึกที่: {path}")

if __name__ == '__main__':
    main()
