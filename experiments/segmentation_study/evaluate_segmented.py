import os
import glob
import sys
import argparse
from datetime import datetime
from ultralytics import RTDETR

# Champion Model v5 Benchmark Metrics (on Test Set)
V5_BENCHMARK = {
    "bacterial_leaf_blight": 0.695,
    "brown_spot": 0.461,
    "leaf_blast": 0.655,
    "leaf_scald": 0.848,
    "sheath_blight": 0.871,
    "all": 0.706,
    "map": 0.3554,
    "precision": 0.7351,
    "recall": 0.7017
}

def get_target_model_path(specific_run=None):
    # กรณีผู้ใช้ระบุโฟลเดอร์หรือไฟล์ weights มาโดยตรง
    if specific_run:
        if os.path.isfile(specific_run):
            return specific_run
        target_pt = os.path.join(specific_run, 'weights', 'best.pt')
        if os.path.exists(target_pt):
            return target_pt

    # ค้นหาไฟล์ best.pt ทั้งหมดในโฟลเดอร์การทดลองนี้
    best_pt_files = glob.glob("**/weights/best.pt", recursive=True)
    if not best_pt_files:
        return None

    # เรียงลำดับไฟล์ตามเวลาที่บันทึกล่าสุด (Timestamp ใหม่ที่สุด)
    best_pt_files.sort(key=os.path.getmtime, reverse=True)
    return best_pt_files[0]

def main():
    parser = argparse.ArgumentParser(description="Evaluate RT-DETR Model on Test Set and Export ONNX")
    parser.add_argument('--run', type=str, default=None, help="Path to run folder or weights (e.g. runs/rtdetr_segmented_ablation)")
    parser.add_argument('--data', type=str, default=None, help="Path to dataset yaml (default: dataset_segmented/data.yaml)")
    parser.add_argument('--imgsz', type=int, default=800, help="Image size for evaluation (default: 800)")
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
    else:
        data_yaml = os.path.join(current_dir, 'dataset_segmented', 'data.yaml')
        
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
    metrics = model.val(data=data_yaml, split='test', imgsz=args.imgsz, project=latest_run, name='eval_test', exist_ok=True)

    # ดึงค่าผลลัพธ์รายโรค
    names = model.names if hasattr(model, 'names') else {
        0: 'bacterial_leaf_blight',
        1: 'brown_spot',
        2: 'leaf_blast',
        3: 'leaf_scald',
        4: 'sheath_blight'
    }

    class_map50 = {}
    if hasattr(metrics.box, 'ap_class_index') and len(metrics.box.ap_class_index) > 0:
        for i, c in enumerate(metrics.box.ap_class_index):
            c_name = names.get(c, f"class_{c}")
            class_map50[c_name] = float(metrics.box.class_result(i)[2])
    elif hasattr(metrics.box, 'ap50') and len(metrics.box.ap50) > 0:
        for idx, ap50 in enumerate(metrics.box.ap50):
            c_name = names.get(idx, f"class_{idx}")
            class_map50[c_name] = float(ap50)

    # ปริ้นท์และบันทึกค่า Metric หลักลงไฟล์ (รูปแบบมาตรฐานเดียวกับ evaluate_model.py)
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

    # เพิ่มตารางเปรียบเทียบกับ Champion Model v5
    comp_lines = []
    comp_lines.append("\n--- Ablation Comparison: Champion v5 vs Segmented Model ---")
    comp_lines.append(f"{'Disease Class':<25} | {'Champion v5':<12} | {'Segmented':<12} | {'Diff'}")
    comp_lines.append("-" * 65)
    for c_id in sorted(names.keys()):
        c_name = names[c_id]
        v5_score = V5_BENCHMARK.get(c_name, 0.0)
        seg_score = class_map50.get(c_name, 0.0)
        diff = seg_score - v5_score
        sign = "+" if diff >= 0 else ""
        comp_lines.append(f"{c_name:<25} | {v5_score*100:>10.2f}% | {seg_score*100:>10.2f}% | {sign}{diff*100:.2f}%")
    all_diff = float(metrics.box.map50) - V5_BENCHMARK['all']
    all_sign = "+" if all_diff >= 0 else ""
    comp_lines.append("-" * 65)
    comp_lines.append(f"{'OVERALL (mAP@0.5)':<25} | {V5_BENCHMARK['all']*100:>10.2f}% | {metrics.box.map50*100:>10.2f}% | {all_sign}{all_diff*100:.2f}%")
    comp_lines.append("=" * 65)
    comp_text = "\n".join(comp_lines)
    print(comp_text)
    
    # เซฟไฟล์ไว้ในโฟลเดอร์เทรนรอบนั้นๆ
    result_file = os.path.join(latest_run, 'test_results.txt')
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(result_text + "\n" + comp_text + "\n")

    print(f"\nบันทึกตัวเลขคะแนนลงไฟล์ {result_file} เรียบร้อยแล้ว")

    # 2. Export โมเดลเป็น ONNX สำหรับนำไปรันบน Jetson / FastAPI
    print("\n--- Exporting to ONNX for Deployment ---")
    path = model.export(format='onnx', imgsz=640, half=True, simplify=True)
    print(f"Export เสร็จสมบูรณ์! ไฟล์ ONNX ถูกบันทึกที่: {path}")

if __name__ == '__main__':
    main()
