import os
import glob
import sys
import argparse
from datetime import datetime
from ultralytics import RTDETR

# Champion Model v5 Benchmark Metrics (on Test Set, from docs/model_evaluation/rtdetr_v5_model_evaluation_report.md)
V5_BENCHMARK = {
    "bacterial_leaf_blight": 0.695,
    "brown_spot": 0.799,
    "leaf_blast": 0.776,
    "leaf_scald": 0.545,
    "sheath_blight": 0.713,
    "all": 0.7056,
    "map": 0.3554,
    "precision": 0.7351,
    "recall": 0.7017
}

def get_target_model_path(specific_run=None):
    # กรณีผู้ใช้ระบุโฟลเดอร์หรือไฟล์ weights มาโดยตรง
    if specific_run:
        if os.path.isfile(specific_run):
            return specific_run
        # ตรวจสอบว่ามี best.pt อยู่โดยตรงหรือไม่
        direct_pt = os.path.join(specific_run, 'best.pt')
        if os.path.exists(direct_pt):
            return direct_pt
        # ตรวจสอบว่ามี weights/best.pt หรือไม่
        target_pt = os.path.join(specific_run, 'weights', 'best.pt')
        if os.path.exists(target_pt):
            return target_pt
        # ค้นหา best.pt ทั้งหมดใน subfolder ของ specific_run
        inner_pt = glob.glob(os.path.join(specific_run, "**/best.pt"), recursive=True)
        if inner_pt:
            inner_pt.sort(key=os.path.getmtime, reverse=True)
            return inner_pt[0]

    # ค้นหาไฟล์ best.pt ทั้งหมดในโฟลเดอร์การทดลองนี้
    best_pt_files = glob.glob("**/weights/best.pt", recursive=True)
    if not best_pt_files:
        best_pt_files = glob.glob("**/best.pt", recursive=True)
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
    
    # Resolve run path before directory change if given
    if args.run:
        if os.path.isabs(args.run):
            resolved_run = args.run
        elif os.path.exists(args.run):
            resolved_run = os.path.abspath(args.run)
        else:
            resolved_run = os.path.join(current_dir, args.run)
    else:
        resolved_run = None

    os.chdir(current_dir)
    weights_path = get_target_model_path(resolved_run)

    if not weights_path or not os.path.exists(weights_path):
        print("Error: ไม่พบไฟล์ best.pt ในระบบ (กรุณาตรวจสอบว่าการเทรนเสร็จสิ้นเรียบร้อยแล้ว)")
        return

    # คำนวณโฟลเดอร์รันหลักเพื่อเก็บผลลัพธ์
    weights_parent = os.path.dirname(os.path.abspath(weights_path))
    if os.path.basename(weights_parent).lower() == 'weights':
        latest_run = os.path.dirname(weights_parent)
    else:
        latest_run = weights_parent
    
    if args.data:
        data_yaml = args.data if os.path.isabs(args.data) else os.path.join(current_dir, args.data)
    else:
        data_yaml = os.path.join(current_dir, 'dataset_segmented', 'data.yaml')
        
    if not os.path.exists(data_yaml):
        print(f"Error: ไม่พบไฟล์คอนฟิกชุดข้อมูลที่ {data_yaml}")
        return

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
    raw_names = model.names if hasattr(model, 'names') else {
        0: 'bacterial_leaf_blight',
        1: 'brown_spot',
        2: 'leaf_blast',
        3: 'leaf_scald',
        4: 'sheath_blight'
    }
    names = {int(k): str(v) for k, v in raw_names.items()}

    class_map50 = {}
    ap_classes = getattr(metrics.box, 'ap_class_index', None)
    if ap_classes is not None and len(ap_classes) > 0:
        for i, c in enumerate(ap_classes):
            c_name = names.get(int(c), f"class_{c}")
            try:
                class_map50[c_name] = float(metrics.box.class_result(i)[2])
            except Exception:
                class_map50[c_name] = float(metrics.box.ap50[i])
    elif hasattr(metrics.box, 'ap50') and len(metrics.box.ap50) > 0:
        for idx, ap50 in enumerate(metrics.box.ap50):
            c_name = names.get(int(idx), f"class_{idx}")
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

    # ตารางเปรียบเทียบเชิงวิชาการกับ Champion Model v5
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
    
    map_diff = float(metrics.box.map) - V5_BENCHMARK['map']
    map_sign = "+" if map_diff >= 0 else ""
    comp_lines.append(f"{'OVERALL (mAP@0.5:0.95)':<25} | {V5_BENCHMARK['map']*100:>10.2f}% | {metrics.box.map*100:>10.2f}% | {map_sign}{map_diff*100:.2f}%")
    
    p_diff = float(metrics.box.mp) - V5_BENCHMARK['precision']
    p_sign = "+" if p_diff >= 0 else ""
    comp_lines.append(f"{'OVERALL Precision':<25} | {V5_BENCHMARK['precision']*100:>10.2f}% | {metrics.box.mp*100:>10.2f}% | {p_sign}{p_diff*100:.2f}%")
    
    r_diff = float(metrics.box.mr) - V5_BENCHMARK['recall']
    r_sign = "+" if r_diff >= 0 else ""
    comp_lines.append(f"{'OVERALL Recall':<25} | {V5_BENCHMARK['recall']*100:>10.2f}% | {metrics.box.mr*100:>10.2f}% | {r_sign}{r_diff*100:.2f}%")
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
    try:
        path = model.export(format='onnx', imgsz=640, half=True, simplify=True)
        print(f"Export เสร็จสมบูรณ์! ไฟล์ ONNX ถูกบันทึกที่: {path}")
    except Exception as e:
        print(f"คำเตือน: ไม่สามารถ Export เป็น ONNX ได้อัตโนมัติ ({e})")

if __name__ == '__main__':
    main()
