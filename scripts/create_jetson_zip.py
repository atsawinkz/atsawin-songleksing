import os
import sys
import zipfile

# ป้องกันปัญหา Unicode
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

zip_name = "rice_disease_jetson_deploy.zip"
root_folder_inside_zip = "rice_leaf_diseases"

files = [
    ".env",
    "api/app.py",
    "api/app_v2.py",
    "api/requirements.txt",
    "n8n/docker-compose.yml",
    "n8n/rice_bot_workflow.json",
    "scripts/bm25_stats.json",
    "scripts/check_jetson_env.py",
    "scripts/export_tensorrt_jetson.py",
    "docs/plans/line_n8n_setup_guide.md",
    "training/runs/detect/runs/rtdetr_rice_v5/weights/best.onnx",
    "training/runs/detect/runs/rtdetr_rice_v5/weights/best.pt",
]

print(f"[INFO] กำลังสร้างไฟล์ ZIP: {zip_name} ...")

with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
    for rel_path in files:
        if os.path.exists(rel_path):
            arcname = os.path.join(root_folder_inside_zip, rel_path).replace("\\", "/")
            zipf.write(rel_path, arcname=arcname)
            size_mb = os.path.getsize(rel_path) / (1024 * 1024)
            print(f"  + เพิ่มไฟล์: {rel_path} ({size_mb:.2f} MB)")
        else:
            print(f"  [WARN] ไม่พบไฟล์: {rel_path} (ข้าม)")

total_size_mb = os.path.getsize(zip_name) / (1024 * 1024)
print("\n" + "=" * 60)
print(f"[SUCCESS] บีบอัดไฟล์ ZIP สำเร็จสมบูรณ์!")
print(f"ที่อยู่ไฟล์: {os.path.abspath(zip_name)}")
print(f"ขนาดไฟล์ ZIP รวม: {total_size_mb:.2f} MB")
print("=" * 60)
