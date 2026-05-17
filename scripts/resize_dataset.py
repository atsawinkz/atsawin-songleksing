"""
Letterbox Resize Dataset
========================
Resize ทุกรูปใน data/raw/ ให้เป็น 640x640 (letterbox)
แล้วบันทึกไปที่ data/processed/ โดยไม่แตะรูปต้นฉบับ

วิธีใช้:
    python scripts/resize_dataset.py
    python scripts/resize_dataset.py --size 640 --preview 5
"""

import os
import sys
import argparse
from pathlib import Path

# Force UTF-8 encoding for stdout/stderr to prevent UnicodeEncodeError on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import cv2
import numpy as np


# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def letterbox_resize(img, target_size=640, pad_color=(114, 114, 114)):
    """
    Resize รูปแบบ letterbox — รักษา aspect ratio แล้ว pad ส่วนที่เหลือ

    Args:
        img: numpy array (BGR) จาก cv2.imread
        target_size: ขนาดเป้าหมาย (จะได้ target_size x target_size)
        pad_color: สี padding (BGR) — ค่าเริ่มต้นเป็นสีเทา

    Returns:
        resized: numpy array ขนาด (target_size, target_size, 3)
        scale: อัตราส่วนที่ใช้ย่อ/ขยาย
    """
    h, w = img.shape[:2]

    # คำนวณ scale — ย่อ/ขยายด้านที่ยาวที่สุดให้พอดี target_size
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    # คำนวณ padding
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    # Pad
    resized = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=pad_color
    )

    return resized, scale


def process_dataset(raw_dir, output_dir, target_size=640, preview_count=0):
    """
    ประมวลผลทั้ง dataset — letterbox resize ทุกรูป

    Args:
        raw_dir: โฟลเดอร์ต้นทาง (data/raw/)
        output_dir: โฟลเดอร์ปลายทาง (data/processed/)
        target_size: ขนาดเป้าหมาย
        preview_count: จำนวนรูปที่จะแสดง preview (0 = ไม่แสดง)
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    if not raw_path.exists():
        print(f"Error: ไม่พบโฟลเดอร์ {raw_path}")
        sys.exit(1)

    # หา subdirectories (แต่ละคลาส)
    classes = sorted([d for d in raw_path.iterdir() if d.is_dir()])
    if not classes:
        print(f"Error: ไม่พบโฟลเดอร์คลาสใน {raw_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"  Letterbox Resize Dataset → {target_size}x{target_size}")
    print("=" * 60)
    print(f"  Input:  {raw_path}")
    print(f"  Output: {out_path}")
    print(f"  คลาส:  {len(classes)} คลาส")
    print("=" * 60)

    total_success = 0
    total_fail = 0
    total_upscaled = 0
    total_downscaled = 0
    summary = []

    for class_dir in classes:
        class_name = class_dir.name
        class_out = out_path / class_name
        class_out.mkdir(parents=True, exist_ok=True)

        # หาไฟล์รูปภาพ
        image_files = sorted([
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ])

        success = 0
        fail = 0
        upscaled = 0
        downscaled = 0
        previewed = 0

        for img_file in image_files:
            try:
                # อ่านรูป
                img = cv2.imread(str(img_file))
                if img is None:
                    raise ValueError("cv2.imread returned None")

                h, w = img.shape[:2]
                scale = target_size / max(h, w)

                # Resize
                result, _ = letterbox_resize(img, target_size)

                # บันทึก
                out_file = class_out / img_file.name
                cv2.imwrite(str(out_file), result)

                success += 1
                if scale > 1:
                    upscaled += 1
                elif scale < 1:
                    downscaled += 1

                # Preview (แสดงหน้าต่าง)
                if preview_count > 0 and previewed < preview_count:
                    window_name = f"{class_name}: {img_file.name} ({w}x{h} -> {target_size}x{target_size})"
                    cv2.imshow(window_name, result)
                    cv2.waitKey(1500)
                    cv2.destroyWindow(window_name)
                    previewed += 1

            except Exception as e:
                fail += 1
                print(f"  ✗ {img_file.name}: {e}")

        total_success += success
        total_fail += fail
        total_upscaled += upscaled
        total_downscaled += downscaled

        status = "✓" if fail == 0 else "⚠"
        same = success - upscaled - downscaled
        print(f"  {status} {class_name:30s} | {success:4d} สำเร็จ | ↑{upscaled} ↓{downscaled} ={same}")
        summary.append({
            "class": class_name,
            "success": success,
            "fail": fail,
            "upscaled": upscaled,
            "downscaled": downscaled,
        })

    # สรุปผล
    print("=" * 60)
    print(f"  รวม: {total_success} สำเร็จ | {total_fail} ล้มเหลว")
    print(f"  Upscaled (ขยาย):   {total_upscaled} รูป")
    print(f"  Downscaled (ย่อ):  {total_downscaled} รูป")
    print(f"  ไม่เปลี่ยนขนาด:    {total_success - total_upscaled - total_downscaled} รูป")
    print(f"  Output: {out_path}")
    print("=" * 60)

    if total_fail > 0:
        print(f"\n⚠ พบ {total_fail} รูปที่ประมวลผลล้มเหลว — ตรวจสอบข้อมูลด้านบน")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Letterbox Resize ทุกรูปใน data/raw/ ให้เป็น NxN"
    )
    parser.add_argument(
        "--size", type=int, default=640,
        help="ขนาดเป้าหมาย (default: 640)"
    )
    parser.add_argument(
        "--input", type=str, default=str(RAW_DIR),
        help=f"โฟลเดอร์ input (default: {RAW_DIR})"
    )
    parser.add_argument(
        "--output", type=str, default=str(PROCESSED_DIR),
        help=f"โฟลเดอร์ output (default: {PROCESSED_DIR})"
    )
    parser.add_argument(
        "--preview", type=int, default=0,
        help="จำนวนรูปที่จะแสดง preview ต่อคลาส (default: 0)"
    )
    args = parser.parse_args()

    process_dataset(args.input, args.output, args.size, args.preview)
