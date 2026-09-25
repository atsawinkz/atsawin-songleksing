"""
Batch Rice Leaf Segmentation using Meta's Segment Anything Model (SAM / MobileSAM)
Guided by YOLO Bounding Box Prompts from 'data/labeled_datacentric-blb'
Designed to be executed on NVIDIA GeForce RTX 4090 (24GB VRAM)
"""
import os
import shutil
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import SAM

def create_dataset_yaml(dest_dir: Path):
    yaml_content = f"""# Dataset config for segmented rice leaves (SAM Ablation Study)
path: {dest_dir.resolve().as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 5
names:
  0: bacterial_leaf_blight
  1: brown_spot
  2: leaf_blast
  3: leaf_scald
  4: sheath_blight
"""
    yaml_path = dest_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"Created dataset config at: {yaml_path}")

def parse_yolo_bboxes(label_path: Path, img_w: int, img_h: int):
    """Convert YOLO format (class, cx, cy, w, h) to pixel [x1, y1, x2, y2]"""
    if not label_path.exists():
        return []
    
    bboxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cx, cy, w, h = map(float, parts[1:5])
                x1 = max(0, int((cx - w / 2) * img_w))
                y1 = max(0, int((cy - h / 2) * img_h))
                x2 = min(img_w, int((cx + w / 2) * img_w))
                y2 = min(img_h, int((cy + h / 2) * img_h))
                if x2 > x1 and y2 > y1:
                    bboxes.append([x1, y1, x2, y2])
    return bboxes

def find_prominent_vegetation_bbox(img_bgr, min_ratio: float = 0.10):
    """
    Locates the prominent green rice plant/leaf in images without bounding boxes (e.g. Healthy leaves).
    Returns [x1, y1, x2, y2] bounding box or None if no clear vegetation contour found.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Target green rice foliage hue: 25 to 85, saturation > 35, value > 35
    green_mask = cv2.inRange(hsv, (25, 35, 35), (85, 255, 255))
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        if (bw * bh) / (w * h) >= min_ratio:
            return [x, y, x + bw, y + bh]
    return None

def process_split(split_name: str, src_split_dir: Path, dest_split_dir: Path, model, device: str):
    src_images_dir = src_split_dir / "images"
    src_labels_dir = src_split_dir / "labels"

    dest_images_dir = dest_split_dir / "images"
    dest_labels_dir = dest_split_dir / "labels"

    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_labels_dir.mkdir(parents=True, exist_ok=True)

    # Deduplicate image files (prevents duplicate matches on case-insensitive filesystems)
    image_files = sorted(list(set(src_images_dir.glob("*.[jJ][pP][gG]")) | 
                              set(src_images_dir.glob("*.[jJ][pP][eE][gG]")) | 
                              set(src_images_dir.glob("*.[pP][nN][gG]"))))

    total_images = len(image_files)
    print(f"\nProcessing split [{split_name}]: {total_images} unique images found.")

    skipped = 0
    processed = 0

    pbar = tqdm(image_files, desc=f"SAM-{split_name.upper()}", unit="img")
    with torch.no_grad():
        for img_path in pbar:
            out_img_path = dest_images_dir / f"{img_path.stem}.jpg"
            src_label_path = src_labels_dir / f"{img_path.stem}.txt"
            dest_label_path = dest_labels_dir / f"{img_path.stem}.txt"

            # 1. Copy label file if exists
            if src_label_path.exists() and not dest_label_path.exists():
                shutil.copy2(src_label_path, dest_label_path)

            # 2. Skip if already processed
            if out_img_path.exists():
                skipped += 1
                continue

            try:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                h, w = img_bgr.shape[:2]

                bboxes = parse_yolo_bboxes(src_label_path, w, h)

                # Segment with SAM using Enclosing BBox or Point prompts to capture the whole leaf
                if bboxes:
                    min_x = max(0, min(b[0] for b in bboxes) - 10)
                    min_y = max(0, min(b[1] for b in bboxes) - 10)
                    max_x = min(w, max(b[2] for b in bboxes) + 10)
                    max_y = min(h, max(b[3] for b in bboxes) + 10)
                    enc_box = [min_x, min_y, max_x, max_y]

                    box_area_ratio = ((max_x - min_x) * (max_y - min_y)) / (w * h)
                    if box_area_ratio < 0.03:
                        # Single tiny spot (< 3% of image): prompt with point at lesion center to capture the leaf body
                        cx = (min_x + max_x) // 2
                        cy = (min_y + max_y) // 2
                        results = model(img_bgr, points=[[cx, cy]], labels=[1], device=device, verbose=False)
                    else:
                        # Enclosing bounding box covering all lesions across the leaf
                        results = model(img_bgr, bboxes=[enc_box], device=device, verbose=False)
                else:
                    # For images without disease bboxes (Healthy rice leaves):
                    # Locate prominent green foliage to avoid mud/gap center points
                    veg_bbox = find_prominent_vegetation_bbox(img_bgr)
                    if veg_bbox:
                        results = model(img_bgr, bboxes=[veg_bbox], device=device, verbose=False)
                    else:
                        # Fallback to image center point
                        center_point = [[w // 2, h // 2]]
                        results = model(img_bgr, points=center_point, labels=[1], device=device, verbose=False)

                # Combine predicted masks
                final_mask = np.zeros((h, w), dtype=bool)
                if results and len(results) > 0 and results[0].masks is not None:
                    masks_data = results[0].masks.data.cpu().numpy() # shape (N, H, W)
                    for m in masks_data:
                        # Resize mask if needed
                        if m.shape != (h, w):
                            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                        final_mask = np.logical_or(final_mask, m > 0.5)

                # Safety Guardrail: if mask captures < 5% of image, fallback to original
                mask_ratio = np.count_nonzero(final_mask) / (h * w)
                if mask_ratio < 0.05:
                    # Keep original image to avoid blank black-screen corruption
                    cv2.imwrite(str(out_img_path), img_bgr)
                else:
                    # Composite onto pure black background
                    segmented_img = np.zeros_like(img_bgr)
                    segmented_img[final_mask] = img_bgr[final_mask]
                    cv2.imwrite(str(out_img_path), segmented_img)

                processed += 1
            except Exception as e:
                tqdm.write(f"Error processing {img_path.name}: {e}")

    print(f"[{split_name}] Completed! Processed: {processed}, Skipped: {skipped}, Total: {total_images}")

def main():
    parser = argparse.ArgumentParser(description="Batch segment dataset using SAM for Ablation Study")
    parser.add_argument("--src-dir", type=str, default="../../data/labeled_datacentric-blb",
                        help="Path to source dataset (default: data/labeled_datacentric-blb)")
    parser.add_argument("--dest-dir", type=str, default="dataset_segmented",
                        help="Path to output dataset (default: dataset_segmented)")
    parser.add_argument("--model", type=str, default="mobile_sam.pt",
                        choices=["mobile_sam.pt", "sam_b.pt", "sam_l.pt"],
                        help="SAM model variant (default: mobile_sam.pt for high speed & low VRAM)")
    parser.add_argument("--splits", type=str, default="all",
                        choices=["all", "test", "valid", "train"],
                        help="Which split to process (default: all)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    src_dir = (script_dir / args.src_dir).resolve()
    dest_dir = (script_dir / args.dest_dir).resolve()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Meta Segment Anything Model (SAM) Batch Pipeline (Step 1)")
    print("=" * 60)
    print(f"Model Variant       : {args.model}")
    print(f"Device              : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Source Dataset      : {src_dir}")
    print(f"Destination Dataset : {dest_dir}")
    print(f"Target Splits       : {args.splits}")
    print("=" * 60)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {src_dir}")

    # Load SAM model via Ultralytics (downloads automatically on first run)
    print(f"Loading SAM model [{args.model}]...")
    model = SAM(args.model)
    print("SAM model ready.")

    splits_to_process = ["test", "valid", "train"] if args.splits == "all" else [args.splits]

    start_time = time.time()
    for split in splits_to_process:
        src_split_dir = src_dir / split
        dest_split_dir = dest_dir / split
        if not src_split_dir.exists():
            print(f"Warning: Split {src_split_dir} does not exist, skipping.")
            continue
        process_split(split, src_split_dir, dest_split_dir, model, device)

    # Generate data.yaml
    create_dataset_yaml(dest_dir)

    elapsed = time.time() - start_time
    print("=" * 60)
    print(f"SAM Batch Segmentation completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)!")
    print(f"Dataset ready at: {dest_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
