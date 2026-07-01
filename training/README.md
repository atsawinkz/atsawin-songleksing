# Training — RT-DETR-L Rice Disease Detection

คู่มือการเทรนโมเดล **RT-DETR-L** สำหรับตรวจจับโรคข้าว 5 คลาส

---

## สิ่งที่ต้องเตรียม

### 1. ติดตั้ง Dependencies

```bash
pip install ultralytics
```

ตรวจสอบ CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
# ต้องได้ True
```

### 2. ตรวจสอบ Dataset

ต้องมี `data/labeled/` พร้อม (Roboflow export — YOLO format):

```
data/labeled/
├── train/images/  + train/labels/
├── valid/images/  + valid/labels/
└── test/images/   + test/labels/
```

---

## วิธีเทรน

```bash
cd training/
python train_rtdetr.py
```

> [!NOTE]
> ต้องรันจาก `training/` เท่านั้น เพราะ `dataset.yaml` ใช้ path แบบ absolute

---

## Hyperparameters

| Parameter | ค่า | หมายเหตุ |
|---|---|---|
| `model` | `rtdetr-l.pt` | Pre-trained RT-DETR-L จาก COCO |
| `epochs` | 100 | จำนวน epoch สูงสุด |
| `imgsz` | 640 | Input image size (px) |
| `batch` | 8 | Batch size — ปรับเป็น **16** ถ้า VRAM พอ |
| `lr0` | 0.0001 | Initial Learning Rate |
| `optimizer` | AdamW | Optimizer |
| `warmup_epochs` | 5 | Warmup ก่อน LR เต็ม |
| `patience` | 15 | Early stopping (หยุดถ้า mAP ไม่ดีขึ้น 15 epoch) |
| `amp` | True | Mixed Precision (fp16) — ประหยัด VRAM |
| `cos_lr` | True | Cosine LR Scheduler |
| `augment` | True | Built-in Ultralytics augmentation |
| `workers` | 4 | DataLoader workers |
| `cache` | True | Cache dataset ใน RAM |
| `device` | 0 | GPU index (0 = GPU แรก) |

### ฮาร์ดแวร์ที่ใช้เทรน

- **GPU:** NVIDIA RTX 2000 Ada Generation — 16GB VRAM
- **Deploy:** NVIDIA Jetson AGX Orin (32GB)

---

## Dataset Config (`dataset.yaml`)

```yaml
path: C:/Users/COLORFUL/Documents/rice_leaf_diseases/data/labeled
train: train/images
val:   valid/images
test:  test/images

nc: 5
names:
  0: bacterial_leaf_blight   # โรคใบขีดสีน้ำตาล
  1: brown_spot               # โรคใบจุดสีน้ำตาล
  2: leaf_blast               # โรคไหม้
  3: leaf_scald               # โรคใบไหม้แห้ง
  4: sheath_blight            # โรคกาบใบแห้ง
```

> **หมายเหตุ:** `healthy_rice_leaf` ไม่มีใน dataset นี้ เพราะไม่มีรอยโรคให้ตี Bounding Box
> ระบบจะตอบ "ใบข้าวปกติ" เมื่อ RT-DETR ตรวจจับไม่พบรอยโรค

---

## ผลลัพธ์หลังเทรน

ผลเทรนจะอยู่ที่: `training/runs/rtdetr_rice_v1/`

```
runs/rtdetr_rice_v1/
├── weights/
│   ├── best.pt     ← โมเดลที่ดีที่สุด (ใช้ Deploy)
│   └── last.pt     ← Checkpoint ล่าสุด (ใช้ Resume)
├── results.csv     ← ค่า mAP, Loss ทุก epoch
├── confusion_matrix.png
├── PR_curve.png
└── results.png
```

---

## Metrics ที่ต้องบันทึก (ตาราง 3-2 ในเล่ม)

| Metric | คำอธิบาย | เป้าหมาย |
|---|---|---|
| mAP@0.5 | Mean Average Precision (IoU ≥ 0.5) | ≥ 0.70 |
| mAP@0.5:0.95 | mAP หลาย IoU threshold | ≥ 0.50 |
| Precision | ความถูกต้องในการตรวจจับ | — |
| Recall | ความครอบคลุมในการตรวจพบ | — |
| F1-Score | จุดสมดุล Precision-Recall | — |
| Inference Time | ms/รูป บนเครื่องเทรน | — |

---

## ถ้าผลไม่ดี (mAP < 0.7)

1. เพิ่ม `epochs` หรือลด `lr0` ลง (เช่น `0.00005`)
2. เพิ่ม `batch` เป็น 16 เพื่อ gradient ที่เสถียรกว่า
3. ดู Confusion Matrix → คลาสไหน AP ต่ำ → เพิ่มรูปในคลาสนั้น + เทรนใหม่
4. ลอง Resume จาก checkpoint: เพิ่ม `resume=True` ใน `model.train()`

---

## Export โมเดลสำหรับ Deploy (Jetson AGX Orin)

หลังเทรนเสร็จ export เป็น ONNX:

```python
from ultralytics import RTDETR

model = RTDETR('runs/rtdetr_rice_v1/weights/best.pt')
model.export(format='onnx', imgsz=640, simplify=True)
```

---

## Confidence Threshold (ใช้ใน n8n Workflow)

| Confidence | การตอบกลับ |
|---|---|
| > 0.5 | แสดงชื่อโรค + สาเหตุ + วิธีรักษา |
| 0.3 – 0.5 | "ไม่แน่ใจ กรุณาถ่ายภาพใหม่ให้ชัดขึ้น" |
| < 0.3 | "ไม่สามารถระบุรอยโรคได้ กรุณาลองใหม่" |
| ไม่พบ detection | "ใบข้าวปกติ ไม่พบรอยโรค" |
