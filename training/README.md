# Training — RT-DETR-L Rice Disease Detection (Phase 4)

คู่มือการเทรนโมเดล **RT-DETR-L** สำหรับตรวจจับโรคข้าว 5 คลาส

เนื่องจากการเทรนโมเดล RT-DETR-L ให้มีประสิทธิภาพสูงสุด (อัปเดตสเปคล่าสุดเป็นเครื่อง Altos BrainSphere P150 F8 พร้อม RTX 4090 24GB) จำเป็นต้องรันบนเครื่องเทรนเฉพาะทาง ไม่ใช่เครื่องที่กำลังเปิดแชทอยู่ตอนนี้ ผมจึงได้จัดเตรียมโค้ดและวิธีการให้คุณอัศวินสามารถคัดลอกไปรันบนเครื่องเทรนได้อย่างสะดวกรวดเร็วครับ

## ไฟล์ที่เตรียมไว้สำหรับการเคลื่อนย้ายไปเทรน

### [MODIFY] `dataset.yaml`
- ปรับเปลี่ยน Path ของชุดข้อมูลจากแบบเจาะจงเครื่อง (Absolute) เป็น `../data/labeled` (Relative) เพื่อให้สามารถก๊อปปี้ไปวางเครื่องไหนก็รันได้ทันที

### [NEW] `evaluate_model.py`
- สร้างสคริปต์ตัวใหม่สำหรับทำ Evaluation โดยเฉพาะ 
- สคริปต์นี้จะอ่านโมเดลที่ดีที่สุดหลังเทรนเสร็จ (`best.pt`) มาประเมินผลกับ Test Set (หาค่า mAP, Precision, Recall) 
- และจะทำการ Export ตัวโมเดลให้ออกมาเป็นนามสกุล `.onnx` แบบ `half=True` (FP16) ขนาด 640x640 เพื่อให้พร้อมนำไปรันบนกล่อง NVIDIA Jetson (Phase 6) ได้อย่างลื่นไหลที่สุด

---

## ขั้นตอนการนำไปรันบนเครื่องเทรน (Step-by-Step Guide)

### 1. คัดลอกโฟลเดอร์ไปยังเครื่องเทรน
ให้คัดลอก 2 โฟลเดอร์นี้ไปไว้บนเครื่องที่จะใช้เทรน (ให้อยู่ระดับเดียวกัน):
- โฟลเดอร์รูปภาพ: `data/labeled/`
- โฟลเดอร์โค้ดเทรน: `training/`

### 2. ตรวจสอบ Dataset
ต้องมี `data/labeled/` พร้อม (Roboflow export — YOLO format):
```text
data/labeled/
├── train/images/  + train/labels/
├── valid/images/  + valid/labels/
└── test/images/   + test/labels/
```

### 3. ติดตั้ง Dependencies และตรวจสอบ CUDA
เปิด Terminal บนเครื่องเทรนและรัน:
```bash
pip install ultralytics torch torchvision
```
ตรวจสอบ CUDA ว่าใช้งานได้หรือไม่:
```bash
python -c "import torch; print(torch.cuda.is_available())"
# ต้องได้ True
```

### 4. รันเทรนโมเดล
เข้าไปในโฟลเดอร์ `training/` แล้วรันสคริปต์เทรน (ใช้เวลาสักพักใหญ่):
```bash
cd training/
python train_rtdetr.py
```
> [!NOTE]
> ต้องรันจาก `training/` เท่านั้น เพื่อให้ relative path หา folder รูปภาพเจอ

*(ระหว่างนี้โปรแกรมจะสร้างโฟลเดอร์ `runs/rtdetr_rice_v1/` และเซฟ Checkpoint ให้อัตโนมัติ)*

### 5. รันประเมินผลและ Export ONNX
เมื่อการเทรนเสร็จสมบูรณ์ ให้รันสคริปต์ตัวใหม่ที่สร้างให้ เพื่อเก็บคะแนน Test Set และแปลงไฟล์สำหรับ Jetson:
```bash
python evaluate_model.py
```

### 6. นำไฟล์กลับมาที่โปรเจกต์หลัก
เมื่อรันจบทั้งหมด สิ่งที่คุณต้องก๊อปปี้กลับมาใส่ในเครื่องโปรเจกต์หลักของเราเพื่อใช้ใน Phase ถัดไปคือ:
1. ไฟล์ **`training/runs/rtdetr_rice_v1/weights/best.onnx`** (หัวใจหลักของบอทเรา)
2. โฟลเดอร์ **`training/runs/rtdetr_rice_v1/`** ทั้งหมด (เพื่อเอา Confusion Matrix และกราฟไปใส่ในรูปเล่มปริญญานิพนธ์)

---

## ข้อมูลอ้างอิงเชิงลึก (Reference & Hyperparameters)

### ฮาร์ดแวร์ที่ใช้เทรน
- **GPU:** NVIDIA GeForce RTX 4090 — 24GB VRAM
- **CPU:** Intel Core i9-14900K (32 Cores)
- **RAM:** 64 GB
- **Deploy:** NVIDIA Jetson AGX Orin (32GB)

### Hyperparameters ที่ใช้ใน `train_rtdetr.py`

| Parameter | ค่า | หมายเหตุ |
|---|---|---|
| `model` | `rtdetr-l.pt` | Pre-trained RT-DETR-L จาก COCO |
| `epochs` | 100 | จำนวน epoch สูงสุด |
| `imgsz` | 640 | Input image size (px) |
| `batch` | 16 | Batch size — ปรับเป็น **32** ได้เนื่องจาก VRAM 24GB |
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

### Dataset Config (`dataset.yaml`)
```yaml
path: ../data/labeled
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

### ผลลัพธ์หลังเทรน
ผลเทรนจะอยู่ที่: `training/runs/rtdetr_rice_v1/`
```text
runs/rtdetr_rice_v1/
├── weights/
│   ├── best.pt     ← โมเดลที่ดีที่สุด (ใช้ Deploy)
│   └── last.pt     ← Checkpoint ล่าสุด (ใช้ Resume)
├── results.csv     ← ค่า mAP, Loss ทุก epoch
├── confusion_matrix.png
├── PR_curve.png
└── results.png
```

### Metrics ที่ต้องบันทึก (ตาราง 3-2 ในเล่ม)
| Metric | คำอธิบาย | เป้าหมาย |
|---|---|---|
| mAP@0.5 | Mean Average Precision (IoU ≥ 0.5) | ≥ 0.70 |
| mAP@0.5:0.95 | mAP หลาย IoU threshold | ≥ 0.50 |
| Precision | ความถูกต้องในการตรวจจับ | — |
| Recall | ความครอบคลุมในการตรวจพบ | — |
| F1-Score | จุดสมดุล Precision-Recall | — |
| Inference Time | ms/รูป บนเครื่องเทรน | — |

### กรณีผลเทรนไม่เป็นไปตามเป้า (mAP < 0.7)
1. เพิ่ม `epochs` หรือลด `lr0` ลง (เช่น `0.00005`)
2. เพิ่ม `batch` เป็น 16 เพื่อ gradient ที่เสถียรกว่า
3. ดู Confusion Matrix → คลาสไหน AP ต่ำ → เพิ่มรูปในคลาสนั้น + เทรนใหม่
4. ลอง Resume จาก checkpoint: เพิ่ม `resume=True` ใน `model.train()`

### Confidence Threshold (ใช้ใน n8n Workflow)
| Confidence | การตอบกลับ |
|---|---|
| > 0.5 | แสดงชื่อโรค + สาเหตุ + วิธีรักษา |
| 0.3 – 0.5 | "ไม่แน่ใจ กรุณาถ่ายภาพใหม่ให้ชัดขึ้น" |
| < 0.3 | "ไม่สามารถระบุรอยโรคได้ กรุณาลองใหม่" |
| ไม่พบ detection | "ใบข้าวปกติ ไม่พบรอยโรค" |
