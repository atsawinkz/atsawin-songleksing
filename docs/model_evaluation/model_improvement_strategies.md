# คู่มือและแผนยุทธศาสตร์การปรับปรุงประสิทธิภาพโมเดล (Model Improvement Strategies)
**โครงการ:** ระบบเชี่ยวชาญเพื่อการวินิจฉัยโรคพืชในข้าวและให้คำแนะนำด้วยปัญญาประดิษฐ์ (Rice Disease LINE Chatbot)  
**เอกสารสำหรับ:** การยกระดับประสิทธิภาพโมเดล RT-DETR-L และระบบ RAG Pipeline  
**วันที่จัดทำ:** 17 สิงหาคม 2569  

---

## 1. บทวิเคราะห์ประสิทธิภาพและจุดคอขวดปัจจุบัน (Current Bottlenecks Analysis)

จากการประเมินผลโมเดล **RT-DETR-L (Version 1-4)** บนชุดข้อมูลทดสอบ (Test Set 354 รูป / 742 วัตถุ) สรุปคะแนนแยกตามชนิดโรคได้ดังนี้:

| ชนิดโรคข้าว (Disease Class) | จำนวนรูป (Images) | Precision | Recall | **mAP@0.5** | สถานะประสิทธิภาพ |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Brown Spot (โรคใบจุดสีน้ำตาล)** | 61 | 69.2% | 82.3% | **80.4%** | ยอดเยี่ยมมาก |
| **Leaf Blast (โรคใบไหม้)** | 57 | 76.1% | 75.3% | **77.6%** | ยอดเยี่ยมมาก |
| **Sheath Blight (โรคกาบใบแห้ง)** | 60 | 65.3% | 80.0% | **74.5%** | ดีมาก |
| **Leaf Scald (โรคใบขอบกาบแห้ง)** | 53 | 48.9% | 49.3% | **53.3%** | ปานกลาง |
| **Bacterial Leaf Blight (โรคขอบใบแห้ง)** | 64 | 22.8% | 17.8% | **10.5%** | **จุดคอขวดหลัก (Bottleneck)** |
| **ภาพรวมทุกคลาส (Overall Average)** | **354** | **56.5%** | **60.9%** | **59.3%** | **พร้อมใช้งาน (Target ถัดไป: > 70%)** |

### 1.1 สาเหตุเชิงลึกของโรคขอบใบแห้ง (Bacterial Leaf Blight - BLB):
1. **ลักษณะทางกายภาพของรอยโรค (Aspect Ratio):** แผลขอบใบแห้งจะลามเป็นแนวยาวขนานกับขอบใบข้าว ทำให้ Bounding Box มีความยาวและแคบมาก ซึ่งสถาปัตยกรรม Feature Pyramid ทั่วไปในความละเอียด 640x640 จะจับขอบเขตได้ยากกว่ารอยโรคทรงกลม/วงรี (เช่น Brown Spot)
2. **ความคล้ายคลึงระหว่างคลาส (Inter-class Similarity):** ลักษณะแผลแห้งสีฟางข้าวมีส่วนคล้ายกับ *Leaf Scald* และอาการปลายใบแห้งตามธรรมชาติ ส่งผลให้โมเดลสับสนในการจำแนก
3. **การตีกรอบ Bounding Box (Label Noise):** ในชุดข้อมูลเดิม บางรูปมีการตีกรอบคลุมทั้งใบ แทนที่จะตีกรอบเฉพาะแผลขอบใบ

---

## 2. ยุทธศาสตร์ที่ 1: การจูน Hyperparameters & Retraining (Quick Wins บน RTX 4090)
> **ข้อดี:** ไม่ต้องเสียเวลาแก้ Dataset สามารถรันเทรนใหม่ได้ทันที และใช้ศักยภาพของฮาร์ดแวร์ RTX 4090 (24GB VRAM) ได้คุ้มค่าที่สุด

### 2.1 เพิ่มความละเอียดของภาพอินพุต (`imgsz=800` หรือ `1024`)
- **หลักการ:** การเพิ่มขนาดภาพจาก `640x640` เป็น `800x800` หรือ `1024x1024` จะช่วยเพิ่มความหนาแน่นของพิกเซลบริเวณขอบใบข้าว ทำให้ Transformer Decoder สกัด Feature รอยโรคริ้วยาวและจุดแผลเล็กๆ ได้ชัดเจนขึ้นอย่างมหาศาล
- **การใช้ทรัพยากร:** RTX 4090 มี VRAM 24GB รองรับ `imgsz=800, batch=16` หรือ `imgsz=1024, batch=8` ได้อย่างราบรื่น

### 2.2 ปรับ Learning Rate Scheduler เป็น Cosine Annealing
- **ตั้งค่า:** `cos_lr=True`, `lr0=0.0005`, `lrf=0.01`, `warmup_epochs=5`
- **ผลลัพธ์:** ช่วยให้โมเดลค่อยๆ ปรับค่าน้ำหนักอย่างนุ่มนวลในช่วงต้น (Warmup) และค่อยๆ ลด Learning Rate แบบเส้นโค้ง Cosine ป้องกันไม่ให้ตกหลุม Local Minima

### 2.3 ขยาย Epochs และเพิ่มค่าความอดทน (Patience)
- **ตั้งค่า:** `epochs=100`, `patience=25`
- **ผลลัพธ์:** ในรอบก่อนหน้า ระบบตัดจบที่ Epoch 45 (Best อยู่ที่ Epoch 30) การเพิ่มเป็น 100 รอบจะเปิดโอกาสให้โมเดลได้เรียนรู้รอยโรคยากๆ เพิ่มเติม

### 2.4 เพิ่มการทำ Data Augmentation เฉพาะทาง
- **เปิดใช้ Mixup & Mosaic อย่างเหมาะสม:** `mosaic=0.5`, `mixup=0.1` เพื่อจำลองกรณีใบข้าวที่มีรอยโรคซ้อนทับกันหลายโรค
- **ปรับ Color Jitter:** เพิ่ม `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4` เพื่อจำลองสภาพแสงแดดและร่มเงาในแปลงนาจริง

---

## 3. ยุทธศาสตร์ที่ 2: Data-Centric Enhancement (การปรับปรุงคุณภาพข้อมูลภาพ)
> **ข้อดี:** เพิ่มความแม่นยำที่แก่นของปัญหา โดยตรงกับหลักการ Data-Centric AI

1. **Re-labeling & Bounding Box Tightening (ใน Roboflow):**
   - ตรวจสอบภาพในคลาส *Bacterial Leaf Blight* และ *Leaf Scald*
   - ปรับกรอบ Bounding Box ให้กระชับเฉพาะบริเวณที่มีแผลติดเชื้อจริง ตัดส่วนใบเขียวที่ไม่มีอาการออก
2. **Hard Example Mining:**
   - ดึงภาพใน Test Set ที่โมเดลทำนายผิดพลาด (False Negative / False Positive) นำมาเพิ่มการหมุน (Rotation), ปรับความสว่าง (Brightness) และใส่เพิ่มเข้าไปในชุด Train Set

---

## 4. ยุทธศาสตร์ที่ 3: Inference & Threshold Optimization (การปรับจูนตอนใช้งานจริง)

1. **Per-Class Confidence Thresholding (เกณฑ์ความมั่นใจแยกรายโรค):**
   - โรคที่โมเดลแม่นยำสูง (*Brown Spot, Leaf Blast*): ตั้ง Threshold ไว้ที่ **`0.50`**
   - โรคที่รอยโรคยาวและจับยาก (*Bacterial Leaf Blight*): ปรับเกณฑ์เฉพาะคลาสลงมาที่ **`0.35`** เพื่อเพิ่มอัตราการตรวจจับ (Recall)
2. **Test-Time Augmentation (TTA):**
   - สั่ง `model.predict(source, augment=True)` ตอน Inference เพื่อให้โมเดลทำนายภาพแบบสลับด้าน/ปรับขนาดซ้ำ แล้วรวมผลลัพธ์ (Ensemble) เพิ่มความแม่นยำขึ้น 2-3% ทันที

---

## 5. ยุทธศาสตร์ที่ 4: RAG Pipeline Optimization (ยกระดับการตอบคำถามวิชาการ)

1. **เพิ่ม Cross-Encoder Reranker (เช่น `bge-reranker-base` / Cohere Rerank):**
   - นำ Chunks ที่ค้นหาได้จาก Hybrid Search (BM25 + Pinecone) จำนวน 8 ชิ้น มาผ่าน Reranker เพื่อคัดกรอง 3 ชิ้นที่ดีที่สุดส่งต่อให้ Gemini ช่วยลด Noise และเพิ่ม Context Precision > 95%
2. **Context-Aware Prompt Refinement:**
   - เสริม Prompt ให้แยกแยะอัตราส่วนการใช้สารเคมีตามระยะการเจริญเติบโตของต้นข้าว (ระยะกล้า, ระยะแตกกอ, ระยะออกรวง)

---

## 6. ตารางเปรียบเทียบผลลัพธ์ที่คาดการณ์ (Expected Outcomes Roadmap)

| ระยะการปรับปรุง (Phase) | การดำเนินการหลัก | mAP@0.5 ปัจจุบัน | **mAP@0.5 ที่คาดหวัง** | ความพร้อมในการทำ |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline (ปัจจุบัน)** | RT-DETR-L (`imgsz=640, epochs=45`) | **59.3%** | 59.3% | สำเร็จแล้ว |
| **Level 1: Hyperparameter Tuning** | `imgsz=800` + Cosine LR + `epochs=100` บน RTX 4090 | 59.3% | **68.0% - 74.0%** | **แนะนำให้ทำทันที** |
| **Level 2: Data-Centric AI** | Clean Label BLB บน Roboflow + TTA | ~70.0% | **75.0% - 82.0%** | ทำเพิ่มเติมเมื่อต้องการคะแนนสูงสุด |
| **Level 3: Full-Stack Optimization** | Level 1 + Level 2 + Per-Class Thresholds + Reranker | ~78.0% | **> 85.0%** | ระดับงานวิจัยระดับสูง (SOTA) |

---

## 7. ตัวอย่างคำสั่งและสคริปต์สำหรับ Retrain บน RTX 4090

```python
# ตัวอย่างการตั้งค่าเทรนรอบปรับปรุง (v2) ใน training/train_rtdetr_v2.py
from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")

results = model.train(
    data="training/dataset.yaml",
    epochs=100,
    imgsz=800,              # เพิ่มความละเอียดภาพเป็น 800x800 px
    batch=16,               # ปรับตาม VRAM 24GB ของ RTX 4090
    workers=16,
    optimizer="AdamW",
    lr0=0.0005,
    lrf=0.01,
    cos_lr=True,            # Cosine Learning Rate Scheduler
    warmup_epochs=5,
    mosaic=0.5,             # Augmentation
    mixup=0.1,
    patience=25,            # ความอดทนสำหรับ Early Stopping
    amp=True,               # FP16 Automatic Mixed Precision
    project="runs/detect",
    name="rtdetr_rice_v2_800",
    save=True
)
```
