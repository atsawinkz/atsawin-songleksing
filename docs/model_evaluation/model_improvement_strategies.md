# คู่มือและแผนยุทธศาสตร์การปรับปรุงประสิทธิภาพโมเดล (Model Improvement Strategies)
**โครงการ:** ระบบเชี่ยวชาญเพื่อการวินิจฉัยโรคพืชในข้าวและให้คำแนะนำด้วยปัญญาประดิษฐ์ (Rice Disease LINE Chatbot)  
**เอกสารสำหรับ:** การยกระดับประสิทธิภาพโมเดล RT-DETR-L และระบบ RAG Pipeline  
**วันที่ปรับปรุงล่าสุด:** 25 สิงหาคม 2569  

---

## 1. วิวัฒนาการและการปรับแต่งโมเดลในแต่ละเวอร์ชัน (Model Evolution: v1 -> v2 -> v3 -> v4)

เพื่อยกระดับความแม่นยำของโมเดล RT-DETR-L สู่มาตรฐานสูงสุดระดับงานวิจัยสากล (State-of-the-Art) โครงการได้แบ่งการทดลองและปรับปรุงออกเป็น 4 เวอร์ชันอย่างเป็นระบบ:

### 1.1 เวอร์ชัน 1: โมเดลตั้งต้น (Version 1 - Baseline)
- **แนวคิด:** ฝึกสอนโมเดล Pre-trained RT-DETR-L (จาก COCO) ด้วยค่ามาตรฐานพื้นฐาน
- **การตั้งค่า:** `imgsz=640`, `batch=16`, `lr0=0.0001`, `optimizer='AdamW'`, `patience=15`, `epochs=100` (ตัดจบที่ Epoch 45 / Best Epoch 30)
- **ผลลัพธ์ (Test Set):** mAP50 = **59.3%**, Precision = **56.5%**, Recall = **60.9%**
- **ข้อค้นพบ:** Brown Spot (80.4%), Leaf Blast (77.6%), Sheath Blight (74.5%) ทำได้ดีมาก แต่ Bacterial Leaf Blight (BLB) อยู่ที่ 10.5% เนื่องจากเป็นแผลยาวแคบ

---

### 1.2 เวอร์ชัน 2: ปรับจูน Hyperparameters และ Cosine Scheduler (Version 2 - Optimized Hyperparameters)
- **แนวคิด:** ขยายรอบการเรียนรู้ ใช้ Cosine Learning Rate และเปิดใช้ Multi-scale Augmentation (Mosaic & Mixup)
- **การตั้งค่า:** `imgsz=640`, `lr0=0.0005`, `lrf=0.01`, `cos_lr=True`, `warmup_epochs=5`, `patience=25`, `mosaic=0.5`, `mixup=0.1` (เทรน 88 Epochs / Best Epoch 63)
- **ผลลัพธ์ (Test Set):** mAP50 = **58.3%**, Precision = **58.5% (+2.0%)**, Recall = **60.5%**
- **ข้อค้นพบสำคัญ:**
  1. **Precision สูงขึ้นชัดเจน:** ช่วยลดอัตราการแจ้งเตือนผิดพลาด (False Positive) ในระบบใช้งานจริง
  2. **Leaf Blast พุ่งแตะ 81.8% (+4.2%)** และ **Sheath Blight ขึ้นเป็น 75.8% (Precision พุ่งจาก 65.3% เป็น 74.2%)**
  3. **ข้อจำกัด:** `mosaic=0.5` บีบอัดริ้วแผลแคบยาวของ BLB และ Leaf Scald ให้เล็กลงเหลือเพียง 1-3 พิกเซล ทำให้คะแนนสองโรคนี้ลดลง

---

### 1.3 เวอร์ชัน 3: รักษาสัณฐานวิทยาแผลแคบยาวและ Fine-tune (Version 3 - Morphological Preservation & Champion Model)
- **แนวคิด:** แก้ปัญหาการบีบอัดพิกเซลของแผลยาวแคบ พร้อมต่อยอดความฉลาดจาก Best Weights ของ v2
- **การตั้งค่า:**
  1. **ปิด Mosaic และ Mixup (`mosaic=0.0`, `mixup=0.0`):** รักษาความละเอียดและความกว้างของริ้วรอยโรค BLB และ Leaf Scald ไว้ 100%
  2. **Fine-tune จาก Best Weights เดิม:** โหลดน้ำหนักที่ดีที่สุดจาก v2 มาเทรนต่อโดยไม่ต้องเริ่มจาก COCO ศูนย์ใหม่
  3. **ลด Learning Rate (`lr0=0.0002`):** ปรับจูนอย่างนุ่มนวลเพื่อรักษาความจำของโรคที่เก่งอยู่แล้ว
  4. **ใช้ Geometric Augmentation ที่เหมาะกับใบข้าว:** พลิกภาพ (`fliplr=0.5`, `flipud=0.5`), หมุนภาพ (`degrees=15.0`), สเกลเบาๆ (`scale=0.2`), และ Color Jitter (`hsv`)
- **ผลลัพธ์ (Test Set):** **mAP50 = 60.2% (New Record)**, Precision = 57.7%, **Recall = 61.9% (Highest)**, **Leaf Blast พุ่งทะลุ 86.9%**, Leaf Scald ฟื้นตัวเป็น 50.8%

---

### 1.4 เวอร์ชัน 4: ยกระดับความละเอียดสูงและเร่งความเร็วฮาร์ดแวร์ (Version 4 - High-Resolution 800px & SOTA Optimization)
- **แนวคิด:** ทลายขีดจำกัดความละเอียดภาพ 640px เพื่อเพิ่มพื้นที่พิกเซลสำหรับรอยโรคริ้วยาวแคบ พร้อมเปิดใช้การเร่งความเร็วฮาร์ดแวร์ระดับ GPU Tensor Cores + RAM 64GB
- **การตั้งค่าสำคัญ:**
  1. **เพิ่มความละเอียดภาพเป็น 800x800 (`imgsz=800`):** เพิ่มความหนาแน่นพิกเซลขึ้น **+56%** ช่วยให้ Transformer Feature Pyramid จับขอบเขตริ้วแผลแคบของโรคขอบใบแห้งและปลายใบแห้งได้คมชัดสูงสุด
  2. **Fine-tune ต่อยอดจาก Champion Weights ของ v3:** โหลดค่าน้ำหนักจาก v3 มาเทรนต่อด้วย `lr0=0.0001` เพื่อรักษาฐานความแม่นยำของ Leaf Blast 86.9% และ Brown Spot 80.2% ไว้
  3. **เร่งความเร็วระดับฮาร์ดแวร์ (Hardware Acceleration):**
     - `cache='ram'`: โหลด Dataset ทั้งหมดเก็บใน RAM 64GB ตัดปัญหาคอขวด Disk I/O 100%
     - `workers=8`: จับคู่กับ 8 Performance Cores (P-Cores) ของ Intel Core i9-14900K
     - `torch.backends.cudnn.benchmark = True`: บังคับเลือก CUDA Kernel ที่เร็วที่สุดสำหรับขนาดภาพคงที่
  4. **ปรับถ่วงน้ำหนัก Box Loss (`box=8.5`):** บังคับให้โมเดลโฟกัสการจัดตำแหน่งกรอบ Bounding Box ให้กระชับแนบสนิทกับรอยโรคมากยิ่งขึ้น

---

## 2. ตารางเปรียบเทียบพารามิเตอร์การเทรนทั้ง 4 เวอร์ชัน

| พารามิเตอร์ (Parameter) | เวอร์ชัน 1 (Baseline) | เวอร์ชัน 2 (Tuned HP) | เวอร์ชัน 3 (Morphological) | **เวอร์ชัน 4 (High-Res SOTA)** | วัตถุประสงค์ทางเทคนิค |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Base Model Weights** | `rtdetr-l.pt` (COCO) | `rtdetr-l.pt` (COCO) | `best.pt` (จาก v2) | **`best.pt` (จาก v3)** | ต่อยอดความฉลาดสะสม ไม่เริ่มจากศูนย์ |
| **Input Image Size (`imgsz`)** | `640` | `640` | `640` | **`800`** | **เพิ่มความหนาแน่นพิกเซล +56% เก็บแผลแคบ** |
| **Batch Size (`batch`)** | `16` | `16` | `16` | **`16`** | ปรับขนาดเต็มประสิทธิภาพ VRAM 24GB |
| **Initial Learning Rate (`lr0`)** | `0.0001` | `0.0005` | `0.0002` | **`0.0001`** | Fine-tune นุ่มนวลบนสเกลภาพใหม่ |
| **Learning Rate Scheduler** | Linear Decay | Cosine Annealing | Cosine Annealing | **Cosine Annealing** | ปรับลดตามเส้นโค้ง Cosine ไม่ตก Local Minima |
| **Warmup Epochs** | 3 | 5 | 3 | **3** | วอร์มอัพค่าน้ำหนักช่วงต้น |
| **Early Stopping Patience** | 15 | 25 | 25 | **25** | ให้โอกาสโมเดลเรียนรู้เคสยากๆ เต็มที่ |
| **Mosaic / Mixup** | Default | `0.5` / `0.1` | `0.0` / `0.0` (ปิด) | **`0.0` / `0.0` (ปิด)** | รักษาสัณฐานวิทยาแผลริ้วยาวต่อเนื่อง |
| **Geometric Augmentation** | Default | Default | Flips + Rotation 15° | **Flips + Rotation 15°** | จำลองสภาพใบข้าวลู่ลมในแปลงนา |
| **RAM Caching (`cache`)** | `False` | `False` | `False` | **`'ram'`** | แคชภาพลง 64GB RAM เร่งความเร็ว GPU |
| **Dataloader Workers** | `16` | `16` | `16` | **`8`** | ลด Context Switching บน 8 P-Cores |
| **Box Loss Gain (`box`)** | `7.5` (Default) | `7.5` | `7.5` | **`8.5`** | เพิ่มความคมชัดในการตีกรอบพิกัดแผล |
| **cuDNN Benchmark** | Default | Default | Default | **`True`** | เร่งความเร็ว CUDA Kernel ระดับฮาร์ดแวร์ |

---

## 3. ความเข้าใจเรื่องการเพิ่มพูนข้อมูล (Offline vs Online Augmentation)

ในกระบวนการพัฒนาโมเดล มีการเพิ่มพูนข้อมูล 2 ระดับที่ทำหน้าที่เสริมซึ่งกันและกัน:

```
[ภาพถ่ายต้นฉบับ 3,829 รูป]
            │
            ▼  (ขั้นตอนที่ 1: Offline Augmentation ใน Roboflow)
[Dataset Labeled 7,416 รูปบนดิสก์]
            │
            ▼  (ขั้นตอนที่ 2: Online Augmentation ใน PyTorch Dataloader ตอนเทรน)
[ภาพไดนามิกสุ่มมุมมอง/แสงใหม่ทุก Epoch สู่ GPU RTX 4090]
```

1. **Offline Augmentation (ใน Dataset / Roboflow):**
   - ทำครั้งเดียวก่อนเริ่มเทรน เพื่อเพิ่มจำนวนรูปภาพพื้นฐานจาก 3,829 รูป เป็น 7,416 รูป บันทึกเป็นไฟล์ตายตัวบนฮาร์ดดิสก์
2. **Online Augmentation (ในสคริปต์ `train_rtdetr_v4.py`):**
   - ทำงานแบบ Real-time ในหน่วยความจำขณะดึงภาพเข้า GPU ในแต่ละ Epoch ตัว Dataloader จะสุ่มองศาการหมุนและการกลับด้านใหม่ทุกรอบ ทำให้โมเดลเห็นภาพที่มีความหลากหลายไม่ซ้ำกันตลอด 100 รอบ ช่วยป้องกัน Overfitting สูงสุด

---

## 4. แผนยุทธศาสตร์ขั้นสูงในอนาคต (Advanced Future Roadmap)

### 4.1 ระดับการใช้งานจริง: Per-Class Confidence Thresholding
- ในระบบ API บน FastAPI สามารถกำหนดเกณฑ์ความมั่นใจแยกตามชนิดโรคได้:
  - โรคที่มีความแม่นยำสูงมาก (*Leaf Blast, Brown Spot, Sheath Blight*): ใช้ Threshold **`0.45`**
  - โรคที่รอยโรคยาวแคบและต้องการความครอบคลุม (*BLB, Leaf Scald*): ใช้ Threshold **`0.25`**
  - ช่วยดึงค่า Recall ของโรคขอบใบแห้งขึ้นทันทีในการวินิจฉัยผ่าน LINE Chatbot

### 4.2 ระดับข้อมูล: Data-Centric AI (Roboflow Clean-up)
- ตรวจสอบภาพคลาส Bacterial Leaf Blight ใน Roboflow และกระชับกรอบ Bounding Box ให้ครอบคลุมเฉพาะบริเวณเนื้อเยื่อที่มีแผลติดเชื้อจริง ตัดส่วนใบเขียวออกเพื่อลด Label Noise

### 4.3 ระดับระบบ RAG: Hybrid Reranker Pipeline
- นำโมเดล Cross-Encoder (เช่น `bge-reranker-base`) มาคัดกรอง Chunks จาก Pinecone และ BM25 เพื่อดัน Context Precision ของคำแนะนำภาษาไทยให้เกิน 95%

---

## 5. สรุปคำสั่งสำหรับการเทรนและประเมินผลบน RTX 4090

```bash
# 1. รันการเทรน Version 4 (High-Resolution 800px)
python3 training/train_rtdetr_v4.py

# 2. รันการประเมินผล Test Set และ Export ONNX ของ Version 4
python3 training/evaluate_model.py --run training/runs/rtdetr_rice_v4
```
