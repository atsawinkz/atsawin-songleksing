# รายงานผลการเทรนและการประเมินประสิทธิภาพโมเดล RT-DETR-L (Object Detection)
**โครงการ:** ระบบเชี่ยวชาญเพื่อการวินิจฉัยโรคพืชในข้าวและให้คำแนะนำด้วยปัญญาประดิษฐ์ (Rice Disease LINE Chatbot)  
**วันที่บันทึก:** 2 สิงหาคม 2569  
**โมเดลหลัก:** Real-Time Detection Transformer Large (RT-DETR-L)

---

## 1. บทสรุปผู้บริหาร (Executive Summary)

การเทรนโมเดล **RT-DETR-L** สำหรับตรวจจับและระบุตำแหน่งโรคใบข้าว 5 ชนิด ได้เสร็จสิ้นสมบูรณ์บนเครื่องประมวลผลประสิทธิภาพสูง (NVIDIA GeForce RTX 4090) โดยโมเดลผ่านการเรียนรู้ทั้งหมด 45 Epochs (หยุดทำงานอัตโนมัติด้วยระบบ Early Stopping ที่ Epoch 45 และเลือกสมองที่ดีที่สุดใน **Epoch 30**)

จากการทดสอบกับชุดข้อมูล **Test Set (354 รูปที่ไม่เคยเห็นมาก่อน)** โมเดลสามารถทำคะแนนความแม่นยำรวม **mAP50 ได้สูงถึง 59.3%** และ **Recall รวม 60.9%** โดยเฉพาะโรคหลักอย่าง **Brown Spot (80.4%)**, **Leaf Blast (77.6%)**, และ **Sheath Blight (74.5%)** มีความแม่นยำในระดับสูงมาก โมเดลถูกทำการส่งออก (Export) เป็นไฟล์มาตรฐาน **ONNX (FP16)** ขนาด 62.9 MB เรียบร้อยแล้ว พร้อมนำไปเชื่อมต่อเข้ากับระบบ n8n Workflow และ LINE Chatbot ใน Phase 6 ทันที

---

## 2. สภาพแวดล้อมและโครงสร้างการเทรน (Training Environment & Config)

### 2.1 สเปคฮาร์ดแวร์ที่ใช้เทรน (Training Hardware)
- **เครื่องเทรน:** Altos BrainSphere P150 F8
- **GPU:** NVIDIA GeForce RTX 4090 (24GB VRAM)
- **CPU:** Intel Core i9-14900K (32 Threads)
- **RAM:** 64 GB
- **OS:** Ubuntu 24.04 LTS

### 2.2 การตั้งค่า Hyperparameters (`train_rtdetr.py`)
| Parameter | Value | คำอธิบาย |
| :--- | :--- | :--- |
| `model` | `rtdetr-l.pt` | Pre-trained RT-DETR-L จาก COCO Dataset |
| `imgsz` | `640` | ขนาดภาพอินพุต (640x640 px) |
| `batch` | `16` | Batch Size สำหรับ RTX 4090 |
| `workers` | `16` | Dataloader Threads ปลดล็อกตามสเปค i9-14900K |
| `optimizer` | `AdamW` | Initial Learning Rate `lr0 = 0.0001` |
| `amp` | `True` | Automatic Mixed Precision (FP16) |
| `patience` | `15` | Early Stopping หยุดเมื่อไม่พัฒนาติดต่อกัน 15 รอบ |
| `cache` | `False` | โหลดจาก NVMe SSD ป้องกัน RAM Overflow |

---

## 3. ข้อมูลชุดภาพสำหรับเทรน (Dataset Distribution)

ชุดข้อมูลภาพโรคใบข้าว 5 คลาสหลัก ผ่านการทำ Letterbox Resize (640x640) และแบ่งสัดส่วนดังนี้:
- **Train Set:** 7,416 รูป (ใช้สำหรับเทรนหลัก)
- **Validation Set:** 704 รูป (ใช้สำหรับปรับค่าน้ำหนักระหว่างเทรน)
- **Test Set:** 354 รูป (ใช้สำหรับวัดผลสัมฤทธิ์รอบสุดท้าย)

---

## 4. ผลการประเมินประสิทธิภาพบนชุดข้อมูลทดสอบ (Test Set Evaluation)

การประเมินผลผ่านสคริปต์ `evaluate_model.py` บนชุดข้อมูล **Test Set (354 รูป / 742 Bounding Boxes)** ได้ผลลัพธ์ดังนี้:

### 4.1 ตารางสรุปคะแนนภาพรวม (Overall Metrics)
| Metric | Validation Set (Epoch 30) | Test Set (Unseen Data) | สถานะ / คำวิเคราะห์ |
| :--- | :---: | :---: | :--- |
| **mAP@0.5** | 52.9% | **59.3%** | สูงกว่า Validation Set แสดงว่าไม่มีอาการ Overfitting |
| **mAP@0.5:0.95** | 28.1% | **30.8%** | ค่าเฉลี่ยความแม่นยำกรอบวัตถุระดับเข้มงวด |
| **Precision** | 58.1% | **56.5%** | ความถูกต้องเมื่อโมเดลทำนายว่าเป็นโรค |
| **Recall** | 54.9% | **60.9%** | อัตราการตรวจพบโรคจากโรคจริงทั้งหมด |

### 4.2 ตารางสรุปคะแนนแยกตามชนิดโรค (Per-Class Breakdown)
| ชื่อโรค (Disease Class) | จำนวนรูป (Images) | จำนวนวัตถุ (Instances) | Precision | Recall | **mAP50** | **mAP50-95** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🤎 **Brown Spot (โรคใบจุดสีน้ำตาล)** | 61 | 215 | 69.2% | 82.3% | **80.4%** | 43.4% |
| 🍃 **Leaf Blast (โรคใบไหม้)** | 57 | 81 | 76.1% | 75.3% | **77.6%** | 38.7% |
| 🌾 **Sheath Blight (โรคกาบใบแห้ง)** | 60 | 130 | 65.3% | 80.0% | **74.5%** | 48.3% |
| 🍂 **Leaf Scald (โรคใบขอบกาบแห้ง)** | 53 | 69 | 48.9% | 49.3% | **53.3%** | 20.7% |
| ⚠️ **Bacterial Leaf Blight (โรคขอบใบแห้ง)** | 64 | 247 | 22.8% | 17.8% | **10.5%** | 2.7% |
| **รวมทุกคลาส (All Classes)** | **354** | **742** | **56.5%** | **60.9%** | **59.3%** | **30.8%** |

---

## 5. การวิเคราะห์เชิงลึกและอภิปรายผล (Deep Analysis)

1. **โรคที่มีประสิทธิภาพสูงมาก (Top Performers):**
   - **Brown Spot (80.4% mAP50)** และ **Leaf Blast (77.6% mAP50)** มีคะแนนสูงมากเนื่องจากลักษณะจุดรอยโรคมีความเด่นชัดและมีอัตรา Contrast สูงกับผืนใบข้าว
   - **Sheath Blight (74.5% mAP50)** มีอัตรา **Recall ถึง 80.0%** หมายความว่าเมื่อเกิดโรคกาบใบแห้งขึ้น โมเดลสามารถตรวจจับเจอเกือบทั้งหมด

2. **สาเหตุคลาส Bacterial Leaf Blight คะแนนต่ำ (10.5% mAP50):**
   - โรคขอบใบแห้ง (Bacterial Leaf Blight) มีลักษณะแผลลามตามยาวบริเวณขอบใบ ซึ่งมักมีลักษณะซ้อนทับและใกล้เคียงกับอาการขอบใบไหม้ธรรมดาหรือโรค Leaf Scald ส่งผลให้การตี Bounding Box เกิดความสับสน
   - อย่างไรก็ตาม ในระบบใช้งานจริงเมื่อนำไปทำงานร่วมกับ **RAG (Gemini 2.5 Flash)** ตัว LLM จะนำข้อมูลบริบทจากผู้ใช้มาช่วยกรองและยืนยันซ้ำอีกระดับหนึ่ง

3. **ความเร็วการประมวลผล (Inference Speed):**
   - Pre-process: **0.4 ms**
   - Inference: **5.5 ms** (~181 FPS บน RTX 4090)
   - Post-process: **0.2 ms**
   - ความเร็วรวมต่ำกว่า 10 ms ต่อรูป ทำให้พร้อมรองรับการส่งภาพจากผู้ใช้ผ่าน LINE Chatbot ได้แบบ Real-time

---

## 6. สรุปไฟล์โมเดลพร้อมใช้งาน (Deployment Artifacts)

- **ไฟล์น้ำหนัก PyTorch:** `runs/detect/runs/rtdetr_rice_v1-4/weights/best.pt` (66.2 MB)
- **ไฟล์น้ำหนัก ONNX (พร้อม Deploy บน Jetson):** `runs/detect/runs/rtdetr_rice_v1-4/weights/best.onnx` (62.9 MB, FP16 Half Precision)
- **ไฟล์รายงานคะแนน Text:** `runs/detect/runs/rtdetr_rice_v1-4/test_results.txt`

---

## 7. แผนการดำเนินการขั้นถัดไป (Next Steps)

1. ✅ **ปิด Phase 4 (Model Training & Evaluation):** บันทึกผลสำเร็จลงในระบบ
2. 🚀 **ลุยต่อ Phase 6 (LINE Chatbot & n8n Integration):**
   - นำไฟล์ `best.onnx` ไปสร้างบริการ Inference Service (FastAPI / Triton / ONNX Runtime Python)
   - เชื่อมต่อ API กับ n8n Workflow เพื่อรับรูปภาพจากผู้ใช้ LINE -> ส่งวิเคราะห์โรค -> ส่งชื่อโรคเข้า RAG Pipeline ต่อไป
