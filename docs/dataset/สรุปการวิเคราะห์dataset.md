# สรุปการวิเคราะห์ Rice Leaf Diseases (Dataset ปัจจุบัน)
*อัปเดตล่าสุด: 16 พ.ค. 2568*



## 1. ภาพรวม Dataset
6 คลาส รวมทั้งหมด 3,829 รูป:
  - Bacterial Leaf Blight : 636 รูป  (16.6%)
  - Brown Spot            : 646 รูป  (16.9%)
  - Healthy Rice Leaf     : 653 รูป  (17.1%)
  - Leaf Blast            : 634 รูป  (16.6%)
  - Leaf Scald            : 628 รูป  (16.4%)
  - Sheath Blight         : 632 รูป  (16.5%)

หมายเหตุ: คลาส Tungro ถูกนำออกจาก dataset ในการอัปเดตครั้งนี้
นามสกุลไฟล์: .jpg ทั้งหมด — ตรวจสอบแล้ว ไม่พบไฟล์เสีย (corrupted)

ความละเอียดรูปภาพที่พบใน Dataset (จาก check_dataset.py):
  - 224x224   : 2,931 รูป  (ส่วนใหญ่ — ขนาดมาตรฐานสำหรับโมเดล)
  - 1600x1200 :   787 รูป
  - 4000x3000 :    37 รูป
  - 2160x3840 :    17 รูป
  - 2448x3264 :    15 รูป
  - อื่นๆ     :    ~13 รูป



## 2. สถานะ Class Balance
Dataset ชุดนี้มีความสมดุลสูงมาก — แต่ละคลาสมีจำนวนใกล้เคียงกัน
(ต่างกันสูงสุดเพียง 25 รูป ระหว่าง Leaf Scald 628 กับ Healthy 653)

ปัญหา Class Imbalance รุนแรงที่เคยมีในชุดข้อมูลเก่า (บางคลาสมี 2,000+)
ได้รับการแก้ไขแล้วในชุดข้อมูลปัจจุบัน



## 3. โปรเจกต์นี้ใช้โมเดลอะไร?
- RT-DETR (Real-Time DEtection Transformer) สำหรับ Object Detection
- ต้องตี Bounding Box ทุกรูป (ไม่ใช่แค่ภาพรวม classification)
- ใช้ YOLO Format ผ่าน Roboflow สำหรับ Labeling
- ใช้ Data Augmentation: Rotation, Brightness, Blur, Flip, Crop
- ใช้ Gemini 2.5 Flash สำหรับ RAG + Pinecone Vector DB + n8n workflow



## 4. ปัญหาที่ยังต้องระวัง
4.1 ความละเอียดรูปไม่สม่ำเสมอ (สำคัญ)
  - มีรูปหลายขนาดปะปนกัน (224x224, 1600x1200, 4000x3000 ฯลฯ)
  - ต้อง Resize ทุกรูปให้เป็นขนาดเดียวกัน (640x640) ก่อนเทรน
  - ควรใช้ letterbox resize เพื่อรักษา aspect ratio

4.2 Visual Similarity (คู่ที่แยกยาก)
  - Bacterial Leaf Blight <-> Leaf Scald
    (รอยไหม้ขอบใบ คล้ายกันมาก)
  - Brown Spot <-> Leaf Blast (ระยะแรก)
    (เป็นจุดเล็กสีน้ำตาลเหมือนกัน)
  - โรคระยะเริ่มต้น <-> Healthy
  - ส่งผลให้โมเดลมี Confusion สูงในคู่เหล่านี้

4.3 จำนวนรูปน้อยสำหรับ Object Detection
  - ~3,829 รูป ค่อนข้างน้อยสำหรับ Object Detection
  - ต้องพึ่ง Pre-trained Weights (COCO) และ Augmentation หนัก
  - เป้าหมายหลังทำ Augmentation ควรได้ ~8,000-12,000 รูป

4.4 ฮาร์ดแวร์และการจัดการ VRAM
  - NVIDIA RTX 2000 Ada Generation (16GB VRAM)
  - VRAM 16GB เพียงพอสำหรับ RT-DETR-L (imgsz 640)
  - สามารถขยาย Batch Size ได้ถึง 16-32 เพื่อให้เทรนเร็วขึ้น
  - ยังแนะนำใช้ Mixed Precision Training (fp16) เพื่อประสิทธิภาพสูงสุด

4.5 Missing Hyperparameters ใน Project Plan
  - ไม่ระบุ Learning Rate, Batch Size, Epochs, Optimizer
  - ไม่บอก Input Image Size
  - ไม่บอกใช้ Pre-trained Weights หรือ Train from scratch
  - ไม่บอก Scheduler / Warmup Strategy

4.6 ปัญหาเดิม (แก้ไขแล้ว)
  - วิธีการประเมินผล: เดิมใช้ Likert Scale วัด mAP/Precision/Recall
    -> ต้องแก้เป็นตารางค่าตัวเลขจากโมเดลโดยตรง
  - ค่าใช้จ่าย Cloud: Pinecone และ Gemini API มี Cost/Rate Limit
    -> ยืนยันการใช้งาน Pinecone และ Gemini เป็นตัวเลือกหลัก



## 5. สรุป Actions ที่ต้องทำ
เสร็จแล้ว:
- [x] ตรวจสอบไฟล์เสีย -> ไม่พบ
- [x] ยืนยันจำนวนไฟล์แต่ละคลาส
- [x] ตัด Tungro ออกจาก dataset
- [x] อัปเดตเอกสาร
- [x] Resize รูปทั้งหมดให้เป็น 640x640 (letterbox)
- [x] วางแผนและเริ่ม Label Bounding Box ใน Roboflow
- [x] ระบุ RT-DETR Variant และ Hyperparameters ใน Project Plan
- [x] แก้ตาราง 3-2 ใน Project — เปลี่ยนจาก Likert Scale
      เป็น Performance Metrics Table
- [x] เตรียม Budget Pinecone/Gemini หรือหา Local Alternative
- [x] ออกแบบ Augmentation Pipeline
