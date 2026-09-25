# Rice Leaf Disease — Segmentation Study (Ablation Study)

การทดลองเปรียบเทียบเชิงสัณฐานวิทยาและประสิทธิภาพ (Ablation Study) ระหว่าง **Direct Detection (Champion v5)** กับ **Segmented-First Detection** สำหรับนำผลการทดลองไปลงในปริญญานิพนธ์ (บทที่ 4)

---

## สถาปัตยกรรมและไฟล์ในโฟลเดอร์นี้

1. **`batch_segment_sam.py` (Step 1):**
   * สคริปต์ตัดพื้นหลังภาพแปลงนาออกเป็นสีดำด้วย **Meta Segment Anything Model (MobileSAM)**
   * ใช้ Bounding Box ของรอยโรคใน `data/labeled_datacentric-blb` เป็น Prompt นำทาง เพื่อรับประกันว่ารอยโรคและใบข้าวจะไม่ถูกลบหาย
   * มี Safety Guardrail ป้องกันภาพจอดำ (< 2% non-black fallback to original)
   * บันทึกผลลัพธ์ไว้ที่ `dataset_segmented/` พร้อมสร้าง `data.yaml` อัตโนมัติ

2. **`train_segmented_rtdetr.py` (Step 2):**
   * สคริปต์เทรนโมเดล **RT-DETR-L** บนชุดข้อมูลที่ตัดพื้นหลังแล้ว (`dataset_segmented/`)
   * ควบคุมตัวแปร (Controlled Experiment): ใช้ Hyperparameters ชุดเดียวกับ Champion Model v5 (imgsz 800, batch 16, lr0 0.0001, no mosaic) ทุกประการ
   * บันทึก Weights และ Logs ไว้ที่ `runs/rtdetr_segmented_ablation/`

---

## ขั้นตอนการรันบนเครื่อง NVIDIA GeForce RTX 4090

```bash
# 1. เข้าโฟลเดอร์การทดลอง
cd experiments/segmentation_study

# 2. ตัดพื้นหลัง Dataset ทั้งหมดด้วย SAM (~3-5 นาที)
python3 batch_segment_sam.py

# 3. สั่งเทรน RT-DETR บนชุดข้อมูลที่ตัดพื้นหลังแล้ว
python3 train_segmented_rtdetr.py
```
