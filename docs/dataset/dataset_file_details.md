# รายละเอียดไฟล์ Dataset

## 1. ชุดข้อมูล Labeled (data/labeled - สำหรับ Object Detection)

ชุดข้อมูล Labeled ถูกจัดสัดส่วนออกเป็น 3 ชุด (Train / Valid / Test) รวมทั้งสิ้น **8,474 รูป** และมีจำนวน Bounding Box (Instances) รวม **18,030 กล่อง** ครอบคลุม 5 คลาสโรคพืช:

### จำนวนรูปภาพและ Bounding Boxes แยกตามคลาส

| คลาส (Class Name) | Train (รูป / กล่อง) | Valid (รูป / กล่อง) | Test (รูป / กล่อง) | รวมรูปภาพที่พบโรค (Images) | รวม Bounding Boxes (Instances) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **bacterial_leaf_blight** | 1,335 / 5,155 | 127 / 520 | 64 / 247 | **1,526** | **5,922** |
| **brown_spot** | 1,308 / 4,603 | 125 / 467 | 61 / 215 | **1,494** | **5,285** |
| **leaf_blast** | 1,194 / 1,616 | 113 / 150 | 57 / 81 | **1,364** | **1,847** |
| **leaf_scald** | 1,110 / 1,584 | 105 / 156 | 53 / 69 | **1,268** | **1,809** |
| **sheath_blight** | 1,254 / 2,761 | 119 / 276 | 60 / 130 | **1,433** | **3,167** |
| **รวม Bounding Boxes ทั้งหมด** | **— / 15,719** | **— / 1,569** | **— / 742** | **—** | **18,030** |

---

## 2. สรุปสัดส่วน Dataset Split (data/labeled)

- **Train Set**: 7,416 รูป (87.5%) | 15,719 Bounding Boxes
- **Validation Set**: 704 รูป (8.3%) | 1,569 Bounding Boxes
- **Test Set**: 354 รูป (4.2%) | 742 Bounding Boxes
- **รวมภาพทั้งหมด**: 8,474 รูป | **18,030 Bounding Boxes**

---

## 3. ชุดข้อมูล Raw เดิม (data/raw - ภาพต้นฉบับก่อน Split/Augmentation)

ชุดข้อมูลภาพต้นฉบับแยกโฟลเดอร์ 6 คลาส (รวมคลาส Healthy Rice Leaf) มีจำนวนรวม **3,829 รูป**:

- **Bacterial Leaf Blight**: 636 รูป
- **Brown Spot**: 646 รูป
- **Healthy Rice Leaf**: 653 รูป
- **Leaf Blast**: 634 รูป
- **Leaf Scald**: 628 รูป
- **Sheath Blight**: 632 รูป
