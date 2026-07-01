# RAGAS Evaluation - Rice Disease LINE Chatbot

ไฟล์นี้ใช้สำหรับประเมิน RAG Pipeline ด้วย RAGAS framework โดยแบ่งการทำงานออกเป็น 2 ขั้นตอนหลักเพื่อหลีกเลี่ยงข้อจำกัด (Rate Limit) ของโมเดล

## สิ่งที่ต้องติดตั้ง

```bash
pip install ragas datasets pinecone-client google-generativeai pandas
```

## วิธีใช้งาน

### 1. ไปที่โฟลเดอร์ RAGAS_Evaluation
```bash
cd RAGAS_Evaluation
```

### 2. สร้างกระดาษคำตอบ (Generation Step)
รันสคริปต์นี้เพื่อดึงข้อมูลจาก Pinecone และให้ **Gemini 3.1 Flash Lite** สร้างคำตอบ (ดึงมาทีละ 3 Chunks: Top K = 3)
```bash
set GEMINI_API_KEY=YOUR_GEMINI_KEY
set PINECONE_API_KEY=YOUR_PINECONE_KEY
python 01_generate_answers.py
```
*(เมื่อเสร็จสิ้น ระบบจะบันทึกผลลัพธ์คำตอบทั้ง 70 ข้อลงใน `evaluation_results/eval_results.json`)*

### 3. ประเมินผลข้อสอบ (Evaluation Step)
รันสคริปต์นี้เพื่อทำหน้าที่เป็นกรรมการตรวจข้อสอบและให้คะแนน โดยใช้ **OpenCodeZen (DeepSeek V4 Flash Free)**
```bash
set OPENCODE_ZEN_API_KEY=YOUR_OPENCODEZEN_KEY
python 02_evaluate_scores.py
```

## Options (สำหรับ 01_generate_answers.py)

| Flag | Default | Description |
|---|---|---|
| --api-key | env var | Gemini API key (override env var) |
| --pinecone-key | env var | Pinecone API key (override env var) |
| --pinecone-index | rice-knowledge | Pinecone index name |
| --top-k | 3 | Number of chunks to retrieve |
| --ground-truth | ragas_ground_truth.json | Ground truth Q&A file |
| --output | evaluation_results | Output directory |

## Output Files

- `evaluation_results/eval_results.json`   - กระดาษคำตอบที่สร้างโดย Gemini (ประกอบด้วย คำถาม, เฉลย, Context, คำตอบที่สร้าง)
- `evaluation_results/eval_details.csv`    - คะแนน RAGAS แบบละเอียดแยกรายข้อ (ทำหน้าที่เป็น Checkpoint ป้องกันข้อมูลหาย)
- `evaluation_results/eval_scores.json`    - ไฟล์สรุปคะแนน RAGAS เฉลี่ยรวมทั้งหมด 4 ด้าน

## RAGAS Metrics

| Metric | Range | Meaning |
|---|---|---|
| context_precision | 0-1 | ความแม่นยำในการดึง Context ที่เกี่ยวข้อง (ดึงมาตรงจุดแค่ไหน) |
| faithfulness | 0-1 | ความถูกต้องของคำตอบเทียบกับ Context (ไม่มีการมโนข้อมูลเอง) |
| answer_relevancy | 0-1 | ความตรงประเด็นของคำตอบต่อคำถาม (ตอบตรงคำถามไหม) |
| context_recall | 0-1 | ความสามารถในการดึง Context ที่จำเป็นมาได้ครบถ้วนหรือไม่ |

## Troubleshooting

- `ModuleNotFoundError`: รัน `pip install ragas datasets pinecone-client google-generativeai pandas`
- `KeyError: 0`: มักเกิดจากการใช้ไลบรารี Ragas ผิดเวอร์ชัน แต่ในโค้ดปัจจุบันได้รับการแก้ไขด้วย Pandas DataFrame แล้ว
- API key error: ตรวจสอบการตั้งค่า environment variable ในไฟล์ `.env` หรือตั้งค่าผ่าน Command Line

## หมายเหตุ

- สคริปต์ออกแบบให้รันจากโฟลเดอร์ `RAGAS_Evaluation/` เท่านั้น
- **Ground Truth มี 70 คำถาม-คำตอบ** (อัปเดตใหม่) ครอบคลุมโรคข้าว 5 ชนิดที่เราใช้ฝึกสอนโมเดล (โรคไหม้, ขอบใบแห้ง, ใบจุดสีน้ำตาล, กาบใบแห้ง, ใบไหม้แห้ง) โดยตัดโรคทังโร (Tungro) ออกไปแล้ว

## การทำงานของสคริปต์ (Workflow Deep Dive)

เพื่อให้เข้าใจกลไกเบื้องหลัง สามารถกลับมาทบทวนการทำงานของทั้ง 2 สคริปต์ได้ตามนี้ครับ:

### 1. `01_generate_answers.py` (ระบบค้นหาและตอบคำถามแบบ Semantic Search)
สคริปต์นี้เปรียบเสมือน **"แชตบอตทำข้อสอบ"** มีลำดับการทำงานคือ:
1. **เตรียมตัว:** อ่านโจทย์ 70 ข้อจากไฟล์ `ragas_ground_truth.json`
2. **ค้นหาข้อมูล (Retrieve):** นำคำถามไปแปลงเป็นตัวเลข (Vector) ด้วยโมเดล `gemini-embedding-001` แล้วค้นหาความเหมือนในฐานข้อมูล Pinecone (ดึงมา 3 ชิ้น ตามค่า Top K = 3)
3. **สร้างคำตอบ (Generate):** ส่งคำถามพร้อมกับบริบท (Context) 3 ชิ้นนั้นไปให้โมเดล **Gemini 3.1 Flash Lite** เพื่อสร้างคำตอบที่ตรงประเด็น
4. **บันทึก:** เซฟ [โจทย์, เฉลย, โพยที่หาได้, คำตอบที่สร้าง] ลงไฟล์ `evaluation_results/eval_results.json` 

### 2. `02_evaluate_scores.py` (ระบบคุณครูตรวจข้อสอบ)
สคริปต์นี้ทำหน้าที่เป็น **"กรรมการตรวจข้อสอบ"** มีลำดับการทำงานคือ:
1. **เตรียมความพร้อม:** ดึงกระดาษคำตอบจากไฟล์ `eval_results.json` ทั้ง 70 ข้อ
2. **แต่งตั้งกรรมการ:** 
   - **กรรมการตรวจเนื้อหา (LLM Judge):** ใช้ **DeepSeek V4 Flash Free** (ผ่าน OpenCodeZen) ในการอ่านวิเคราะห์และให้คะแนน เพื่อหลีกเลี่ยงปัญหา Rate Limit ของ Gemini
   - **กรรมการตรวจคำศัพท์ (Embedding):** ใช้ `gemini-embedding-001` ในการคำนวณระยะห่างของคำ (ซึ่งกินโควตาน้อยมาก)
3. **ตรวจข้อสอบ:** ส่งข้อมูลเข้าไลบรารี RAGAS เพื่อประเมินคะแนนทั้ง 4 ด้าน
4. **เซฟตี้ (Checkpoint):** ทันทีที่ประเมินเสร็จ ระบบจะรีบแปลงผลคะแนนแบบแยกรายข้อลงไฟล์ `evaluation_results/eval_details.csv` เพื่อป้องกันไม่ให้ข้อมูลหายหากระบบมีปัญหาตอนท้าย (แก้ปัญหา `KeyError` ในอดีต)
5. **สรุปผล:** คำนวณค่าเฉลี่ยของทั้ง 4 ด้าน และบันทึกลง `evaluation_results/eval_scores.json`
