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
รันสคริปต์นี้เพื่อดึงข้อมูลจาก Pinecone และให้ **Gemini 3.1 Flash Lite** สร้างคำตอบ (ดึงมาทีละ 5 Chunks: Top K = 5)
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
| --top-k | 5 | Number of chunks to retrieve |
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
- **Ground Truth มี 70 คำถาม-คำตอบ** (อัปเดตใหม่) ครอบคลุมโรคข้าว 5 ชนิดที่เราใช้ฝึกสอนโมเดล (โรคไหม้, ขอบใบแห้ง, ใบจุดสีน้ำตาล, กาบใบแห้ง, ใบไหม้แห้ง)

## การทำงานของสคริปต์ (Workflow Deep Dive)

เพื่อให้เข้าใจกลไกเบื้องหลัง สามารถกลับมาทบทวนการทำงานของทั้ง 3 สคริปต์ได้ตามนี้ครับ:

### 1. `01_generate_answers.py` (ระบบค้นหาและตอบคำถามแบบ Hybrid Search)
สคริปต์นี้เปรียบเสมือน **"แชตบอตทำข้อสอบ"** มีลำดับการทำงานคือ:
1. **เตรียมตัว:** อ่านโจทย์ 70 ข้อจากไฟล์ `ragas_ground_truth.json`
2. **ค้นหาข้อมูล (Retrieve):** ใช้ระบบ **Hybrid Search** (ผสมผสานระหว่าง Gemini Dense Vector เพื่อความหมาย และ PyThaiNLP BM25 Sparse Vector เพื่อคีย์เวิร์ดภาษาไทย) โดยถ่วงน้ำหนักด้วย Alpha Blending = 0.7 แล้วดึงข้อมูลจาก Pinecone (Top K = 5)
3. **สร้างคำตอบ (Generate):** ส่งคำถามพร้อมกับบริบท (Context) 5 ชิ้นนั้นไปให้โมเดล **Gemini 3.1 Flash Lite** เพื่อสร้างคำตอบที่ตรงประเด็น
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

### 3. `03_fix_evaluate_scores.py` (ระบบหน่วยซ่อมแซมคะแนนฉุกเฉิน)
สคริปต์นี้ถูกพัฒนาขึ้นมาเพื่อแก้ปัญหาเฉพาะหน้า กรณีที่เซิร์ฟเวอร์ตรวจข้อสอบ (DeepSeek) ทำงานไม่ทันหรือโควตาหมดกลางคัน (เกิด Error Timeout หรือ 503) ทำให้มีคะแนนบางข้อหล่นหายและกลายเป็นช่องว่าง (`NaN`) ในไฟล์ CSV โดยมีขั้นตอนการทำงานคือ:
1. **สแกนหาช่องโหว่:** โหลดไฟล์ `eval_details.csv` ขึ้นมาสแกนหาแถวที่มีค่าคะแนนเป็นช่องว่าง (`NaN`)
2. **จัดชุดซ่อมแซม:** ดึงเฉพาะ "คำถามและคำตอบ" ในข้อที่มีปัญหาเหล่านั้นออกมาเป็นชุดย่อย (เช่น ดึงมาแค่ 33 ข้อ จาก 70 ข้อ)
3. **ส่งซ่อม:** ใช้ API Key สำรอง (หรืออันใหม่ที่โควตาเต็ม) ส่งข้อสอบชุดย่อยนี้ไปให้ RAGAS ตรวจใหม่อีกครั้ง
4. **ประกอบร่าง (Patching):** เมื่อได้คะแนนครบถ้วนแล้ว สคริปต์จะนำคะแนนที่ได้ใหม่ ไป "หยอด" อุดรูรั่วในแถวเดิมของ `eval_details.csv` โดยไม่กระทบกับคะแนนข้ออื่นที่เคยตรวจผ่านไปแล้ว
5. **สรุปผลใหม่:** คำนวณค่าเฉลี่ยใหม่แบบสมบูรณ์ 100% และบันทึกทับลงใน `eval_scores.json`
