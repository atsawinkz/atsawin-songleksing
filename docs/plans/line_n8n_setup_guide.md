# คู่มือการติดตั้งและเปิดใช้งาน LINE Chatbot (Phase 6 Deployment Guide)

คู่มือนี้สรุปขั้นตอนการเปิดใช้งาน **LINE Official Account (Messaging API)**, การรัน **n8n Workflow**, และเซิร์ฟเวอร์ **FastAPI (RT-DETR + RAG API)** เพื่อเปิดให้บริการระบบวินิจฉัยโรคข้าวและตอบคำถามวิชาการแก่เกษตรกร

---

## 📦 0. การเตรียมไฟล์สำหรับเครื่องเซิร์ฟเวอร์

หากย้ายระบบไปรันบนเครื่องเซิร์ฟเวอร์ใหม่ (เช่น เครื่อง Ubuntu RTX 4090 หรือ Jetson AGX Orin):
1. คัดลอกไฟล์ `rice_bot_deployment.zip` ไปวางบนเครื่องเป้าหมาย
2. แตกไฟล์ด้วยคำสั่ง:
   ```bash
   unzip rice_bot_deployment.zip -d rice_leaf_diseases
   cd rice_leaf_diseases
   ```

---

## 1. การเปิดใช้งาน FastAPI Service (RT-DETR + RAG API)

เซิร์ฟเวอร์ FastAPI ทำหน้าที่เป็นสมองประมวลผลทั้ง RT-DETR Model (`best.onnx`) และ RAG Pipeline:

```bash
# 1. ติดตั้ง Dependencies ทั้งหมด ( Fast-API, PyTorch, Ultralytics, Pinecone, PyThaiNLP, Gemini )
pip install -r api/requirements.txt

# 2. รัน FastAPI Server (พอร์ต 8000)
python api/app.py
```

- **ตรวจสอบการทำงาน:** เปิดเบราว์เซอร์ไปที่ `http://localhost:8000/health` (ต้องขึ้น status: "online") หรือ `http://localhost:8000/docs` (Swagger UI)

---

## 2. การติดตั้ง n8n ด้วย Docker Compose

```bash
# ย้ายเข้าโฟลเดอร์ n8n แล้วสั่งรันใน Background
cd n8n
docker compose up -d
```
*หมายเหตุ: ไฟล์ `n8n/docker-compose.yml` ถูกตั้งค่า `restart: always`, Timezone `Asia/Bangkok`, และเก็บบันทึกข้อมูลถาวรไว้ใน Volume `n8n_data` เรียบร้อยแล้ว*

---

## 3. การ Import n8n Workflow

1. เปิดเบราว์เซอร์ไปที่ n8n (เช่น `http://localhost:5678` หรือ `http://158.108.113.124:5678`)
2. ไปที่เมนู **Workflows** -> กดปุ่ม **Import from File**
3. เลือกไฟล์ `n8n/rice_bot_workflow.json`
4. *หมายเหตุ: คีย์ **LINE Channel Access Token** ถูกฝังตั้งค่าล่วงหน้าไว้ในไฟล์เรียบร้อยแล้ว ไม่จำเป็นต้องไล่แก้เอง*
5. กดสวิตช์มุมขวาบนเปลี่ยนสถานะเป็น **Active** เพื่อเริ่มทำงาน

---

## 4. การเปิด HTTPS Webhook สำหรับ LINE

เนื่องจาก LINE Platform บังคับว่า Webhook URL **ต้องเป็น HTTPS เท่านั้น** (ไม่รองรับ HTTP ธรรมดา) โดยสามารถเลือกวิธีเปิดท่อได้ 2 ทาง:

### 🌟 ทางเลือก A: ใช้ Pinggy (เหมาะสำหรับ: ทดสอบด่วน / สาธิตตอนนำเสนอผลงาน)
*ข้อดี: ไม่ต้องติดตั้ง ไม่ต้องสมัคร Token แต่ Free Session จะจำกัดเวลา ~60 นาที/ครั้ง*
```bash
ssh -p 443 -R0:localhost:5678 a.pinggy.io
```
- หน้าจอจะแสดง URL แบบ `https://...` (เช่น `https://xxxx.a.pinggy.link`)
- เติม `/webhook/line-webhook` ต่อท้าย กลายเป็น:
  `https://xxxx.a.pinggy.link/webhook/line-webhook`

---

### 🔧 ทางเลือก B: ใช้ ngrok (แนะนำสำหรับ: เปิดรันใช้งานยาวๆ ต่อเนื่อง)
*ข้อดี: ผูก Authtoken ฟรีครั้งเดียว แล้วเปิดรันค้างไว้ได้ยาวนาน ไม่ตัดทุก 60 นาที*
1. ผูกโทเค็น (ครั้งแรกครั้งเดียว):
   ```bash
   ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN>
   ```
2. รันคำสั่งเปิดพอร์ตค้างไว้:
   ```bash
   ngrok http 5678
   ```
- เติม `/webhook/line-webhook` ต่อท้ายลิงก์ `https://` ที่ได้ (เช่น `https://xxxx.ngrok-free.app/webhook/line-webhook`)

---

### 4.3 นำลิงก์ไปใส่ใน LINE Developers Console:
นำลิงก์ `https://...` ที่ได้จากทางเลือก A หรือ B ไปวางที่ช่อง **Webhook URL** ใน [LINE Developers Console](https://developers.line.biz/) -> กด **Update** -> กดเปิดสวิตช์ **Use webhook** เป็น `ON`

---

## 5. การทดสอบการใช้งานบนมือถือ (Testing)

1. สแกน QR Code เพิ่มเพื่อนแชทบอทใน LINE
2. **ทดสอบส่งรูปภาพ:** ถ่ายภาพใบข้าวที่เป็นโรคส่งเข้าไป -> แชทบอทจะตอบกลับด้วยชื่อโรค (ความมั่นใจ %) และวิธีรักษาเชิงวิชาการจากกรมการข้าว
3. **ทดสอบพิมพ์ข้อความ:** พิมพ์ถาม "โรคใบจุดสีน้ำตาลรักษายังไง" -> แชทบอทจะดึงคำตอบจาก RAG Pipeline ตอบกลับทันที
