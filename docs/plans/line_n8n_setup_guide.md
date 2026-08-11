# คู่มือการตั้งค่า LINE Official Account & n8n Workflow (Phase 6)

คู่มือนี้สรุปขั้นตอนการสร้าง **LINE Official Account (Messaging API)**, การติดตั้ง **n8n Workflow**, และการเปิดใช้งานเซิร์ฟเวอร์ **FastAPI (RT-DETR + RAG API)** เพื่อเริ่มให้บริการแชทบอทตอบคำถามและวินิจฉัยโรคข้าว

---

## 1. การเปิดใช้งาน FastAPI Service (RT-DETR + RAG API)

ก่อนเปิด n8n ต้องเปิดให้บริการ API เซิร์ฟเวอร์ในระบบก่อน:

```bash
# 1. ติดตั้ง Dependencies
pip install -r api/requirements.txt

# 2. รัน FastAPI Server (พอร์ต 8000)
python api/app.py
# หรือ
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

ทดสอบเปิดเบราว์เซอร์ไปที่: `http://localhost:8000/health` หรือ `http://localhost:8000/docs` (Swagger UI)

---

## 2. การสร้างและตั้งค่า LINE Official Account (Messaging API)

1. เข้าไปที่ [LINE Developers Console](https://developers.line.biz/)
2. สร้าง **Provider** ใหม่ (เช่น `Rice Leaf Disease Diagnosis`)
3. สร้าง **Messaging API Channel** 
4. เข้าไปที่เมนู **Messaging API Settings**:
   - คัดลอก **Channel Access Token (long-lived)** 
   - คัดลอก **Channel Secret** จากเมนู Basic Settings
5. ตั้งค่า **Webhook**:
   - เปิดสวิตช์ **Use webhook** เป็น `ON`
   - ปิดสวิตช์ **Auto-response messages** และ **Greeting messages** ใน LINE Official Account Manager เพื่อให้ n8n ทำหน้าที่ตอบแทนทั้งหมด

---

## 3. การติดตั้ง n8n และการนำเข้า Workflow

### 3.1 การรัน n8n ด้วย Docker Compose (แนะนำ)
```bash
# ย้ายเข้าโฟลเดอร์ n8n แล้วสั่งรันใน Background
cd n8n
docker compose up -d
```
*หมายเหตุ: ไฟล์ `n8n/docker-compose.yml` ถูกตั้งค่า `restart: always`, Timezone `Asia/Bangkok`, และเชื่อมต่อ IP `158.108.113.124:5678` ไว้เรียบร้อยแล้ว ข้อมูลทั้งหมดจะถูกเก็บถาวรใน Volume `n8n_data`*

### 3.2 การ Import Workflow
1. เปิดหน้าเว็บ n8n (เช่น `http://localhost:5678`)
2. ไปที่เมนู **Workflows** -> **Import from File**
3. เลือกไฟล์ `n8n/rice_bot_workflow.json`
4. ดับเบิลคลิกที่โหนด **HTTP Request** ของ LINE เพื่อเปลี่ยนคำว่า `YOUR_LINE_CHANNEL_ACCESS_TOKEN` เป็น Token จริงจาก LINE Console
5. บันทึกและกดสวิตช์มุมขวาบนให้เป็น **Active**

---

## 4. การเปิด Webhook สำหรับ LINE (Tunneling / Public URL)

หากรัน n8n บนเครื่อง Local ต้องทำการเปิด Public URL เพื่อให้ LINE ส่ง Webhook เข้ามาได้:

```bash
# ใช้ ngrok ชี้ไปที่พอร์ต 5678 (พอร์ตของ n8n)
ngrok http 5678
```

คัดลอก HTTPS URL ที่ได้จาก ngrok (เช่น `https://xxxx.ngrok-free.app/webhook/line-webhook`) ไปแปะที่ช่อง **Webhook URL** ใน LINE Developers Console แล้วกดปุ่ม **Verify**

---

## 5. การทดสอบการใช้งาน (Testing)

1. **ทดสอบส่งรูปภาพ:** ถ่ายภาพใบข้าวที่เป็นโรคส่งเข้าไปในแชท LINE -> แชทบอทจะตอบกลับด้วยผลวิเคราะห์โรค (ความมั่นใจ %) และวิธีรักษาเชิงวิชาการจากกรมการข้าว
2. **ทดสอบพิมพ์ข้อความ:** พิมพ์ถาม "โรคใบจุดสีน้ำตาลรักษายังไง" -> แชทบอทจะดึงคำตอบจาก RAG Pipeline ตอบกลับทันที
