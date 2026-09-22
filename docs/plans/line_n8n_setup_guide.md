# คู่มือการติดตั้งและเปิดใช้งาน LINE Chatbot (Phase 6 Deployment Guide)

คู่มือนี้สรุปขั้นตอนการเปิดใช้งาน **LINE Official Account (Messaging API)**, การรัน **n8n Workflow**, และเซิร์ฟเวอร์ **FastAPI (RT-DETR + RAG API)** เพื่อเปิดให้บริการระบบวินิจฉัยโรคข้าวและตอบคำถามวิชาการแก่เกษตรกร

---

## 📂 0. การเตรียมไฟล์สำหรับเครื่องเซิร์ฟเวอร์

หากย้ายระบบไปรันบนเครื่องเซิร์ฟเวอร์ใหม่ (เช่น เครื่อง Ubuntu RTX 4090 หรือ Jetson AGX Orin):
1. คัดลอกโฟลเดอร์โปรเจกต์ `rice_leaf_diseases` ไปวางบนเครื่องเป้าหมาย
2. เปิด Terminal แล้วย้ายเข้าสู่โฟลเดอร์โปรเจกต์:
   ```bash
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

## 4. การเปิด HTTPS Webhook สำหรับ LINE (ngrok)

เนื่องจาก LINE Platform บังคับว่า Webhook URL **ต้องเป็น HTTPS เท่านั้น** (ไม่รองรับ HTTP ธรรมดา) เราจึงใช้ **ngrok** ในการเปิดท่อความปลอดภัย:

### 4.1 ผูก ngrok Authtoken (ทำเพียงครั้งแรกครั้งเดียว):
1. สมัคร/เข้าสู่ระบบที่ [ngrok Dashboard](https://dashboard.ngrok.com/) (ล็อกอินด้วย Google ฟรี)
2. ไปที่เมนู **Your Authtoken** แล้วคัดลอกโทเค็น
3. รันคำสั่งผูกโทเค็นใน Terminal:
   ```bash
   ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN>
   ```

### 4.2 เริ่มต้นเปิดท่อ HTTPS (ใช้ Static Domain ถาวร):
1. รันคำสั่ง ngrok โดยระบุ Static Domain ถาวร:
   ```bash
   ngrok http --url=snowsuit-baboon-zealous.ngrok-free.dev 5678
   ```
2. ลิงก์ Webhook ถาวรของคุณอัศวินจะเป็น:
   ```http
   https://snowsuit-baboon-zealous.ngrok-free.dev/webhook/line-webhook
   ```
3. นำลิงก์ `https://...` นี้ไปวางที่ช่อง **Webhook URL** ใน [LINE Developers Console](https://developers.line.biz/) -> กด **Update** -> กดเปิดสวิตช์ **Use webhook** เป็น `ON`

*หมายเหตุ: เมื่อใช้คำสั่ง `--url=snowsuit-baboon-zealous.ngrok-free.dev` ไม่ว่าจะเปิด-ปิดเครื่องใหม่กี่ครั้ง ลิงก์จะไม่เปลี่ยนอีกเลยครับ*

---

## 5. การทดสอบการใช้งานบนมือถือ (Testing)

1. สแกน QR Code เพิ่มเพื่อนแชทบอทใน LINE
2. **ทดสอบส่งรูปภาพ:** ถ่ายภาพใบข้าวที่เป็นโรคส่งเข้าไป -> แชทบอทจะตอบกลับด้วยชื่อโรค (ความมั่นใจ %) และวิธีรักษาเชิงวิชาการจากกรมการข้าว
3. **ทดสอบพิมพ์ข้อความ:** พิมพ์ถาม "โรคใบจุดสีน้ำตาลรักษายังไง" -> แชทบอทจะดึงคำตอบจาก RAG Pipeline ตอบกลับทันที

---

## 🔄 6. การเปิดรันใหม่เมื่อปิด/เปิดเครื่องเซิร์ฟเวอร์ (Server Restart Guide)

หากมีการปิดเครื่องเซิร์ฟเวอร์แล้วเปิดใหม่ สามารถสั่งรันระบบกลับขึ้นมาได้ง่ายๆ เพียง 3 คำสั่ง:

```bash
# 1. เปิดเซิร์ฟเวอร์ FastAPI (สมอง AI)
python api/app.py

# 2. เปิดระบบ n8n Automation (หากยังไม่ได้รัน)
cd n8n && sudo docker-compose up -d   # หรือ docker compose up -d

# 3. เปิดท่อ ngrok ด้วย Static Domain ถาวร
ngrok http --url=snowsuit-baboon-zealous.ngrok-free.dev 5678
```
*หมายเหตุ: บน Jetson ให้ใช้ `sudo docker-compose up -d` และเนื่องจากใช้ Static Domain URL ถาวร จึงไม่ต้องเข้าไปตั้งค่าแก้ Webhook URL บน LINE Developers ใหม่เลย*

---

## 🧹 7. การปิดใช้งานและล้างคีย์ความปลอดภัยออกจากเครื่อง (Cleanup & Decommissioning Guide)

เมื่อจบโปรเจกต์ หรือต้องการยกเลิกการใช้งานระบบอย่างสมบูรณ์ เพื่อป้องกันไม่ให้คีย์ความปลอดภัยค้างในเครื่อง:

### 7.1 ลบ ngrok Authtoken ออกจากเครื่อง:
- **วิธีที่ 1 (แก้ไขไฟล์คอนฟิก):** พิมพ์คำสั่ง `ngrok config edit` แล้วลบบรรทัด `authtoken: ...` ออก
- **วิธีที่ 2 (ลบไฟล์คอนฟิกทิ้ง):**
  - **Windows (PowerShell):** `Remove-Item "$env:LOCALAPPDATA\ngrok\ngrok.yml"`
  - **Linux / Ubuntu:** `rm ~/.config/ngrok/ngrok.yml`
- **ลบ Static Domain บน Cloud:** เข้าไปที่ [ngrok Domains Dashboard](https://dashboard.ngrok.com/cloud-edge/domains) กด `...` -> **Delete Domain**

### 7.2 ล้างข้อมูล n8n ถาวร (ลบ Workflow และ Credentials):
```bash
cd n8n
docker compose down -v
```
*(การใส่แฟล็ก `-v` จะทำการลบ Docker Volume `n8n_data` ล้างคีย์และ Workflow ออกทั้งหมดทันที)*

### 7.3 ยกเลิก LINE Channel Access Token:
- เข้าไปที่ [LINE Developers Console](https://developers.line.biz/) -> แท็บ **Messaging API** -> กดปุ่ม **Reissue** เพื่อให้โทเค็นเดิมที่เคยฝังไว้ในเครื่องใช้ไม่ได้ทันที
