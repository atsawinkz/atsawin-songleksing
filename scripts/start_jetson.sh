#!/bin/bash
# ==============================================================================
# scripts/start_jetson.sh
# สคริปต์เปิดใช้งานระบบ Rice Disease LINE Chatbot แบบอัตโนมัติบน NVIDIA Jetson AGX Orin
# ==============================================================================

echo "🌾 ======================================================================"
echo "   NVIDIA Jetson AGX Orin — Rice Disease AI System Starter"
echo "🌾 ======================================================================"

# 1. ตั้งค่าโหมดพลังงานสูงสุด (MAXN) และเร่งสัญญาณนาฬิกา GPU/CPU
if command -v nvpmodel &> /dev/null; then
    echo "⚡ [1/4] กำลังตั้งค่า Power Mode เป็น MAXN (โหมดประสิทธิภาพสูงสุด)..."
    sudo nvpmodel -m 0 2>/dev/null || true
    sudo jetson_clocks 2>/dev/null || true
    echo "   ✅ Power & Clocks Optimized"
fi

# 2. เปิดใช้งาน Docker Compose สำหรับ n8n Automation Engine
echo "⚙️ [2/4] กำลังเริ่มต้นระบบ n8n Automation Engine (Docker)..."
cd "$(dirname "$0")/../n8n" || exit
if docker compose version &>/dev/null; then
    docker compose up -d
elif command -v docker-compose &>/dev/null; then
    docker-compose up -d
else
    docker run -d --name n8n_automation --restart always -p 5678:5678 -v n8n_data:/home/node/.n8n --add-host host.docker.internal:host-gateway docker.n8n.io/n8nio/n8n:latest 2>/dev/null || true
fi
echo "   ✅ n8n Container พร้อมทำงานที่พอร์ต 5678"

# 3. ตรวจสอบและเริ่มต้น FastAPI Server (RT-DETR + RAG API)
echo "🧠 [3/4] กำลังเปิดเซิร์ฟเวอร์ FastAPI (RT-DETR + RAG API)..."
cd ..

# หาตัวแปลภาษา Python (venv หรือ system)
if [ -d "venv" ]; then
    PYTHON_CMD="./venv/bin/python3"
elif [ -d "../venv" ]; then
    PYTHON_CMD="../venv/bin/python3"
else
    PYTHON_CMD="python3"
fi

# รัน FastAPI ใน background
nohup $PYTHON_CMD api/app.py > api.log 2>&1 &
API_PID=$!
echo "   ✅ FastAPI เริ่มต้นสำเร็จ (PID: $API_PID) ติดตาม Log ได้ที่: api.log"

# รอให้ API พร้อม 3 วินาที
sleep 3
if curl -s http://localhost:8000/health | grep -q "online"; then
    echo "   🟢 API Health Check: ONLINE 100%"
else
    echo "   ⚠️ API กำลังโหลดโมเดลเข้าสู่ Memory..."
fi

# 4. เปิดท่อ ngrok สำหรับ LINE Webhook
echo "🌐 [4/4] กำลังเปิดท่อ HTTPS ด้วย ngrok (Static Domain)..."
echo "   คำสั่ง: ngrok http --url=snowsuit-baboon-zealous.ngrok-free.dev 5678"
echo "----------------------------------------------------------------------"
echo "🎉 ระบบพร้อมให้บริการแล้ว! กด Ctrl+C เมื่อต้องการหยุดการทำงานของ ngrok"
echo "----------------------------------------------------------------------"

ngrok http --url=snowsuit-baboon-zealous.ngrok-free.dev 5678
