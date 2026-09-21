from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from ultralytics import RTDETR
from PIL import Image
import io
import os
import glob
import math
import string
import json
import time
import requests
from collections import Counter
from dotenv import load_dotenv

# โหลด API Keys จาก .env
load_dotenv()

app = FastAPI(
    title="Rice Disease LINE Chatbot Unified API",
    description="API บริการตรวจจับโรคใบข้าว (RT-DETR) และตอบคำถามวิชาการ (RAG Pipeline)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 1. RT-DETR Configuration & Name Mapping ===
DISEASE_NAME_MAP = {
    "bacterial_leaf_blight": {
        "th": "โรคขอบใบแห้ง",
        "en": "Bacterial Leaf Blight",
        "scientific": "Xanthomonas oryzae pv. oryzae"
    },
    "brown_spot": {
        "th": "โรคใบจุดสีน้ำตาล",
        "en": "Brown Spot",
        "scientific": "Bipolaris oryzae"
    },
    "healthy": {
        "th": "ใบข้าวสมบูรณ์ดี (ไม่พบโรค)",
        "en": "Healthy Rice Leaf",
        "scientific": "N/A"
    },
    "leaf_blast": {
        "th": "โรคใบไหม้",
        "en": "Leaf Blast",
        "scientific": "Magnaporthe oryzae"
    },
    "leaf_scald": {
        "th": "โรคใบขอบกาบแห้ง",
        "en": "Leaf Scald",
        "scientific": "Microdochium oryzae"
    },
    "sheath_blight": {
        "th": "โรคกาบใบแห้ง",
        "en": "Sheath Blight",
        "scientific": "Rhizoctonia solani"
    }
}

def find_model_path():
    # 1. ลำดับสูงสุด: TensorRT Engine (.engine) สำหรับ NVIDIA Jetson AGX Orin (ความเร็วสูงสุด)
    search_engine = glob.glob("**/weights/best.engine", recursive=True)
    if search_engine:
        search_engine.sort(key=os.path.getmtime, reverse=True)
        return search_engine[0]

    # 2. ลำดับสอง: ONNX FP16 (.onnx)
    search_onnx = glob.glob("**/weights/best.onnx", recursive=True)
    if search_onnx:
        search_onnx.sort(key=os.path.getmtime, reverse=True)
        return search_onnx[0]

    # 3. ลำดับสาม: PyTorch Checkpoint (.pt)
    search_pt = glob.glob("**/weights/best.pt", recursive=True)
    if search_pt:
        search_pt.sort(key=os.path.getmtime, reverse=True)
        return search_pt[0]

    return "rtdetr-l.pt"

MODEL_PATH = find_model_path()
print(f"📦 Loading RT-DETR Model from: {MODEL_PATH}")
model = RTDETR(MODEL_PATH)


# === 2. RAG Pipeline Setup (Hybrid Search + Pinecone + PyThaiNLP) ===
DISEASE_KEYWORDS = {
    "bacterial_leaf_blight": ["ขอบใบแห้ง", "bacterial leaf blight", "blb", "ใบไหม้ขอบ"],
    "brown_spot": ["ใบจุดสีน้ำตาล", "brown spot", "จุดสีน้ำตาล", "ตาเหยี่ยว"],
    "leaf_blast": ["ไหม้", "blast", "leaf blast", "rice blast", "ไหม้คอรวง", "เน่าคอรวง", "pyricularia"],
    "leaf_scald": ["ใบขีดสีน้ำตาล", "leaf scald", "ขีดสีน้ำตาล", "ใบขีด"],
    "sheath_blight": ["กาบใบแห้ง", "sheath blight", "กาบใบ", "rhizoctonia"],
}

def get_emb(text: str) -> List[float]:
    """สร้าง Dense Vector 3072 มิติผ่าน Gemini Embedding REST API (เข้ากันได้กับ Python 3.8+ และ Jetson)"""
    api_key = os.getenv('GEMINI_API_KEY', '')
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={api_key}"
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]}
    }
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get('embedding', {}).get('values', [])

def detect_disease_from_query(query: str):
    query_lower = query.lower()
    for disease_id, keywords in DISEASE_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                return disease_id
    return None

def retrieve_rag_context(query: str, disease_id_override: Optional[str] = None, top_k: int = 5):
    from pinecone import Pinecone
    import mmh3
    import pythainlp

    pinecone_key = os.getenv('PINECONE_API_KEY', '')
    pc = Pinecone(api_key=pinecone_key)
    idx = pc.Index("rice-knowledge")

    dense_vector = get_emb(query)

    # Sparse BM25 vector calculation
    sparse_vector = None
    bm25_stats_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'bm25_stats.json')
    if os.path.exists(bm25_stats_path):
        with open(bm25_stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)

        query_clean = query.translate(str.maketrans('', '', string.punctuation)).lower()
        tokens = pythainlp.word_tokenize(query_clean, keep_whitespace=False)
        tokens = [t for t in tokens if len(t) > 1 or t.isdigit()]
        tf = Counter(tokens)

        doc_freq = stats['doc_freq']
        num_docs = stats['num_docs']

        indices = []
        values = []
        for term, freq in tf.items():
            token_id = abs(mmh3.hash(term)) % (2**31)
            df = doc_freq.get(term, 0)
            idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
            score = idf * freq
            if score > 0:
                indices.append(token_id)
                values.append(round(score, 6))

        if indices:
            sparse_vector = {"indices": indices, "values": values}

    disease_id = disease_id_override or detect_disease_from_query(query)

    # Alpha blending (Dense 70%, Sparse 30%)
    alpha = 0.7
    dense_vector = [v * alpha for v in dense_vector]
    if sparse_vector:
        sparse_vector["values"] = [v * (1.0 - alpha) for v in sparse_vector["values"]]

    query_params = {
        "vector": dense_vector,
        "top_k": top_k,
        "include_metadata": True,
    }
    if sparse_vector:
        query_params["sparse_vector"] = sparse_vector
    if disease_id:
        query_params["filter"] = {"disease_id": {"$eq": disease_id}}

    res = idx.query(**query_params)
    contexts = [m['metadata']['text'] for m in res.get('matches', [])]
    return contexts, disease_id

def generate_rag_answer(query: str, contexts: List[str]) -> str:
    api_key = os.getenv('GEMINI_API_KEY', '')
    ctx = "\n".join(contexts)
    prompt = (
        "คุณคือผู้เชี่ยวชาญด้านโรคข้าวจากกรมการข้าว\n"
        "ตอบคำถามโดยอ้างอิงจาก Context ที่ให้เท่านั้น\n"
        "ถ้าไม่พบข้อมูลใน Context ให้ตอบว่าไม่ทราบ และกระชับ สรุปเข้าใจง่าย เหมาะสำหรับชาวนาอ่านบน LINE\n"
        "ห้ามสร้างข้อมูลเอง (No Hallucination)\n\n"
        f"Context:\n{ctx}\n\n"
        f"คำถาม: {query}"
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                return data['candidates'][0]['content']['parts'][0]['text']
            elif resp.status_code == 429:
                time.sleep(2)
                continue
            else:
                resp.raise_for_status()
        except Exception as e:
            if attempt < 2 and ('429' in str(e) or 'Quota' in str(e) or 'exhausted' in str(e).lower()):
                time.sleep(2)
            else:
                raise e
    return "ขออภัยครับ ระบบประมวลผลคำตอบขัดข้องชั่วคราว"


# === 3. Pydantic Models ===
class RAGQueryRequest(BaseModel):
    question: str
    disease_id: Optional[str] = None


# === 4. Endpoints ===
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": MODEL_PATH,
        "service": "Rice Disease LINE Chatbot API"
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # อ่านไฟล์ภาพ
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="ไฟล์ที่อัปโหลดไม่ใช่รูปภาพที่ถูกต้อง (รองรับ JPEG, PNG)")

    try:
        results = model(image, imgsz=640, verbose=False)[0]

        detections = []
        max_confidence = 0.0
        primary_detection = None

        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls[0].item())
                class_name_raw = results.names[class_id]
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()

                info = DISEASE_NAME_MAP.get(class_name_raw, {
                    "th": class_name_raw,
                    "en": class_name_raw,
                    "scientific": "Unknown"
                })

                det_item = {
                    "class_id": class_id,
                    "class_name_raw": class_name_raw,
                    "disease_name_th": info["th"],
                    "disease_name_en": info["en"],
                    "scientific_name": info["scientific"],
                    "confidence": round(conf, 4),
                    "bbox": [round(x, 2) for x in xyxy]
                }
                detections.append(det_item)

                if conf > max_confidence:
                    max_confidence = conf
                    primary_detection = det_item

        if max_confidence >= 0.50 and primary_detection:
            status = "detected"
            message = f"พบรอยโรค {primary_detection['disease_name_th']} (ความมั่นใจ {primary_detection['confidence']*100:.1f}%)"
        elif 0.30 <= max_confidence < 0.50 and primary_detection:
            status = "uncertain"
            message = f"ไม่แน่ใจเกี่ยวกับรอยโรค (พบแนวโน้มเป็น {primary_detection['disease_name_th']} ความมั่นใจ {primary_detection['confidence']*100:.1f}%) กรุณาถ่ายภาพใหม่ให้ชัดเจนยิ่งขึ้น"
        else:
            status = "no_disease"
            message = "ไม่พบรอยโรคพืช หรือใบข้าวสมบูรณ์ดี"

        return {
            "success": True,
            "status": status,
            "message": message,
            "max_confidence": round(max_confidence, 4),
            "primary_detection": primary_detection,
            "all_detections": detections,
            "total_detected_count": len(detections)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการประมวลผลรูปภาพ: {str(e)}")

@app.post("/rag_query")
def rag_query(req: RAGQueryRequest):
    try:
        contexts, detected_disease = retrieve_rag_context(req.question, req.disease_id)
        if not contexts:
            return {
                "success": True,
                "question": req.question,
                "answer": "ขออภัยครับ ไม่พบข้อมูลเกี่ยวกับเรื่องนี้ในฐานข้อมูลกรมการข้าว",
                "contexts": []
            }
        answer = generate_rag_answer(req.question, contexts)
        return {
            "success": True,
            "question": req.question,
            "disease_id_filtered": detected_disease,
            "answer": answer,
            "contexts": contexts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการสืบค้น RAG: {str(e)}")

@app.post("/diagnose_and_advise")
async def diagnose_and_advise(file: UploadFile = File(...)):
    """
    Endpoint รวมร่าง: ตรวจรูปด้วย RT-DETR -> ถ้าพบโรค ดึงวิธีรักษาจาก RAG ทันทีในคำขอเดียว!
    """
    pred_res = await predict_image(file)
    if pred_res["status"] == "detected" and pred_res["primary_detection"]:
        disease_raw = pred_res["primary_detection"]["class_name_raw"]
        disease_th = pred_res["primary_detection"]["disease_name_th"]
        
        query_text = f"วิธีการรักษาและป้องกัน{disease_th}"
        rag_res = rag_query(RAGQueryRequest(question=query_text, disease_id=disease_raw))

        return {
            "success": True,
            "detection": pred_res,
            "advice": rag_res.get("answer", "")
        }
    else:
        return {
            "success": True,
            "detection": pred_res,
            "advice": "ไม่จำเป็นต้องใช้วิธีรักษา เนื่องจากไม่พบรอยโรคหรือภาพไม่ชัดเจน"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
