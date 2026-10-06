from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from ultralytics import RTDETR
from PIL import Image
import io
import os
import sys
import glob
import math
import string
import json
import time
import requests
import re
from collections import Counter
from dotenv import load_dotenv

# โหลด API Keys จาก .env
load_dotenv()

# เพิ่ม directory ของสคริปต์เข้า sys.path เพื่อความสะดวกในการรัน
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def sanitize_line_text(text: str) -> str:
    """ลบเครื่องหมายดอกจัน Markdown (** และ *) ออกให้เกลี้ยงสำหรับแสดงผลบน LINE"""
    if not text:
        return text
    # 1. ลบ ** (Bold ใน Markdown)
    text = text.replace('**', '')
    # 2. ลบ * ที่เหลือทั้งหมด
    text = text.replace('*', '')
    return text.strip()

app = FastAPI(
    title="Rice Disease LINE Chatbot Unified API v3",
    description="API บริการตรวจจับโรคใบข้าว เวอร์ชัน 3: Dual-Confidence (80%) + Multi-Model Fallback (Gemini 2.5 Flash -> Gemini 3.1 Flash Lite -> Gemini 3.5 Flash Lite) + Graceful Context Fallback",
    version="3.0.0"
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
    for p in [
        "training/runs/detect/runs/rtdetr_rice_v5/weights/best.engine",
        "training/runs/detect/runs/rtdetr_rice_v5/weights/best.pt",
        "training/runs/detect/runs/rtdetr_rice_v5/weights/best.onnx",
    ]:
        if os.path.exists(p):
            return p
    search_pt = glob.glob("**/weights/best.pt", recursive=True)
    if search_pt:
        return search_pt[0]
    return "rtdetr-l.pt"

MODEL_PATH = find_model_path()
print(f"[v3] Loading RT-DETR Model from: {MODEL_PATH}")
model = RTDETR(MODEL_PATH)


# === 2. RAG Pipeline Setup (Hybrid Search + Pinecone + PyThaiNLP) ===
DISEASE_KEYWORDS = {
    "bacterial_leaf_blight": ["ขอบใบแห้ง", "bacterial leaf blight", "blb", "ใบไหม้ขอบ"],
    "brown_spot": ["ใบจุดสีน้ำตาล", "brown spot", "จุดสีน้ำตาล", "ตาเหยี่ยว"],
    "leaf_blast": ["ไหม้", "blast", "leaf blast", "rice blast", "ไหม้คอรวง", "เน่าคอรวง", "pyricularia"],
    "leaf_scald": ["ใบขีดสีน้ำตาล", "leaf scald", "ขีดสีน้ำตาล", "ใบขีด"],
    "sheath_blight": ["กาบใบแห้ง", "sheath blight", "กาบใบ", "rhizoctonia"],
}

_PINECONE_INDEX = None
_BM25_STATS = None

def get_pinecone_index():
    global _PINECONE_INDEX
    if _PINECONE_INDEX is None:
        try:
            from pinecone import Pinecone
            pinecone_key = os.getenv('PINECONE_API_KEY', '')
            if pinecone_key:
                pc = Pinecone(api_key=pinecone_key)
                _PINECONE_INDEX = pc.Index("rice-knowledge")
        except Exception as e:
            print(f"[v3 Warning] ไม่สามารถเชื่อมต่อ Pinecone: {e}")
    return _PINECONE_INDEX

def get_bm25_stats():
    global _BM25_STATS
    if _BM25_STATS is None:
        bm25_stats_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'bm25_stats.json')
        if os.path.exists(bm25_stats_path):
            try:
                with open(bm25_stats_path, 'r', encoding='utf-8') as f:
                    _BM25_STATS = json.load(f)
            except Exception as e:
                print(f"[v3 Warning] ไม่สามารถโหลด BM25 stats: {e}")
    return _BM25_STATS

def get_emb(text: str) -> List[float]:
    """สร้าง Dense Vector 3072 มิติผ่าน Gemini Embedding REST API พร้อมระบบ Error Handling"""
    api_key = os.getenv('GEMINI_API_KEY', '')
    if not api_key:
        return []
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={api_key}"
    payload = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]}
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get('embedding', {}).get('values', [])
        else:
            print(f"[v3 Warning] Gemini Embedding API ตอบกลับ HTTP {resp.status_code}")
            return []
    except Exception as e:
        print(f"[v3 Warning] เกิดข้อผิดพลาดขณะสร้าง Embedding: {e}")
        return []

def detect_disease_from_query(query: str):
    query_lower = query.lower()
    for disease_id, keywords in DISEASE_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                return disease_id
    return None

def retrieve_rag_context(query: str, disease_id_override: Optional[str] = None, top_k: int = 5):
    import mmh3
    import pythainlp

    disease_id = disease_id_override or detect_disease_from_query(query)
    idx = get_pinecone_index()
    if not idx:
        return [], disease_id

    dense_vector = get_emb(query)

    # Sparse BM25 vector calculation (ใช้ Cache ใน RAM)
    sparse_vector = None
    stats = get_bm25_stats()
    if stats:
        query_clean = query.translate(str.maketrans('', '', string.punctuation)).lower()
        tokens = pythainlp.word_tokenize(query_clean, keep_whitespace=False)
        tokens = [t for t in tokens if len(t) > 1 or t.isdigit()]
        tf = Counter(tokens)

        doc_freq = stats.get('doc_freq', {})
        num_docs = stats.get('num_docs', 1)

        indices = []
        values = []
        for term, freq in tf.items():
            token_id = abs(mmh3.hash(term)) % (2**31)
            df = doc_freq.get(term, 0)
            idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
            score = idf * freq
            if score > 0:
                indices.append(token_id)
                values.append(score)

        if indices:
            sparse_vector = {"indices": indices, "values": values}

    query_params = {
        "top_k": top_k,
        "include_metadata": True
    }
    if dense_vector and len(dense_vector) == 3072:
        query_params["vector"] = dense_vector
    elif sparse_vector:
        query_params["vector"] = [0.0] * 3072
    else:
        return [], disease_id

    if sparse_vector:
        query_params["sparse_vector"] = sparse_vector
    if disease_id:
        query_params["filter"] = {"disease_id": {"$eq": disease_id}}

    try:
        res = idx.query(**query_params)
        contexts = [m['metadata']['text'] for m in res.get('matches', [])]
        return contexts, disease_id
    except Exception as e:
        print(f"[v3 Warning] ไม่สามารถสืบค้น Pinecone ได้: {e}")
        return [], disease_id


# === 3. Multi-Model Gemini LLM Fallback (v3 Core Feature) ===
CANDIDATE_GEMINI_MODELS = [
    "gemini-2.5-flash",       # โมเดลหลัก (ตรงตามเล่มปริญญานิพนธ์)
    "gemini-3.1-flash-lite",  # โมเดลสำรอง 1 (โควตาสูง 500 RPD, เร็ว เสถียร ไม่ติด 503)
    "gemini-3.5-flash-lite",  # โมเดลสำรอง 2 (โควตาสูง 500 RPD)
    "gemini-2.0-flash"        # โมเดลสำรอง 3
]

def format_emergency_context_fallback(contexts: Optional[List[str]]) -> str:
    """
    กรณีฉุกเฉินที่ Google Gemini ขัดข้องทุกโมเดล:
    นำข้อความการรักษาจากฐานข้อมูล Pinecone (คู่มือกรมการข้าว) มาจัดส่งให้ผู้ใช้ทันที
    เพื่อป้องกันไม่ให้ API พังเป็น 500 หรือทำให้ LINE นิ่งสนิท
    """
    if not contexts:
        return (
            "ขออภัยครับ ระบบประมวลผลคำแนะนำของ AI ขัดข้องชั่วคราว\n"
            "กรุณาลองส่งภาพหรือสอบถามเข้ามาใหม่อีกครั้งในภายหลังครับ"
        )
    
    clean_parts = [sanitize_line_text(c) for c in contexts[:2]]
    return (
        "คำแนะนำการดูแลรักษาตามคู่มือกรมการข้าว:\n\n"
        + "\n\n".join(clean_parts)
    )

def call_gemini_with_fallback(prompt: str, contexts: Optional[List[str]] = None) -> str:
    """
    [v3 Core] ยิงคำขอไปยัง Gemini พร้อมระบบ Fallback หลายระดับ:
    1. ลองเรียก Gemini 2.5 Flash เป็นหลักก่อน
    2. หากเจอ 503 (Overloaded) หรือ 429 (Quota เต็ม) จะสลับไปเรียก Gemini 3.1 Flash Lite / 3.5 Flash Lite ทันที
    3. หากทุกโมเดลขัดข้อง จะใช้ Graceful Fallback ดึงข้อมูลจาก Pinecone แทน เพื่อไม่ให้เกิด Error 500
    """
    api_key = os.getenv('GEMINI_API_KEY', '')
    if not api_key:
        print("[v3 Warning] ไม่พบ GEMINI_API_KEY ในตัวแปรสภาพแวดล้อม")
        return format_emergency_context_fallback(contexts)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    for model_name in CANDIDATE_GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        
        try:
            resp = requests.post(url, json=payload, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                raw_answer = data['candidates'][0]['content']['parts'][0]['text']
                print(f"[v3 RAG] สำเร็จ: ประมวลผลคำตอบด้วยโมเดล {model_name}")
                return sanitize_line_text(raw_answer)

            status = resp.status_code
            print(f"[v3 RAG] โมเดล {model_name} ตอบกลับรหัส HTTP {status}")

            # หากเป็น 503 (Overload/Service Unavailable) หรือ 429 (Quota/Rate Limit เต็ม)
            # สลับไปใช้โมเดลสำรองตัวถัดไปทันทีอัตโนมัติ (เช่น 3.1 Flash Lite / 3.5 Flash Lite)
            if status in [503, 429]:
                print(f"[v3 RAG Fallback] ตรวจพบ HTTP {status} บน {model_name} (ติด 503 หรือโควตาเต็ม) -> สลับไปใช้โมเดลสำรองถัดไปทันที...")
                continue
            else:
                print(f"[v3 RAG Fallback] โมเดล {model_name} ตอบกลับรหัส HTTP {status} -> สลับไปยังโมเดลถัดไป...")
                continue

        except requests.exceptions.Timeout:
            print(f"[v3 RAG Timeout] โมเดล {model_name} หมดเวลาเชื่อมต่อ (เกิน 12 วินาที) -> สลับไปใช้โมเดลสำรองถัดไปทันที...")
            continue
        except Exception as e:
            print(f"[v3 RAG Exception] ข้อผิดพลาดขณะเรียก {model_name}: {e} -> สลับไปยังโมเดลสำรอง...")
            continue

    # หากลองทุกโมเดลแล้วยังไม่สำเร็จ ให้ใช้ข้อมูลจาก Pinecone โดยตรง
    print("[v3 RAG Emergency Fallback] ไม่สามารถเชื่อมต่อ Gemini ได้ทุกโมเดล กำลังใช้ข้อมูลจาก Pinecone สำรอง")
    return format_emergency_context_fallback(contexts)


def generate_rag_answer(query: str, contexts: List[str]) -> str:
    """ตอบคำถามข้อความทั่วไป (Text RAG) ผ่านระบบ Fallback v3"""
    ctx = "\n".join(contexts)
    prompt = (
        "คุณคือผู้เชี่ยวชาญด้านโรคข้าวจากกรมการข้าว ให้คำปรึกษาแก่เกษตรกรผ่าน LINE\n"
        "คำแนะนำในการตอบ:\n"
        "1. ตอบคำถามโดยอ้างอิงจาก Context ที่ให้เท่านั้น สรุปกระชับ เข้าใจง่าย เหมาะสำหรับอ่านบน LINE\n"
        "2. ห้ามสร้างข้อมูลเอง (No Hallucination) และห้ามใช้อิโมจิ (Emoji) ในคำตอบเด็ดขาด\n"
        "3. ห้ามใช้เครื่องหมายดอกจัน ** หรือ * ในการเน้นคำเด็ดขาด\n"
        "4. หากมีการแจกแจงรายการหรือขั้นตอน ให้ใช้ตัวเลขลำดับ 1. 2. 3. ชัดเจน\n"
        "5. หากคำถามเป็นเรื่องทั่วไป คำถามนอกเรื่อง หรือไม่พบข้อมูลใน Context ที่เกี่ยวข้องกับข้าวและการเกษตร:\n"
        "   - ห้ามตอบแค่ 'ไม่ทราบ' สั้นๆ เด็ดขาด\n"
        "   - ให้ตอบอย่างสุภาพว่า: 'ขออภัยครับ ระบบนี้เป็น AI ผู้ช่วยวินิจฉัยและให้คำปรึกษาเรื่องโรคข้าวจากกรมการข้าวโดยเฉพาะ จึงสามารถตอบได้เฉพาะเรื่องโรคข้าวและการดูแลรักษาเท่านั้นครับ'\n"
        "   - พร้อมให้คำแนะนำวิธีการใช้งาน เช่น สามารถส่งรูปถ่ายใบข้าวเพื่อให้ระบบช่วยตรวจจับรอยโรค หรือพิมพ์สอบถามเกี่ยวกับโรคข้าว เช่น 'วิธีรักษาโรคขอบใบแห้ง' หรือ 'อาการโรคใบไหม้'\n\n"
        f"Context:\n{ctx}\n\n"
        f"คำถาม: {query}"
    )
    return call_gemini_with_fallback(prompt, contexts)

def generate_multi_candidate_advice(candidates: List[dict], contexts: List[str]) -> str:
    """
    [v3] สร้างคำแนะนำสำหรับกรณีความมั่นใจต่ำกว่า 80%:
    1. ให้คำแนะนำการรักษาโรคอันดับ 1 เป็นหลัก
    2. ให้จุดสังเกตรอยโรคในแปลงเพื่อช่วยแยกแยะโรคที่เข้าข่าย
    """
    ctx = "\n".join(contexts)
    cand_summary = "\n".join([
        f"{c['rank']}. {c['disease_th']} ({c['percentage']})"
        for c in candidates
    ])
    primary_name = candidates[0]['disease_th']

    prompt = (
        "คุณคือผู้เชี่ยวชาญด้านโรคข้าวจากกรมการข้าว ให้คำปรึกษาแก่เกษตรกรผ่าน LINE\n"
        f"AI ตรวจพบรอยโรคที่มีความมั่นใจต่ำกว่า 80% โดยมีโอกาสเป็นโรคดังนี้:\n{cand_summary}\n\n"
        "กรุณาสรุปคำแนะนำสั้นกระชับ เป็นภาษาไทยที่อ่านง่าย เหมาะกับหน้าจอ LINE แบ่งเป็น 2 หัวข้อดังนี้:\n\n"
        f"1. แนวทางรักษาและดูแลเบื้องต้น (เน้นโรคที่มีโอกาสเป็นมากที่สุดคือ: {primary_name}):\n"
        "   - สรุปแนวทางจัดการเบื้องต้นที่สำคัญ เช่น การปรับระดับน้ำ การปรับปุ๋ย หรือสารเคมีที่แนะนำตาม Context\n\n"
        "2. จุดสังเกตในแปลงเพื่อแยกแยะโรค:\n"
        "   - อธิบายสั้นๆ ข้อละ 1 บรรทัด ว่าแผลของแต่ละโรคข้างต้นสังเกตต่างกันอย่างไรในแปลงจริง\n\n"
        "ท้ายข้อความให้บอกว่า 'หากตรวจสอบแล้วสงสัยโรคใดเป็นพิเศษ สามารถพิมพ์ชื่อโรคเข้ามาเพื่อขอข้อมูลละเอียดได้ครับ'\n"
        "ห้ามแต่งข้อมูลเท็จ ตอบตาม Context ที่ให้เท่านั้น และห้ามใส่อิโมจิ (Emoji) ในข้อความเด็ดขาด\n"
        "ห้ามใช้เครื่องหมายดอกจัน ** หรือ * ในการเน้นคำเด็ดขาด และให้ใช้ตัวเลขลำดับ 1. 2. 3. ชัดเจน\n\n"
        f"Context:\n{ctx}"
    )
    return call_gemini_with_fallback(prompt, contexts)


# === 4. Pydantic Models ===
class RAGQueryRequest(BaseModel):
    question: str
    disease_id: Optional[str] = None


# === 5. Endpoints (Version 3) ===
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "version": "3.0.0",
        "mode": "Dual-Confidence (80%) + Multi-Model Fallback (Gemini 2.5 Flash -> 3.1 Flash Lite -> 3.5 Flash Lite)",
        "model_loaded": MODEL_PATH,
        "service": "Rice Disease LINE Chatbot API v3"
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    [v3] วิเคราะห์รูปภาพ:
    - ถ้าความมั่นใจสูงสุด >= 80% -> สรุปเป็นโรคเดียว (High Confidence)
    - ถ้า 30% <= ความมั่นใจ < 80% -> รวบรวม Top 3 โรคที่มีโอกาสเป็นมากที่สุด พร้อม %
    - ถ้าความมั่นใจ < 30% -> ไม่พบรอยโรคหรือภาพไม่ชัดเจน
    """
    try:
        contents = await file.read()
        if len(contents) > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="ไฟล์มีขนาดใหญ่เกินไป (รองรับขนาดไม่เกิน 20MB)")
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="ไฟล์ภาพว่างเปล่า (0 bytes)")
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="ไฟล์ที่อัปโหลดไม่ใช่รูปภาพที่ถูกต้อง (รองรับ JPEG, PNG)")

    try:
        results = model(image, imgsz=640, conf=0.01, verbose=False)[0]

        class_scores = {name: 0.0 for name in results.names.values()}
        detections = []

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = results.names[cls_id]
                conf = float(box.conf[0].item())
                xyxy = [round(x, 2) for x in box.xyxy[0].tolist()]

                if conf > class_scores[cls_name]:
                    class_scores[cls_name] = conf

                info = DISEASE_NAME_MAP.get(cls_name, {
                    "th": cls_name,
                    "en": cls_name,
                    "scientific": "Unknown"
                })

                detections.append({
                    "class_id": cls_id,
                    "class_name_raw": cls_name,
                    "disease_name_th": info["th"],
                    "disease_name_en": info["en"],
                    "confidence": round(conf, 4),
                    "bbox": xyxy
                })

        sorted_diseases = sorted(
            [(k, v) for k, v in class_scores.items() if k != "healthy"],
            key=lambda x: x[1],
            reverse=True
        )

        primary_raw, max_conf = sorted_diseases[0] if sorted_diseases else ("healthy", 0.0)
        primary_info = DISEASE_NAME_MAP.get(primary_raw, {"th": primary_raw, "en": primary_raw, "scientific": "Unknown"})

        # === ตัดสินใจตามเกณฑ์ 80% ===
        if max_conf >= 0.80:
            status = "detected_single"
            is_high_confidence = True
            message = (
                f"พบรอยโรค: {primary_info['th']} (ความมั่นใจ {max_conf * 100:.1f}%)\n"
                f"(ระดับความมั่นใจสูงเกิน 80% ระบบยืนยันผลการวินิจฉัยโรคนี้)"
            )
            top_candidates = [{
                "rank": 1,
                "disease_id": primary_raw,
                "disease_th": primary_info["th"],
                "disease_en": primary_info["en"],
                "confidence": round(max_conf, 4),
                "percentage": f"{max_conf * 100:.1f}%"
            }]

        elif 0.30 <= max_conf < 0.80:
            status = "detected_multiple"
            is_high_confidence = False

            top_candidates = []
            for rank, (d_name, score) in enumerate(sorted_diseases[:3], 1):
                if score >= 0.005:
                    info = DISEASE_NAME_MAP.get(d_name, {"th": d_name, "en": d_name})
                    top_candidates.append({
                        "rank": rank,
                        "disease_id": d_name,
                        "disease_th": info["th"],
                        "disease_en": info["en"],
                        "confidence": round(score, 4),
                        "percentage": f"{score * 100:.1f}%"
                    })

            msg_lines = [
                f"ผลการประเมินรอยโรค (พบแนวโน้ม {len(top_candidates)} โรคที่มีโอกาสเป็นมากที่สุด):"
            ]
            for c in top_candidates:
                star = " (มีโอกาสเป็นมากที่สุด)" if c["rank"] == 1 else ""
                msg_lines.append(f"{c['rank']}. {c['disease_th']} — ความเป็นไปได้ {c['percentage']}{star}")
            msg_lines.append("\nเนื่องจากความมั่นใจยังไม่ถึง 80% รอยโรคอาจอยู่ในระยะเริ่มต้นหรือภาพถ่ายยังไม่ชัดเจน แนะนำถ่ายภาพระยะใกล้ที่มีแสงชัดเจนเพิ่มเติมเพื่อยืนยันอีกครั้งครับ")
            message = "\n".join(msg_lines)

        else:
            status = "no_disease"
            is_high_confidence = False
            top_candidates = []
            message = (
                "ไม่พบรอยโรคข้าว หรือภาพถ่ายอาจไม่ใช่ใบข้าว/มีระยะถ่ายไกลเกินไป (ความมั่นใจต่ำกว่า 30%) "
                "หากต้องการตรวจวินิจฉัยโรคข้าว กรุณาถ่ายภาพใบข้าวระยะใกล้ที่มีแสงสว่างชัดเจนเข้ามาใหม่อีกครั้งครับ"
            )

        return {
            "success": True,
            "status": status,
            "is_high_confidence": is_high_confidence,
            "message": message,
            "max_confidence": round(max_conf, 4),
            "primary_disease": primary_info if max_conf >= 0.30 else None,
            "top_candidates": top_candidates,
            "all_detections_count": len(detections)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการประมวลผลรูปภาพ (v3): {str(e)}")

@app.post("/rag_query")
def rag_query(req: RAGQueryRequest):
    try:
        q_clean = req.question.strip().lower()
        if not q_clean:
            return {
                "success": True,
                "question": req.question,
                "disease_id_filtered": None,
                "answer": (
                    "สวัสดีครับ ยินดีต้อนรับสู่ระบบผู้ช่วยวินิจฉัยและให้คำปรึกษาโรคข้าว\n\n"
                    "ท่านสามารถใช้งานได้ดังนี้ครับ:\n"
                    "- กดปุ่ม 'ถ่ายภาพตรวจโรคข้าว' หรือ 'ส่งภาพตรวจโรคข้าว' เพื่อตรวจหาโรคและดูรอยโรค\n"
                    "- พิมพ์สอบถามเกี่ยวกับโรคข้าว เช่น 'วิธีรักษาโรคขอบใบแห้ง' หรือ 'อาการโรคใบไหม้'"
                ),
                "contexts": []
            }
        disease_or_question_terms = ["โรค", "ไหม้", "ขอบใบ", "จุดสีน้ำตาล", "กาบใบ", "รักษา", "อาการ", "ยา", "สาร", "ป้องกัน", "กำจัด", "กี่กรัม", "อัตรา"]
        is_greeting = (
            q_clean in ["สวัสดี", "สวัสดีครับ", "สวัสดีค่ะ", "หวัดดี", "หวัดดีครับ", "หวัดดีค่ะ", "ดีครับ", "ดีค่ะ", "hello", "hi", "hey"]
            or (q_clean.startswith("สวัสดี") and not any(k in q_clean for k in disease_or_question_terms))
        )
        if is_greeting:
            greeting_answer = (
                "สวัสดีครับ ยินดีต้อนรับสู่ระบบผู้ช่วยวินิจฉัยและให้คำปรึกษาโรคข้าว\n\n"
                "ผมพร้อมช่วยตรวจวินิจฉัยและให้คำแนะนำโรคข้าว 5 โรคสำคัญครับ ท่านสามารถ:\n"
                "- กดปุ่ม 'ถ่ายภาพตรวจโรคข้าว' หรือ 'ส่งภาพตรวจโรคข้าว' เพื่อตรวจหาโรคและดูรอยโรค\n"
                "- พิมพ์สอบถามเกี่ยวกับโรคข้าว เช่น 'โรคไหม้รักษายังไง' หรือ 'อาการโรคขอบใบแห้ง'\n\n"
                "มีข้อสงสัยหรือต้องการตรวจโรคข้าว สามารถส่งรูปหรือพิมพ์ถามเข้ามาได้เลยครับ"
            )
            return {
                "success": True,
                "question": req.question,
                "disease_id_filtered": None,
                "answer": greeting_answer,
                "contexts": []
            }

        if any(t in q_clean for t in ["ขอบคุณ", "ขอบพระคุณ", "thank", "thx"]):
            thanks_answer = (
                "ยินดีเป็นอย่างยิ่งครับ หากพบปัญหาโรคข้าวในแปลงนาหรือมีข้อสงสัยเพิ่มเติม "
                "สามารถถ่ายภาพหรือพิมพ์สอบถามเข้ามาได้ตลอดเวลานะครับ ขอให้ผลผลิตข้าวเจริญงอกงามครับ"
            )
            return {
                "success": True,
                "question": req.question,
                "disease_id_filtered": None,
                "answer": thanks_answer,
                "contexts": []
            }

        is_user_guide = (
            q_clean in ["วิธีใช้", "วิธีใช้งาน", "วิธีการใช้งาน", "คู่มือ", "คู่มือการใช้งาน", "ใช้งานยังไง", "ใช้ยังไง", "help"]
            or any(t in q_clean for t in [
                "ขอวิธีการใช้งานแชทบอท",
                "วิธีการใช้งานแชทบอท",
                "วิธีใช้งานแชทบอท",
                "วิธีใช้แชทบอท",
                "วิธีใช้งานระบบ",
                "คู่มือการใช้งาน",
                "แนะนำวิธีใช้",
                "ขั้นตอนการใช้งาน"
            ])
        )
        if is_user_guide:
            guide_answer = (
                "ยินดีต้อนรับสู่ระบบแชทบอทผู้ช่วยวินิจฉัยและให้คำปรึกษาโรคข้าว\n\n"
                "ระบบนี้ช่วยตรวจวินิจฉัยและให้คำปรึกษา 5 โรคข้าวสำคัญ:\n"
                "(โรคขอบใบแห้ง, โรคใบจุดสีน้ำตาล, โรคใบไหม้, โรคใบขอบกาบแห้ง และโรคกาบใบแห้ง)\n\n"
                "วิธีการใช้งาน 3 ขั้นตอน:\n"
                "1. ตรวจโรคจากภาพถ่าย:\n"
                "   - กดปุ่ม 'ถ่ายภาพตรวจโรคข้าว' เพื่อถ่ายภาพจากแปลงนา หรือ 'ส่งภาพตรวจโรคข้าว' เพื่อเลือกภาพจากอัลบั้ม\n"
                "   - ระบบจะวิเคราะห์ระบุชื่อโรค พร้อมประเมินระดับความมั่นใจ\n"
                "2. คำแนะนำการรักษาและสารเคมี:\n"
                "   - หากตรวจพบโรค ระบบจะแนะนำชื่อสารเคมีและอัตราการใช้ที่ถูกต้องตามมาตรฐานกรมการข้าว\n"
                "3. สอบถามข้อมูลและข้อสงสัย:\n"
                "   - สามารถพิมพ์สอบถามชื่อโรคหรืออาการ เช่น 'โรคไหม้รักษายังไง' หรือ 'วิธีป้องกันโรคขอบใบแห้ง' ได้ตลอดเวลา\n\n"
                "ข้อแนะนำ: ควรถ่ายภาพใบข้าวในระยะใกล้ (15-30 ซม.) โฟกัสชัดเจน และมีแสงสว่างเพียงพอ เพื่อให้ระบบตรวจจับได้แม่นยำที่สุดครับ"
            )
            return {
                "success": True,
                "question": req.question,
                "disease_id_filtered": None,
                "answer": guide_answer,
                "contexts": []
            }

        contexts, detected_disease = retrieve_rag_context(req.question, req.disease_id)
        if not contexts:
            return {
                "success": True,
                "question": req.question,
                "answer": (
                    "ขออภัยครับ ระบบนี้เป็น AI ผู้ช่วยวินิจฉัยและให้คำปรึกษาเรื่องโรคข้าวจากกรมการข้าวโดยเฉพาะ "
                    "จึงสามารถตอบได้เฉพาะเรื่องโรคข้าวและการดูแลรักษาเท่านั้นครับ\n\n"
                    "ท่านสามารถใช้งานได้ดังนี้ครับ:\n"
                    "- ส่งรูปถ่ายใบข้าวเพื่อให้ระบบช่วยตรวจจับรอยโรค\n"
                    "- พิมพ์สอบถามเกี่ยวกับโรคข้าว เช่น 'วิธีรักษาโรคขอบใบแห้ง' หรือ 'อาการโรคใบไหม้'"
                ),
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
    [v3 Endpoint หลักสำหรับ n8n LINE Chatbot]:
    - ตรวจภาพใบข้าว
    - ถ้า >= 80%: ให้คำแนะนำการรักษาของโรคเดียวอย่างเจาะลึก
    - ถ้า 30% - 80%: แจ้ง 3 อันดับโรค + ให้แนวทางรักษาโรคอันดับ 1 + ให้จุดสังเกตเพื่อแยกแยะโรค
    - มีระบบ Multi-Model Fallback ป้องกัน Error 503 / 429
    """
    pred_res = await predict_image(file)

    if pred_res["status"] == "detected_single":
        disease_raw = pred_res["primary_disease"]["class_name_raw"] if "class_name_raw" in pred_res["primary_disease"] else pred_res["top_candidates"][0]["disease_id"]
        disease_th = pred_res["top_candidates"][0]["disease_th"]
        
        query_text = f"วิธีการรักษาและป้องกัน{disease_th}"
        contexts, _ = retrieve_rag_context(query_text, disease_id_override=disease_raw, top_k=4)
        advice = generate_rag_answer(query_text, contexts)

        return {
            "success": True,
            "version": "v3",
            "detection": pred_res,
            "advice": advice
        }

    elif pred_res["status"] == "detected_multiple" and pred_res["top_candidates"]:
        candidates = pred_res["top_candidates"]
        primary_disease = candidates[0]
        
        contexts, _ = retrieve_rag_context(
            f"วิธีการรักษาและการป้องกัน {primary_disease['disease_th']}",
            disease_id_override=primary_disease['disease_id'],
            top_k=2
        )
        for c in candidates:
            symptom_ctx, _ = retrieve_rag_context(
                f"ลักษณะอาการและจุดสังเกต {c['disease_th']}",
                disease_id_override=c['disease_id'],
                top_k=2
            )
            contexts.extend(symptom_ctx)
        
        advice = generate_multi_candidate_advice(candidates, contexts)

        return {
            "success": True,
            "version": "v3",
            "detection": pred_res,
            "advice": advice
        }

    else:
        return {
            "success": True,
            "version": "v3",
            "detection": pred_res,
            "advice": (
                "ไม่จำเป็นต้องใช้สารเคมีหรือการรักษา เนื่องจากไม่พบรอยโรคข้าวที่เข้าข่าย "
                "หากในแปลงนามีต้นข้าวที่มีอาการผิดปกติ สามารถถ่ายภาพรอยโรคบนใบข้าวในระยะใกล้ส่งมาตรวจสอบได้ตลอดเวลาครับ"
            )
        }

if __name__ == "__main__":
    import uvicorn
    # รองรับการรันทั้งจากไดเรกทอรี api และไดเรกทอรีราก
    uvicorn.run("app_v3:app", host="0.0.0.0", port=8000, reload=True)
