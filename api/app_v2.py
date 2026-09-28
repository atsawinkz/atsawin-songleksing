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
    title="Rice Disease LINE Chatbot Unified API v2",
    description="API บริการตรวจจับโรคใบข้าว เวอร์ชัน 2: รองรับเกณฑ์ความมั่นใจ 80% (>=80% แจ้งโรคเดียว, <80% แจ้ง Top 3 พร้อมแนวทางและจุดสังเกต)",
    version="2.0.0"
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
print(f"[v2] Loading RT-DETR Model from: {MODEL_PATH}")
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
    """สร้าง Dense Vector 3072 มิติผ่าน Gemini Embedding REST API"""
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
                values.append(score)

        if indices:
            sparse_vector = {"indices": indices, "values": values}

    disease_id = disease_id_override or detect_disease_from_query(query)

    query_params = {
        "vector": dense_vector,
        "top_k": top_k,
        "include_metadata": True
    }
    if sparse_vector:
        query_params["sparse_vector"] = sparse_vector
    if disease_id:
        query_params["filter"] = {"disease_id": {"$eq": disease_id}}

    res = idx.query(**query_params)
    contexts = [m['metadata']['text'] for m in res.get('matches', [])]
    return contexts, disease_id

def generate_rag_answer(query: str, contexts: List[str]) -> str:
    """ตอบคำถามข้อความทั่วไป (Text RAG)"""
    api_key = os.getenv('GEMINI_API_KEY', '')
    ctx = "\n".join(contexts)
    prompt = (
        "คุณคือผู้เชี่ยวชาญด้านโรคข้าวจากกรมการข้าว ให้คำปรึกษาแก่เกษตรกรผ่าน LINE\n"
        "คำแนะนำในการตอบ:\n"
        "1. ตอบคำถามโดยอ้างอิงจาก Context ที่ให้เท่านั้น สรุปกระชับ เข้าใจง่าย เหมาะสำหรับอ่านบน LINE\n"
        "2. ห้ามสร้างข้อมูลเอง (No Hallucination) และห้ามใช้อิโมจิ (Emoji) ในคำตอบเด็ดขาด\n"
        "3. หากคำถามเป็นเรื่องทั่วไป คำถามนอกเรื่อง หรือไม่พบข้อมูลใน Context ที่เกี่ยวข้องกับข้าวและการเกษตร:\n"
        "   - ห้ามตอบแค่ 'ไม่ทราบ' สั้นๆ เด็ดขาด\n"
        "   - ให้ตอบอย่างสุภาพว่า: 'ขออภัยครับ ระบบนี้เป็น AI ผู้ช่วยวินิจฉัยและให้คำปรึกษาเรื่องโรคข้าวจากกรมการข้าวโดยเฉพาะ จึงสามารถตอบได้เฉพาะเรื่องโรคข้าวและการดูแลรักษาเท่านั้นครับ'\n"
        "   - พร้อมให้คำแนะนำวิธีการใช้งาน เช่น สามารถส่งรูปถ่ายใบข้าวเพื่อให้ระบบช่วยตรวจจับรอยโรค หรือพิมพ์สอบถามเกี่ยวกับโรคข้าว เช่น 'วิธีรักษาโรคขอบใบแห้ง' หรือ 'อาการโรคใบไหม้'\n\n"
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

def generate_multi_candidate_advice(candidates: List[dict], contexts: List[str]) -> str:
    """
    [v2 พิเศษ] สร้างคำแนะนำสำหรับกรณีความมั่นใจต่ำกว่า 80%:
    1. ให้คำแนะนำการรักษาโรคอันดับ 1 เป็นหลัก
    2. ให้จุดสังเกตรอยโรคในแปลงเพื่อช่วยแยกแยะโรคที่เข้าข่าย
    """
    api_key = os.getenv('GEMINI_API_KEY', '')
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
        "ห้ามแต่งข้อมูลเท็จ ตอบตาม Context ที่ให้เท่านั้น และห้ามใส่อิโมจิ (Emoji) ในข้อความเด็ดขาด\n\n"
        f"Context:\n{ctx}"
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


# === 4. Endpoints (Version 2) ===
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "version": "2.0.0",
        "mode": "Dual-Confidence Threshold (80% High Conf / Top-3 Candidates)",
        "model_loaded": MODEL_PATH,
        "service": "Rice Disease LINE Chatbot API v2"
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    [v2] วิเคราะห์รูปภาพ:
    - ถ้าความมั่นใจสูงสุด >= 80% -> สรุปเป็นโรคเดียว (High Confidence)
    - ถ้า 30% <= ความมั่นใจ < 80% -> รวบรวม Top 3 โรคที่มีโอกาสเป็นมากที่สุด พร้อม %
    - ถ้าความมั่นใจ < 30% -> ไม่พบรอยโรคหรือภาพไม่ชัดเจน
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="ไฟล์ที่อัปโหลดไม่ใช่รูปภาพที่ถูกต้อง (รองรับ JPEG, PNG)")

    try:
        # ใช้ conf ต่ำเพื่อเก็บ candidate ทุกโรคที่โมเดลมองเห็น
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

        # เรียงลำดับโรคตามความมั่นใจจากมากไปน้อย (ตัดคลาส healthy ออก)
        sorted_diseases = sorted(
            [(k, v) for k, v in class_scores.items() if k != "healthy"],
            key=lambda x: x[1],
            reverse=True
        )

        primary_raw, max_conf = sorted_diseases[0] if sorted_diseases else ("healthy", 0.0)
        primary_info = DISEASE_NAME_MAP.get(primary_raw, {"th": primary_raw, "en": primary_raw, "scientific": "Unknown"})

        # === ตัดสินใจตามเกณฑ์ 80% ===
        if max_conf >= 0.80:
            # กรณีที่ 1: มั่นใจสูงมากตั้งแต่ 80% ขึ้นไป -> รายงานโรคเดียว
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
            # กรณีที่ 2: ความมั่นใจระหว่าง 30% - 79.9% -> รายงาน Top 3 โรค
            status = "detected_multiple"
            is_high_confidence = False

            top_candidates = []
            for rank, (d_name, score) in enumerate(sorted_diseases[:3], 1):
                if score >= 0.005:  # มีคะแนนอย่างน้อย 0.5%
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
            # กรณีที่ 3: ต่ำกว่า 30%
            status = "no_disease"
            is_high_confidence = False
            top_candidates = []
            message = "ไม่พบรอยโรคพืชที่ชัดเจน หรือใบข้าวสมบูรณ์ดี (ความมั่นใจต่ำกว่า 30%) หากสงสัยว่าเป็นโรค กรุณาถ่ายภาพระยะใกล้ที่มีแสงชัดเจนเข้ามาใหม่อีกครั้งครับ"

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
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการประมวลผลรูปภาพ (v2): {str(e)}")

@app.post("/rag_query")
def rag_query(req: RAGQueryRequest):
    try:
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
    [v2 Endpoint หลักสำหรับ n8n LINE Chatbot]:
    - ตรวจภาพใบข้าว
    - ถ้า >= 80%: ให้คำแนะนำการรักษาของโรคเดียวอย่างเจาะลึก
    - ถ้า 30% - 80%: แจ้ง 3 อันดับโรค + ให้แนวทางรักษาโรคอันดับ 1 + ให้จุดสังเกตเพื่อแยกแยะโรค
    """
    pred_res = await predict_image(file)

    if pred_res["status"] == "detected_single":
        # มั่นใจเกิน 80%
        disease_raw = pred_res["primary_disease"]["class_name_raw"] if "class_name_raw" in pred_res["primary_disease"] else pred_res["top_candidates"][0]["disease_id"]
        disease_th = pred_res["top_candidates"][0]["disease_th"]
        
        query_text = f"วิธีการรักษาและป้องกัน{disease_th}"
        contexts, _ = retrieve_rag_context(query_text, disease_id_override=disease_raw, top_k=4)
        advice = generate_rag_answer(query_text, contexts)

        return {
            "success": True,
            "version": "v2",
            "detection": pred_res,
            "advice": advice
        }

    elif pred_res["status"] == "detected_multiple" and pred_res["top_candidates"]:
        # มั่นใจระหว่าง 30% - 80% (แสดง 3 โรค + คำแนะนำแบบรวม)
        candidates = pred_res["top_candidates"]
        primary_disease = candidates[0]
        
        # รวบรวม Context ให้ครบถ้วน:
        # 1. ดึงวิธีรักษาของโรคอันดับ 1
        contexts, _ = retrieve_rag_context(
            f"วิธีการรักษาและการป้องกัน {primary_disease['disease_th']}",
            disease_id_override=primary_disease['disease_id'],
            top_k=2
        )
        # 2. ดึงลักษณะอาการและจุดสังเกตของทุกโรคที่เข้าข่าย
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
            "version": "v2",
            "detection": pred_res,
            "advice": advice
        }

    else:
        # ไม่พบโรค หรือมั่นใจต่ำกว่า 30%
        return {
            "success": True,
            "version": "v2",
            "detection": pred_res,
            "advice": "ไม่จำเป็นต้องใช้สารเคมีหรือวิธีการรักษา เนื่องจากไม่พบรอยโรคที่เข้าข่าย หากต้นข้าวมีอาการผิดปกติ สามารถถ่ายภาพจุดที่เกิดอาการในระยะใกล้ส่งมาตรวจสอบใหม่ได้ครับ"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_v2:app", host="0.0.0.0", port=8000, reload=True)
