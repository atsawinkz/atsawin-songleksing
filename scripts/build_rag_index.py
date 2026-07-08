import os
import sys
import io
import glob
import time
import math
import re
from collections import Counter
from dotenv import load_dotenv

# บังคับใช้ UTF-8 สำหรับการ Print ใน Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
from langchain_text_splitters import MarkdownTextSplitter
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import mmh3

# 1. โหลด Environment Variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("กรุณากำหนด GEMINI_API_KEY และ PINECONE_API_KEY ในไฟล์ .env")

# 2. ตั้งค่า Gemini และ Pinecone
genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "rice-knowledge"
EMBEDDING_MODEL = "models/gemini-embedding-001"
DIMENSION = 3072  # มิติของโมเดล gemini-embedding-001

# === Mapping ชื่อไฟล์ -> ชื่อโรคภาษาไทย+อังกฤษ (สำหรับ Context Enrichment) ===
DISEASE_DISPLAY_NAMES = {
    "bacterial_leaf_blight": "โรคขอบใบแห้ง (Bacterial Leaf Blight)",
    "brown_spot": "โรคใบจุดสีน้ำตาล (Brown Spot)",
    "leaf_blast": "โรคไหม้ (Rice Blast / Leaf Blast)",
    "leaf_scald": "โรคใบขีดสีน้ำตาล (Leaf Scald)",
    "sheath_blight": "โรคกาบใบแห้ง (Sheath Blight)",
}

# ============================================================
# BM25 Sparse Vector Implementation (Manual - ไม่ต้องใช้ pinecone-text)
# ============================================================
def tokenize(text):
    """ตัดคำง่ายๆ สำหรับภาษาไทย+อังกฤษ โดยแยกตามช่องว่าง, วรรคตอน"""
    # แยกตามช่องว่างและเครื่องหมายวรรคตอน แล้ว lowercase
    tokens = re.findall(r'[\w]+', text.lower())
    # กรองคำสั้นมากออก (1 ตัวอักษร) ยกเว้นตัวเลข
    return [t for t in tokens if len(t) > 1 or t.isdigit()]

def compute_bm25_sparse(text, doc_freq, num_docs, avg_dl, k1=1.5, b=0.75):
    """คำนวณ BM25 Sparse Vector สำหรับ 1 document
    คืนค่า dict: {"indices": [...], "values": [...]}"""
    tokens = tokenize(text)
    doc_len = len(tokens)
    tf = Counter(tokens)
    
    indices = []
    values = []
    
    for term, freq in tf.items():
        # Hash token เป็น integer index (ใช้ mmh3 เพื่อให้ consistent)
        token_id = abs(mmh3.hash(term)) % (2**31)
        
        # คำนวณ BM25 score
        df = doc_freq.get(term, 0)
        idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
        tf_norm = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * doc_len / avg_dl))
        score = idf * tf_norm
        
        if score > 0:
            indices.append(token_id)
            values.append(round(score, 6))
    
    return {"indices": indices, "values": values}

def build_bm25_stats(all_texts):
    """สร้างสถิติ BM25 (Document Frequency + Average Doc Length) จาก corpus"""
    doc_freq = Counter()
    total_tokens = 0
    
    for text in all_texts:
        tokens = tokenize(text)
        total_tokens += len(tokens)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_freq[token] += 1
    
    avg_dl = total_tokens / len(all_texts) if all_texts else 1
    return doc_freq, len(all_texts), avg_dl

# ============================================================
# 3. ลบ Index เก่า และสร้างใหม่ด้วย metric=dotproduct
#    (จำเป็นสำหรับ Hybrid Search: Dense + Sparse Vectors)
# ============================================================
print(f"ตรวจสอบ Index '{INDEX_NAME}'...")
if INDEX_NAME in pc.list_indexes().names():
    print(f"  ⚠️  ลบ Index เก่า '{INDEX_NAME}' เพื่อสร้างใหม่แบบ Hybrid...")
    pc.delete_index(INDEX_NAME)
    time.sleep(5)  # รอให้ลบเสร็จ

print(f"  สร้าง Index ใหม่ '{INDEX_NAME}' (metric=dotproduct สำหรับ Hybrid Search)...")
pc.create_index(
    name=INDEX_NAME,
    dimension=DIMENSION,
    metric="dotproduct",  # เปลี่ยนจาก cosine เป็น dotproduct สำหรับ Hybrid Search
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
# รอให้สร้าง index เสร็จ
while not pc.describe_index(INDEX_NAME).status['ready']:
    time.sleep(1)
print(f"  ✅ Index '{INDEX_NAME}' พร้อมใช้งาน!")

index = pc.Index(INDEX_NAME)

# ============================================================
# 4. เตรียม Text Splitter (ปรับ chunk ใหญ่ขึ้น + overlap มากขึ้น)
#    - chunk_size: 1000 -> 1500 (รักษาบริบทให้ครบถ้วน)
#    - chunk_overlap: 200 -> 300 (ป้องกันคำสำคัญถูกตัดขาด)
# ============================================================
markdown_splitter = MarkdownTextSplitter(chunk_size=1500, chunk_overlap=300)

# ============================================================
# 5. อ่านไฟล์ Markdown และทำ Chunking พร้อม Context Enrichment
# ============================================================
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge')
md_files = glob.glob(os.path.join(data_dir, "*.md"))

if not md_files:
    print(f"ไม่พบไฟล์ Markdown ในโฟลเดอร์ {data_dir}")
    exit()

all_chunks = []
all_texts_for_bm25 = []  # เก็บข้อความทั้งหมดไว้สำหรับสร้างสถิติ BM25

print("เริ่มกระบวนการ Chunking + Context Enrichment...")
for file_path in md_files:
    filename = os.path.basename(file_path)
    disease_id = filename.replace('.md', '')
    disease_display = DISEASE_DISPLAY_NAMES.get(disease_id, disease_id)
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # แบ่ง chunk
    chunks = markdown_splitter.create_documents([content])
    print(f"  📄 {filename}: ได้ {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks):
        # === Context Enrichment: แปะชื่อโรคที่หัว Chunk ทุกก้อน ===
        enriched_text = f"[{disease_display}]\n{chunk.page_content}"
        
        all_chunks.append({
            "id": f"{disease_id}_chunk_{i}",
            "text": enriched_text,
            "metadata": {
                "source": filename,
                "disease_id": disease_id,           # สำหรับ Metadata Filtering
                "disease_name": disease_display,     # สำหรับแสดงผล
                "text": enriched_text                # เก็บข้อความไว้ดึงกลับมาใช้ตอนตอบ
            }
        })
        all_texts_for_bm25.append(enriched_text)

print(f"\nรวมทั้งหมด {len(all_chunks)} chunks (Enriched)")

# ============================================================
# 6. สร้างสถิติ BM25 จาก corpus ทั้งหมด
# ============================================================
print("กำลังสร้างสถิติ BM25 จากข้อมูลโรคข้าว...")
doc_freq, num_docs, avg_dl = build_bm25_stats(all_texts_for_bm25)
print(f"  ✅ BM25 Stats: {num_docs} docs, avg_dl={avg_dl:.0f} tokens, vocab_size={len(doc_freq)}")

# บันทึกสถิติ BM25 ไว้สำหรับ query time
import json
bm25_stats_path = os.path.join(os.path.dirname(__file__), 'bm25_stats.json')
bm25_stats = {
    "doc_freq": dict(doc_freq),
    "num_docs": num_docs,
    "avg_dl": avg_dl
}
with open(bm25_stats_path, 'w', encoding='utf-8') as f:
    json.dump(bm25_stats, f, ensure_ascii=False)
print(f"  💾 BM25 stats บันทึกไว้ที่: {bm25_stats_path}")

# ============================================================
# 7. สร้าง Dense + Sparse Embeddings และ Upsert ไปยัง Pinecone
# ============================================================
print("เริ่มการสร้าง Embedding (Dense + Sparse) และอัปโหลดไปยัง Pinecone...")
batch_size = 50

for i in range(0, len(all_chunks), batch_size):
    batch_chunks = all_chunks[i:i + batch_size]
    
    texts = [c["text"] for c in batch_chunks]
    ids = [c["id"] for c in batch_chunks]
    metadatas = [c["metadata"] for c in batch_chunks]
    
    # Dense Embeddings (Gemini)
    response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=texts,
        task_type="retrieval_document",
    )
    dense_embeddings = response['embedding']
    
    # Sparse Embeddings (BM25 - คำนวณเอง)
    sparse_embeddings = [compute_bm25_sparse(t, doc_freq, num_docs, avg_dl) for t in texts]
    
    # รวมข้อมูลสำหรับ upsert แบบ Hybrid
    vectors = []
    for j in range(len(ids)):
        vectors.append({
            "id": ids[j],
            "values": dense_embeddings[j],           # Dense vector (ความหมาย)
            "sparse_values": sparse_embeddings[j],    # Sparse vector (คีย์เวิร์ด BM25)
            "metadata": metadatas[j]
        })
    
    # Upsert ลง Pinecone
    index.upsert(vectors=vectors)
    print(f"  อัปโหลด Chunks {i+1} ถึง {min(i + batch_size, len(all_chunks))} เรียบร้อยแล้ว")

print("\n✅ การสร้าง Hybrid Vector Database เสร็จสมบูรณ์!")
print(f"   Index: '{INDEX_NAME}' | Metric: dotproduct | Chunks: {len(all_chunks)}")
print(f"   Features: Dense (Gemini) + Sparse (BM25) + Context Enrichment + Metadata")
print(f"   สามารถเข้าไปตรวจสอบได้ที่ Pinecone Console")
