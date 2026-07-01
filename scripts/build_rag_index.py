import os
import sys
import io
import glob
import time
from dotenv import load_dotenv

# บังคับใช้ UTF-8 สำหรับการ Print ใน Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from langchain_text_splitters import MarkdownTextSplitter
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

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

# ตรวจสอบและสร้าง Pinecone Index ถ้ายั้งไม่มี
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating Pinecone index: '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1") # เลือก region ตามที่ตั้งค่าใน Pinecone
    )
    # รอให้สร้าง index เสร็จ
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)

index = pc.Index(INDEX_NAME)

# 3. เตรียม Text Splitter (1000 chars, 200 overlap)
markdown_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)

# 4. อ่านไฟล์ Markdown และทำ Chunking
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge')
md_files = glob.glob(os.path.join(data_dir, "*.md"))

if not md_files:
    print(f"ไม่พบไฟล์ Markdown ในโฟลเดอร์ {data_dir}")
    exit()

all_chunks = []
print("เริ่มกระบวนการ Chunking...")
for file_path in md_files:
    filename = os.path.basename(file_path)
    disease_id = filename.replace('.md', '')
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # แบ่ง chunk
    chunks = markdown_splitter.create_documents([content])
    print(f" - {filename}: ได้ {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "id": f"{disease_id}_chunk_{i}",
            "text": chunk.page_content,
            "metadata": {
                "source": filename,
                "disease_id": disease_id,
                "text": chunk.page_content  # เก็บ text ไว้ใน metadata เพื่อให้ดึงกลับมาใช้ตอนตอบคำถาม
            }
        })

print(f"รวมทั้งหมด {len(all_chunks)} chunks")

# 5. สร้าง Embedding และ Upsert ไปยัง Pinecone
print("เริ่มการสร้าง Embedding และอัปโหลดไปยัง Pinecone...")
batch_size = 50  # ขนาด Batch ในการ upsert

for i in range(0, len(all_chunks), batch_size):
    batch_chunks = all_chunks[i:i + batch_size]
    
    texts = [c["text"] for c in batch_chunks]
    ids = [c["id"] for c in batch_chunks]
    metadatas = [c["metadata"] for c in batch_chunks]
    
    # เรียกใช้ Gemini API สร้าง Embeddings
    response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=texts,
        task_type="retrieval_document",
    )
    embeddings = response['embedding']
    
    # รวมข้อมูลสำหรับ upsert: (id, embedding, metadata)
    records = zip(ids, embeddings, metadatas)
    
    # Upsert ลง Pinecone
    index.upsert(vectors=list(records))
    print(f"อัปโหลด Chunks {i+1} ถึง {min(i + batch_size, len(all_chunks))} เรียบร้อยแล้ว")

print("\nการสร้าง Vector Database เสร็จสมบูรณ์! ✅")
print(f"สามารถเข้าไปตรวจสอบ Index '{INDEX_NAME}' ได้ที่ Pinecone Console")
