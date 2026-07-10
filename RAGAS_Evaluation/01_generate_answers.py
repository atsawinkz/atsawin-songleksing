import json, os, sys, argparse, io
from dotenv import load_dotenv
load_dotenv()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

C = {
    'GEMINI_API_KEY': '', 
    'PINECONE_API_KEY': '', 
    'PINECONE_INDEX': 'rice-knowledge', 
    'TOP_K': 5, 
    'GROUND_TRUTH_PATH': os.path.join(os.path.dirname(__file__), 'ragas_ground_truth.json')
}

# === Mapping ชื่อโรคสำหรับ Metadata Filter ===
DISEASE_IDS = {
    "bacterial_leaf_blight": ["ขอบใบแห้ง", "bacterial leaf blight", "blb", "ใบไหม้ขอบ"],
    "brown_spot": ["ใบจุดสีน้ำตาล", "brown spot", "จุดสีน้ำตาล", "ตาเหยี่ยว"],
    "leaf_blast": ["ไหม้", "blast", "leaf blast", "rice blast", "ไหม้คอรวง", "เน่าคอรวง", "pyricularia"],
    "leaf_scald": ["ใบขีดสีน้ำตาล", "leaf scald", "ขีดสีน้ำตาล", "ใบขีด"],
    "sheath_blight": ["กาบใบแห้ง", "sheath blight", "กาบใบ", "rhizoctonia"],
}

def init_llm(cfg):
    import google.generativeai as genai
    genai.configure(api_key=cfg.get('GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY', ''))
    return genai.GenerativeModel('gemini-3.1-flash-lite'), genai

def get_emb(emb, text):
    return emb.embed_content(model='models/gemini-embedding-001', content=text)['embedding']

def detect_disease(query):
    """สกัดชื่อโรคจากคำถามโดยใช้ Keyword Matching
    คืนค่า disease_id หรือ None ถ้าไม่เจอ"""
    query_lower = query.lower()
    for disease_id, keywords in DISEASE_IDS.items():
        for kw in keywords:
            if kw in query_lower:
                return disease_id
    return None

def retrieve(emb, query, cfg):
    """ค้นหาแบบ Hybrid (Dense + Sparse BM25) พร้อม Metadata Filter"""
    import math, re, json
    from collections import Counter
    from pinecone import Pinecone
    import mmh3
    
    pc = Pinecone(api_key=cfg['PINECONE_API_KEY'] or os.getenv('PINECONE_API_KEY', ''))
    idx = pc.Index(cfg['PINECONE_INDEX'])
    
    # Dense vector (ความหมาย)
    dense_vector = get_emb(emb, query)
    
    # Sparse vector (BM25 - คำนวณเอง)
    sparse_vector = None
    bm25_stats_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'bm25_stats.json')
    if os.path.exists(bm25_stats_path):
        with open(bm25_stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        # Tokenize query ด้วย pythainlp
        import string
        import pythainlp
        query_clean = query.translate(str.maketrans('', '', string.punctuation)).lower()
        tokens = pythainlp.word_tokenize(query_clean, keep_whitespace=False)
        tokens = [t for t in tokens if len(t) > 1 or t.isdigit()]
        tf = Counter(tokens)
        doc_len = len(tokens)
        
        doc_freq = stats['doc_freq']
        num_docs = stats['num_docs']
        avg_dl = stats['avg_dl']
        k1, b = 1.5, 0.75
        
        indices = []
        values = []
        for term, freq in tf.items():
            token_id = abs(mmh3.hash(term)) % (2**31)
            df = doc_freq.get(term, 0)
            # คำนวณ BM25 score ฝั่ง Query (ใช้ IDF * Query Freq)
            idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
            score = idf * freq
            if score > 0:
                indices.append(token_id)
                values.append(round(score, 6))
        
        if indices:
            sparse_vector = {"indices": indices, "values": values}
    
    # สกัดชื่อโรคจากคำถาม
    disease_id = detect_disease(query)
    
    # === การทำ Alpha Blending เพื่อเกลี่ยน้ำหนัก Hybrid ===
    # ปรับน้ำหนัก Sparse (BM25) กับ Dense (Vector) ให้สมดุลกัน
    # ปกติ Sparse score จะโดดไปสูงมาก เราจึงใช้ alpha=0.7 เพื่อให้น้ำหนัก Dense 70% และ Sparse 30%
    alpha = 0.7 
    dense_vector = [v * alpha for v in dense_vector]
    if sparse_vector:
        sparse_vector["values"] = [v * (1.0 - alpha) for v in sparse_vector["values"]]
    
    # เตรียม query params
    query_params = {
        "vector": dense_vector,
        "top_k": cfg['TOP_K'],
        "include_metadata": True,
    }
    
    # เพิ่ม Sparse vector ถ้ามี (Hybrid Search)
    if sparse_vector is not None:
        query_params["sparse_vector"] = sparse_vector
    
    # เพิ่ม Metadata Filter ถ้าจับชื่อโรคได้
    if disease_id:
        query_params["filter"] = {"disease_id": {"$eq": disease_id}}
    
    r = idx.query(**query_params)
    return [m['metadata']['text'] for m in r.get('matches', [])]

import time

def generate(llm, query, contexts):
    ctx = chr(10).join(contexts)
    NL = chr(10)
    prompt = ('คุณคือผู้เชี่ยวชาญด้านโรคข้าว' + NL + 'ตอบคำถามโดยอ้างอิงจาก Context ที่ให้เท่านั้น' + NL + 'ถ้าไม่พบข้อมูลใน Context ให้ตอบว่าไม่ทราบ' + NL + 'ห้ามสร้างข้อมูลเอง (No Hallucination)' + NL + NL + 'Context: ' + ctx + NL + NL + 'คำถาม: ' + query)
    
    while True:
        try:
            return llm.generate_content(prompt).text
        except Exception as e:
            if '429' in str(e) or 'Quota' in str(e) or 'exhausted' in str(e).lower():
                print('  [Rate Limit] รอ 15 วินาที...')
                time.sleep(15)
            else:
                raise e

def run_pipeline(cfg):
    with open(cfg['GROUND_TRUTH_PATH'], 'r', encoding='utf-8') as f:
        data = json.load(f)
    total = len(data['qa_pairs'])
    llm, emb = init_llm(cfg)
    results = []
    for i, qa in enumerate(data['qa_pairs']):
        q = qa['question']
        gt = qa['ground_truth']
        
        # สกัดชื่อโรค (แสดงให้เห็นว่า Filter ทำงาน)
        disease_id = detect_disease(q)
        filter_info = f" [Filter: {disease_id}]" if disease_id else " [No Filter]"
        
        ctxs = retrieve(emb, q, cfg)
        ans = generate(llm, q, ctxs)
        results.append({'question': q, 'ground_truth': gt, 'answer': ans, 'contexts': ctxs})
        print('  [' + str(i+1) + '/' + str(total) + ']' + filter_info + ' ' + q[:40] + '...')
    return results

def run_eval(results, cfg):
    import sys
    from types import ModuleType
    m = ModuleType('langchain_community.chat_models.vertexai')
    m.ChatVertexAI = type('ChatVertexAI', (object,), {})
    sys.modules['langchain_community.chat_models.vertexai'] = m

    from datasets import Dataset
    from ragas import evaluate as ev
    from ragas.metrics import context_precision, faithfulness, answer_relevancy, context_recall
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas.run_config import RunConfig
    
    # Configure Gemini for RAGAS
    os.environ['GOOGLE_API_KEY'] = cfg.get('GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY', '')
    ragas_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", max_retries=15)
    ragas_emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    ds = Dataset.from_list(results)
    
    # Run config with max_workers=1 to avoid hitting the 5 RPM rate limit too aggressively
    run_cfg = RunConfig(max_workers=1)
    
    print('เริ่มกระบวนการ RAGAS Evaluation...')
    s = ev(ds, metrics=[context_precision, faithfulness, answer_relevancy, context_recall],
           llm=ragas_llm, embeddings=ragas_emb, run_config=run_cfg, raise_exceptions=False)
    return dict(s)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='RAGAS Evaluation for Rice Disease Chatbot')
    p.add_argument('--api-key', help='Gemini API Key')
    p.add_argument('--pinecone-key', help='Pinecone API Key')
    p.add_argument('--pinecone-index', default=C['PINECONE_INDEX'], help='Pinecone index name')
    p.add_argument('--top-k', type=int, default=C['TOP_K'])
    p.add_argument('--ground-truth', default=C['GROUND_TRUTH_PATH'])
    p.add_argument('--output', default='evaluation_results')
    args = p.parse_args()
    
    os.environ.setdefault('GEMINI_API_KEY', args.api_key or '')
    
    cfg = dict(C)
    cfg['PINECONE_INDEX'] = args.pinecone_index
    cfg['PINECONE_API_KEY'] = args.pinecone_key or os.getenv('PINECONE_API_KEY', '')
    cfg['GEMINI_API_KEY'] = args.api_key or os.getenv('GEMINI_API_KEY', '')
    cfg['TOP_K'] = args.top_k
    cfg['GROUND_TRUTH_PATH'] = args.ground_truth
    os.makedirs(args.output, exist_ok=True)
    
    print('Backend: Gemini 3.1 Flash Lite | DB: Pinecone (' + cfg['PINECONE_INDEX'] + ') | Top-K: ' + str(cfg['TOP_K']))
    print('Mode: Hybrid Search (Dense + BM25) + Metadata Filter + Context Enrichment')
    res_path = os.path.join(args.output, 'eval_results.json')
    
    print('Running RAG pipeline for all questions...')
    results = run_pipeline(cfg)
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('Pipeline complete. Answers saved to ' + res_path)
    print('RAG Generation Complete')
