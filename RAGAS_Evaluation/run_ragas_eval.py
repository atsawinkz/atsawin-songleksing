import json, os, sys, argparse

C = {'LLM_BACKEND': 'gemini', 'GEMINI_API_KEY': '', 'OPENAI_API_KEY': '', 'PINECONE_API_KEY': '', 'DB_TYPE': 'chroma', 'CHROMA_DB_PATH': '../Project/chroma_db', 'PINECONE_INDEX': 'rice-knowledge', 'TOP_K': 3, 'GROUND_TRUTH_PATH': 'ragas_ground_truth.json'}

def init_llm(cfg):
    b = cfg['LLM_BACKEND']
    if b == 'gemini':
        import google.generativeai as genai
        genai.configure(api_key=cfg.get('GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY', ''))
        return genai.GenerativeModel('gemini-2.5-flash'), genai
    elif b == 'openai':
        from openai import OpenAI
        client = OpenAI(api_key=cfg.get('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY', ''))
        return client, client.embeddings.create
    else:
        from openai import OpenAI as O
        client = O(base_url='http://localhost:11434/v1', api_key='ollama')
        return client, client.embeddings.create

def get_emb(llm, emb, text, cfg):
    if cfg['LLM_BACKEND'] == 'gemini':
        return emb.embed_content(model='models/embedding-001', content=text)['embedding']
    return emb(input=text, model='text-embedding-3-small').data[0].embedding

def retrieve_chroma(llm, emb, query, cfg):
    import chromadb
    cl = chromadb.PersistentClient(path=cfg['CHROMA_DB_PATH'])
    col = cl.get_or_create_collection('rice_knowledge')
    qe = get_emb(llm, emb, query, cfg)
    r = col.query(query_embeddings=[qe], n_results=cfg['TOP_K'])
    return r['documents'][0] if r['documents'] else ['']

def retrieve_pinecone(llm, emb, query, cfg):
    from pinecone import Pinecone
    pc = Pinecone(api_key=cfg['PINECONE_API_KEY'] or os.getenv('PINECONE_API_KEY', ''))
    idx = pc.Index(cfg['PINECONE_INDEX'])
    qe = get_emb(llm, emb, query, cfg)
    r = idx.query(vector=qe, top_k=cfg['TOP_K'], include_metadata=True)
    return [m['metadata']['text'] for m in r.get('matches', [])]

def retrieve(llm, emb, query, cfg):
    if cfg['DB_TYPE'] == 'pinecone':
        return retrieve_pinecone(llm, emb, query, cfg)
    return retrieve_chroma(llm, emb, query, cfg)

def generate(llm, query, contexts, cfg):
    ctx = chr(10).join(contexts)
    NL = chr(10)
    prompt = ('คุณคือผู้เชี่ยวชาญด้านโรคข้าว' + NL + 'ตอบคำถามโดยอ้างอิงจาก Context ที่ให้เท่านั้น' + NL + 'ถ้าไม่พบข้อมูลใน Context ให้ตอบว่าไม่ทราบ' + NL + 'ห้ามสร้างข้อมูลเอง (No Hallucination)' + NL + NL + 'Context: ' + ctx + NL + NL + 'คำถาม: ' + query)
    if cfg['LLM_BACKEND'] == 'gemini':
        return llm.generate_content(prompt).text
    r = llm.chat.completions.create(model=cfg.get('OLLAMA_MODEL', 'llama3.2'), messages=[{'role': 'user', 'content': prompt}])
    return r.choices[0].message.content

def run_pipeline(cfg):
    with open(cfg['GROUND_TRUTH_PATH'], 'r', encoding='utf-8') as f:
        data = json.load(f)
    total = len(data['qa_pairs'])
    llm, emb = init_llm(cfg)
    results = []
    for i, qa in enumerate(data['qa_pairs']):
        q = qa['question']
        gt = qa['ground_truth']
        ctxs = retrieve(llm, emb, q, cfg)
        ans = generate(llm, q, ctxs, cfg)
        results.append({'question': q, 'ground_truth': gt, 'answer': ans, 'contexts': ctxs})
        print('  [' + str(i+1) + '/' + str(total) + '] ' + q[:40] + '...')
    return results

def run_eval(results):
    from datasets import Dataset
    from ragas import evaluate as ev
    from ragas.metrics import context_precision, faithfulness, answer_relevance, context_recall
    ds = Dataset.from_list(results)
    s = ev(ds, metrics=[context_precision, faithfulness, answer_relevance, context_recall])
    return dict(s)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='RAGAS Evaluation for Rice Disease Chatbot')
    p.add_argument('--backend', choices=['gemini', 'openai', 'ollama'], default=C['LLM_BACKEND'])
    p.add_argument('--api-key')
    p.add_argument('--db-type', choices=['chroma', 'pinecone'], default=C['DB_TYPE'], help='Vector DB type')
    p.add_argument('--db-path', default=C['CHROMA_DB_PATH'], help='ChromaDB path (if db-type=chroma)')
    p.add_argument('--pinecone-index', default=C['PINECONE_INDEX'], help='Pinecone index name (if db-type=pinecone)')
    p.add_argument('--top-k', type=int, default=C['TOP_K'])
    p.add_argument('--ground-truth', default=C['GROUND_TRUTH_PATH'])
    p.add_argument('--output', default='output')
    args = p.parse_args()
    os.environ.setdefault('GEMINI_API_KEY', args.api_key or '')
    os.environ.setdefault('OPENAI_API_KEY', args.api_key or '')
    cfg = dict(C)
    cfg['LLM_BACKEND'] = args.backend
    cfg['DB_TYPE'] = args.db_type
    cfg['CHROMA_DB_PATH'] = args.db_path
    cfg['PINECONE_INDEX'] = args.pinecone_index
    cfg['PINECONE_API_KEY'] = args.api_key or os.getenv('PINECONE_API_KEY', '')
    cfg['TOP_K'] = args.top_k
    cfg['GROUND_TRUTH_PATH'] = args.ground_truth
    os.makedirs(args.output, exist_ok=True)
    db_label = cfg['DB_TYPE'] + ': ' + (cfg['CHROMA_DB_PATH'] if cfg['DB_TYPE'] == 'chroma' else cfg['PINECONE_INDEX'])
    print('Backend: ' + cfg['LLM_BACKEND'] + ' | DB: ' + db_label + ' | Top-K: ' + str(cfg['TOP_K']))
    print('Running RAG pipeline...')
    results = run_pipeline(cfg)
    rp = os.path.join(args.output, 'eval_results.json')
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('Results saved: ' + rp)
    print('Running RAGAS evaluation...')
    score = run_eval(results)
    sp = os.path.join(args.output, 'eval_scores.json')
    with open(sp, 'w', encoding='utf-8') as f:
        json.dump(score, f, ensure_ascii=False, indent=2)
    print('Scores saved: ' + sp)
    print('=== RAGAS Evaluation Complete ===')
    for k, v in score.items():
        print('  ' + k + ': ' + str(round(v, 4)))
