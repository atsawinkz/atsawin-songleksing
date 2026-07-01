import json, os, sys, io
from dotenv import load_dotenv
load_dotenv()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

def run_eval(results):
    # Mock VertexAI to prevent Langchain error on Python 3.13
    import sys
    from types import ModuleType
    m = ModuleType('langchain_community.chat_models.vertexai')
    m.ChatVertexAI = type('ChatVertexAI', (object,), {})
    sys.modules['langchain_community.chat_models.vertexai'] = m

    from datasets import Dataset
    from ragas import evaluate as ev
    from ragas.metrics import context_precision, faithfulness, answer_relevancy, context_recall
    answer_relevancy.strictness = 1
    
    # 1. Initialize Embeddings (still using Gemini because it's free and we used it for index)
    import google.generativeai as genai
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    genai.configure(api_key=os.getenv('GEMINI_API_KEY', ''))
    ragas_emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 2. Initialize the Judge LLM (OpenCode Zen / DeepSeek V4 Flash Free)
    from langchain_openai import ChatOpenAI
    api_key = os.getenv('OPENCODE_ZEN_API_KEY', '')
    base_url = os.getenv('OPENCODE_ZEN_BASE_URL', 'https://opencode.ai/zen/v1')
    
    ragas_llm = ChatOpenAI(
        model="deepseek-v4-flash-free", 
        api_key=api_key, 
        base_url=base_url,
        max_retries=5
    )
    
    ds = Dataset.from_list(results)
    
    from ragas.run_config import RunConfig
    # Use max_workers=2 to speed it up a bit since OpenCode Zen might have higher limits
    run_cfg = RunConfig(max_workers=2)
    
    print('เริ่มกระบวนการ RAGAS Evaluation')
    s = ev(ds, metrics=[context_precision, faithfulness, answer_relevancy, context_recall],
           llm=ragas_llm, embeddings=ragas_emb, run_config=run_cfg, raise_exceptions=False)
    
    # ดึงค่าคะแนนโดยไม่ใช้ dict(s) เพื่อหลีกเลี่ยง KeyError
    scores = {}
    try:
        # บันทึกรายละเอียดแบบรายข้อ
        df = s.to_pandas()
        df.to_csv(os.path.join(os.path.dirname(__file__), 'evaluation_results', 'eval_details.csv'), index=False, encoding='utf-8-sig')
        
        # คำนวณค่าเฉลี่ยของแต่ละ Metric
        for m in ['context_precision', 'faithfulness', 'answer_relevancy', 'context_recall']:
            if m in df.columns:
                scores[m] = float(df[m].mean())
    except Exception as e:
        print(f"ข้อผิดพลาดระหว่างดึงคะแนน: {e}")
        # Fallback เผื่อ Ragas เวอร์ชันเก่า
        for m in ['context_precision', 'faithfulness', 'answer_relevancy', 'context_recall']:
            try: scores[m] = s[m]
            except: pass

    return scores

if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__), 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    res_path = os.path.join(output_dir, 'eval_results.json')
    if not os.path.exists(res_path):
        print("Error: ไม่พบไฟล์กระดาษคำตอบที่ " + res_path)
        sys.exit(1)
        
    print('กำลังโหลดไฟล์กระดาษคำตอบจาก ' + res_path)
    with open(res_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
        
    print(f"กำลังรันประเมินผลชุดใหญ่ทั้งหมด {len(results)} ข้อ")
        
    scores = run_eval(results)
    
    score_path = os.path.join(output_dir, 'eval_scores.json')
    with open(score_path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
        
    print('ตรวจข้อสอบเสร็จสิ้น! บันทึกคะแนนลงใน ' + score_path)
    print('สรุปผลคะแนน RAGAS')
    for k, v in scores.items():
        print('  ' + k + ': ' + str(round(v, 4)))
