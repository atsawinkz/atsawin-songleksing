import json, os, sys, io
import pandas as pd
from dotenv import load_dotenv

# บังคับการเข้ารหัสเพื่อแก้ปัญหาปริ๊นท์ภาษาไทยบน Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

def fix_evaluation():
    # โหลด Environment Variables
    load_dotenv()
    
    # 1. โหลดข้อมูลเดิม
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, 'evaluation_results')
    
    csv_path = os.path.join(results_dir, 'eval_details.csv')
    json_path = os.path.join(results_dir, 'eval_results.json')
    
    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        print("ไม่พบไฟล์ผลลัพธ์เดิม กรุณาตรวจสอบว่ามีไฟล์ eval_details.csv และ eval_results.json อยู่ในโฟลเดอร์ evaluation_results")
        return

    print("กำลังโหลดไฟล์ผลลัพธ์เดิม...")
    df = pd.read_csv(csv_path)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        all_results = json.load(f)
        
    metrics = ['context_precision', 'faithfulness', 'answer_relevancy', 'context_recall']
    
    # 2. ค้นหาข้อที่พัง (คะแนนเป็น NaN ใน metric ใดๆ)
    # ค้นหา index ของแถวที่มีค่าว่าง
    failed_indices = df[df[metrics].isna().any(axis=1)].index.tolist()
    
    if not failed_indices:
        print("ตรวจสอบแล้ว ไม่พบข้อที่พัง (ไม่มีค่า NaN)! ไม่ต้องซ่อมแซมอะไรครับ")
        return
        
    print(f"พบข้อที่พัง (Timeout/Error) จำนวน {len(failed_indices)} ข้อ ได้แก่ข้อที่: {failed_indices}")
    print("กำลังเตรียมชุดข้อมูลเพื่อนำไปรันซ่อมแซมเฉพาะจุด...")
    
    # 3. สกัดเฉพาะข้อที่พังออกมา
    failed_results = [all_results[i] for i in failed_indices]
    
    # 4. ตั้งค่า RAGAS ใหม่
    # Mock VertexAI to prevent Langchain error on Python 3.13
    import sys
    from types import ModuleType
    m = ModuleType('langchain_community.chat_models.vertexai')
    m.ChatVertexAI = type('ChatVertexAI', (object,), {})
    sys.modules['langchain_community.chat_models.vertexai'] = m

    from datasets import Dataset
    from ragas import evaluate as ev
    from ragas.metrics import context_precision as cp, faithfulness as ft, answer_relevancy as ar, context_recall as cr
    ar.strictness = 1
    
    import google.generativeai as genai
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    genai.configure(api_key=os.getenv('GEMINI_API_KEY', ''))
    ragas_emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    from langchain_openai import ChatOpenAI
    api_key = os.getenv('OPENCODE_ZEN_API_KEY', '')
    base_url = os.getenv('OPENCODE_ZEN_BASE_URL', 'https://opencode.ai/zen/v1')
    
    # ใช้ API Key ใหม่ที่เปลี่ยนอีเมล
    print(f"กำลังเชื่อมต่อกับ OpenCodeZen")
    ragas_llm = ChatOpenAI(
        model="deepseek-v4-flash-free", 
        api_key=api_key, 
        base_url=base_url,
        max_retries=5
    )
    
    ds = Dataset.from_list(failed_results)
    
    from ragas.run_config import RunConfig
    run_cfg = RunConfig(max_workers=1) # ลด workers ลงเพื่อป้องกัน rate limit
    
    # 5. รันซ่อมแซม
    print(f"กำลังส่งให้ DeepSeek ตรวจข้อสอบซ่อม จำนวน {len(failed_indices)} ข้อ...")
    fixed_eval = ev(ds, metrics=[cp, ft, ar, cr], llm=ragas_llm, embeddings=ragas_emb, run_config=run_cfg, raise_exceptions=False)
    
    fixed_df = fixed_eval.to_pandas()
    
    # 6. เอาคะแนนที่ซ่อมเสร็จไปอุดรูรั่วใน DataFrame เดิม
    print("ตรวจซ่อมเสร็จสิ้น! กำลังประกอบร่างข้อมูล...")
    for idx, new_row in fixed_df.iterrows():
        original_idx = failed_indices[idx]
        for m in metrics:
            if m in new_row and pd.notna(new_row[m]):
                df.at[original_idx, m] = new_row[m]
                
    # 7. บันทึกผล
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"บันทึกไฟล์ {csv_path} อัปเดตทับของเดิมเรียบร้อยแล้ว")
    
    # สรุปคะแนนเฉลี่ยใหม่
    scores = {}
    for m in metrics:
        if m in df.columns:
            scores[m] = float(df[m].mean())
            
    score_path = os.path.join(results_dir, 'eval_scores.json')
    with open(score_path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
        
    print('สรุปผลคะแนน RAGAS หลังจากซ่อมแซมเสร็จสิ้น')
    for k, v in scores.items():
        print(f"  {k}: {round(v, 4)}")
        
if __name__ == "__main__":
    fix_evaluation()
