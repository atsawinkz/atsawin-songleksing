# RAGAS Evaluation - Rice Disease LINE Chatbot

ไฟล์นี้ใช้สําหรับประเมิน RAG Pipeline ด้วย RAGAS framework

## สิ่งที่ต้องติดตั้ง

```bash
pip install ragas datasets chromadb google-generativeai openai
```

สําหรับ Ollama (optional, local LLM):
```bash
pip install openai
# ติดตั้ง Ollama: https://ollama.com/
ollama pull llama3.2
ollama pull nomic-embed-text
```

## วิธีใช้งาน

### 1. ไปที่โฟลเดอร์ RAGAS_Evaluation
```bash
cd RAGAS_Evaluation
```

### 2. ใช้ Gemini (default)
```bash
set GEMINI_API_KEY=YOUR_API_KEY
python run_ragas_eval.py
```

### 3. ใช้ OpenAI
```bash
set OPENAI_API_KEY=YOUR_API_KEY
python run_ragas_eval.py --backend openai
```

### 4. ใช้ Ollama Local (ฟรี)
```bash
python run_ragas_eval.py --backend ollama
```

## Options

| Flag | Default | Description |
|---|---|---|
| --backend | gemini | LLM backend: gemini, openai, ollama |
| --api-key | env var | API key (override env var) |
| --db-path | ../Project/chroma_db | Path to ChromaDB |
| --top-k | 3 | Number of chunks to retrieve |
| --ground-truth | ragas_ground_truth.json | Ground truth Q&A file |
| --output | output | Output directory |

## Output Files

output/eval_results.json   - Results from RAG pipeline (80 Q&A)
output/eval_scores.json    - RAGAS metrics scores

## RAGAS Metrics

| Metric | Range | Meaning |
|---|---|---|
| context_precision | 0-1 | ความแม่นยําในการดึง Context ที่เกี่ยวข้อง |
| faithfulness | 0-1 | ความถูกต้องของคําตอบเทียบกับ Context |
| answer_relevance | 0-1 | ความตรงประเด็นของคําตอบต่อคําถาม |
| context_recall | 0-1 | ความสามารถในการดึง Context ที่จําเป็นครบ |

## Troubleshooting

- ModuleNotFoundError: รัน pip install ragas datasets chromadb google-generativeai openai
- No such file or directory: รันจากโฟลเดอร์ RAGAS_Evaluation/ เท่านั้น
- ChromaDB not found: ต้องสร้าง Vector Database ก่อน (Project Plan Phase 5)
- API key error: ตั้ง environment variable หรือใช้ --api-key

## หมายเหตุ

- สคริปต์นี้ออกแบบให้รันจากโฟลเดอร์ RAGAS_Evaluation/ เท่านั้น
- Ground Truth มี 80 คําถาม-คําตอบ 5 หมวด
  - symptom_identification (17)
  - treatment (16)
  - prevention (15)
  - chemical (9)
  - general (23)
- 3 ระดับความยาก: easy(38), medium(35), hard(7)
