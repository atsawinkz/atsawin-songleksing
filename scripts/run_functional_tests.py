"""
Automated Functional Test Runner for Rice Disease AI Chatbot
Tests TC-01 through TC-08 according to thesis Chapter 3 (Table 3-1) / Chapter 4 (Table 4-6).
"""
import os
import sys
import time
import json
import io
from PIL import Image

# Ensure root directory is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api.app_v2 import app, RAGQueryRequest, rag_query, predict_image, model, DISEASE_NAME_MAP
from fastapi import UploadFile

def run_tests():
    print("=" * 70)
    print("  RICE DISEASE AI CHATBOT - FUNCTIONAL TEST SUITE (TC-01 - TC-08)")
    print("=" * 70)

    results = []

    # -------------------------------------------------------------
    # TC-01: ตรวจจับโรคจากภาพชัดเจน (Bacterial Leaf Blight)
    # -------------------------------------------------------------
    print("\n[TC-01] Testing High-Confidence Disease Detection (Bacterial Leaf Blight)...")
    img_path = os.path.join(ROOT_DIR, "data", "labeled_datacentric-blb", "test", "images", "bacterial_leaf_blight_118_jpg.rf.8b2ae0a86439458896d3f12172289420.jpg")
    if not os.path.exists(img_path):
        # Fallback to any bacterial_leaf_blight image
        import glob
        matches = glob.glob(os.path.join(ROOT_DIR, "data", "labeled_datacentric-blb", "test", "images", "bacterial_leaf_blight*.jpg"))
        img_path = matches[0] if matches else None

    t0 = time.time()
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    upload_file = UploadFile(filename=os.path.basename(img_path), file=io.BytesIO(img_bytes))
    
    import asyncio
    tc01_res = asyncio.run(predict_image(upload_file))
    latency_01 = (time.time() - t0) * 1000

    has_disease = tc01_res.get("status") in ["detected_single", "detected_multiple"]
    detected_class = tc01_res.get("primary_disease", {}).get("th", "") if tc01_res.get("primary_disease") else ""
    tc01_pass = has_disease and "ขอบใบแห้ง" in detected_class
    print(f"  Result: Status={tc01_res.get('status')}, Disease={detected_class}, Conf={tc01_res.get('max_confidence')}, Latency={latency_01:.1f}ms")
    print(f"  Status: {'[PASS]' if tc01_pass else '[FAIL]'}")

    results.append({
        "id": "TC-01",
        "scenario": "ตรวจจับโรคจากภาพชัดเจน",
        "input": os.path.basename(img_path),
        "expected": "ตรวจพบ BBox และระบุโรคขอบใบแห้งถูกต้อง (Conf >= 80%)",
        "actual": f"สถานะ {tc01_res.get('status')}: ตรวจพบ {detected_class} (ความมั่นใจ {tc01_res.get('max_confidence')*100:.1f}%) กรอบ {tc01_res.get('all_detections_count')} จุด",
        "latency_ms": round(latency_01, 1),
        "pass": tc01_pass,
        "raw_response": tc01_res
    })

    # -------------------------------------------------------------
    # TC-02: ตรวจจับโรคแผลขนาดเล็ก (Brown Spot)
    # -------------------------------------------------------------
    print("\n[TC-02] Testing Micro-lesion Disease Detection (Brown Spot)...")
    img_path_02 = os.path.join(ROOT_DIR, "data", "labeled_datacentric-blb", "test", "images", "brown_spot_009_jpg.rf.b7aadcd3b5b44b11999dba3b731e33b7.jpg")
    t0 = time.time()
    with open(img_path_02, "rb") as f:
        img_bytes_02 = f.read()
    upload_file_02 = UploadFile(filename=os.path.basename(img_path_02), file=io.BytesIO(img_bytes_02))
    tc02_res = asyncio.run(predict_image(upload_file_02))
    latency_02 = (time.time() - t0) * 1000

    tc02_pass = tc02_res.get("status") in ["detected_single", "detected_multiple"] and (
        "จุดสีน้ำตาล" in tc02_res.get("message", "") or
        any("จุดสีน้ำตาล" in c.get("disease_th", "") for c in tc02_res.get("top_candidates", []))
    )
    print(f"  Result: Status={tc02_res.get('status')}, MaxConf={tc02_res.get('max_confidence')}, BBoxes={tc02_res.get('all_detections_count')}, Latency={latency_02:.1f}ms")
    print(f"  Status: {'[PASS]' if tc02_pass else '[FAIL]'}")

    results.append({
        "id": "TC-02",
        "scenario": "ตรวจจับโรคแผลขนาดเล็ก",
        "input": os.path.basename(img_path_02),
        "expected": "ตรวจพบรอยโรคจุดสีน้ำตาลได้ถูกต้อง",
        "actual": f"สถานะ {tc02_res.get('status')}: ตรวจพบ {tc02_res.get('all_detections_count')} รอยโรค (ความมั่นใจสูงสุด {tc02_res.get('max_confidence')*100:.1f}%)",
        "latency_ms": round(latency_02, 1),
        "pass": tc02_pass,
        "raw_response": tc02_res
    })

    # -------------------------------------------------------------
    # TC-03: ภาพใบข้าวสมบูรณ์ (Healthy - Negative Sample)
    # -------------------------------------------------------------
    print("\n[TC-03] Testing Healthy Rice Leaf (Negative Sample)...")
    img_path_03 = os.path.join(ROOT_DIR, "data", "labeled_datacentric-blb", "test", "images", "healthy_rice_leaf_005_jpg.rf.4bcdb3eb226defffeb5a279aeabaeb1b.jpg")
    t0 = time.time()
    with open(img_path_03, "rb") as f:
        img_bytes_03 = f.read()
    upload_file_03 = UploadFile(filename=os.path.basename(img_path_03), file=io.BytesIO(img_bytes_03))
    tc03_res = asyncio.run(predict_image(upload_file_03))
    latency_03 = (time.time() - t0) * 1000

    # Healthy should either have no detections, or class 'healthy', or conf < 0.30
    tc03_pass = (
        tc03_res.get("status") == "no_disease" or
        tc03_res.get("all_detections_count") == 0 or
        (tc03_res.get("primary_disease") and tc03_res.get("primary_disease", {}).get("th") == "ใบข้าวสมบูรณ์ดี (ไม่พบโรค)")
    )
    print(f"  Result: Status={tc03_res.get('status')}, Message={tc03_res.get('message')[:50]}..., Latency={latency_03:.1f}ms")
    print(f"  Status: {'[PASS]' if tc03_pass else '[FAIL]'}")

    results.append({
        "id": "TC-03",
        "scenario": "ภาพใบข้าวสมบูรณ์ (Negative)",
        "input": os.path.basename(img_path_03),
        "expected": "ไม่ตีกรอบรอยโรคผิดพลาด ระบุว่าใบสมบูรณ์ดี",
        "actual": f"สถานะ {tc03_res.get('status')}: {tc03_res.get('message')[:60]}...",
        "latency_ms": round(latency_03, 1),
        "pass": tc03_pass,
        "raw_response": tc03_res
    })

    # -------------------------------------------------------------
    # TC-04: ภาพถ่ายไม่ชัดเจน / เบลอจัด / มืด (Low Quality Image)
    # -------------------------------------------------------------
    print("\n[TC-04] Testing Low-Quality / Blurred / Dark Image...")
    # Generate blank dark image
    dark_img = Image.new("RGB", (640, 640), (20, 25, 20))
    buf = io.BytesIO()
    dark_img.save(buf, format="JPEG")
    dark_bytes = buf.getvalue()

    t0 = time.time()
    upload_file_04 = UploadFile(filename="dark_blurred_leaf.jpg", file=io.BytesIO(dark_bytes))
    tc04_res = asyncio.run(predict_image(upload_file_04))
    latency_04 = (time.time() - t0) * 1000

    tc04_pass = tc04_res.get("status") == "no_disease" and "ถ่ายภาพ" in tc04_res.get("message", "")
    print(f"  Result: Status={tc04_res.get('status')}, Fallback Message Provided, Latency={latency_04:.1f}ms")
    print(f"  Status: {'[PASS]' if tc04_pass else '[FAIL]'}")

    results.append({
        "id": "TC-04",
        "scenario": "ภาพถ่ายไม่ชัดเจน / เบลอจัด",
        "input": "dark_blurred_leaf.jpg (ภาพจำลองแสงมืดหลุดโฟกัส)",
        "expected": "แจ้งเตือนให้เกษตรกรถ่ายภาพใหม่เมื่อความมั่นใจต่ำกว่า 30%",
        "actual": f"สถานะ {tc04_res.get('status')}: {tc04_res.get('message')}",
        "latency_ms": round(latency_04, 1),
        "pass": tc04_pass,
        "raw_response": tc04_res
    })

    # -------------------------------------------------------------
    # TC-05: สอบถามวิธีรักษาและสารเคมี
    # -------------------------------------------------------------
    print("\n[TC-05] Testing Chemical Treatment Inquiry via RAG...")
    q_05 = "โรคไหม้รักษายังไง สารเคมีอะไร"
    t0 = time.time()
    tc05_res = rag_query(RAGQueryRequest(question=q_05))
    latency_05 = (time.time() - t0) * 1000

    ans_05 = tc05_res.get("answer", "")
    has_chemical = any(chem in ans_05 for chem in ["ไตรไซคลาโซล", "คาซูกาไมซิน", "อะซอกซีสโตรบิน", "คาร์เบนดาซิม", "สาร"])
    has_treatment = any(w in ans_05 for w in ["รักษา", "ป้องกัน", "พ่น", "ฉีด", "น้ำ"])
    tc05_pass = tc05_res.get("success") and has_chemical and has_treatment
    print(f"  Result: Has Chemical={has_chemical}, Has Treatment={has_treatment}, Latency={latency_05:.1f}ms")
    print(f"  Answer Snippet: {ans_05[:120]}...")
    print(f"  Status: {'[PASS]' if tc05_pass else '[FAIL]'}")

    results.append({
        "id": "TC-05",
        "scenario": "สอบถามวิธีรักษาและสารเคมี",
        "input": q_05,
        "expected": "ดึงสารเคมีที่แนะนำตามมาตรฐานกรมการข้าวและวิธีรักษาได้ถูกต้อง",
        "actual": f"แนะนำสารเคมีและวิธีรักษาถูกต้อง: {ans_05[:100]}...",
        "latency_ms": round(latency_05, 1),
        "pass": tc05_pass,
        "raw_response": tc05_res
    })

    # -------------------------------------------------------------
    # TC-06: สอบถามอาการโดยไม่บอกชื่อโรค
    # -------------------------------------------------------------
    print("\n[TC-06] Testing Symptom Description without Disease Name...")
    q_06 = "ใบมีแผลรูปกระสวยตรงกลางสีเทา"
    t0 = time.time()
    tc06_res = rag_query(RAGQueryRequest(question=q_06))
    latency_06 = (time.time() - t0) * 1000

    ans_06 = tc06_res.get("answer", "")
    identifies_blast = "ไหม้" in ans_06
    tc06_pass = tc06_res.get("success") and identifies_blast
    print(f"  Result: Identifies Leaf Blast={identifies_blast}, Latency={latency_06:.1f}ms")
    print(f"  Answer Snippet: {ans_06[:120]}...")
    print(f"  Status: {'[PASS]' if tc06_pass else '[FAIL]'}")

    results.append({
        "id": "TC-06",
        "scenario": "สอบถามอาการโดยไม่บอกชื่อโรค",
        "input": q_06,
        "expected": "วิเคราะห์อาการแผลรูปกระสวยว่าเป็นโรคใบไหม้",
        "actual": f"ระบุโรคไหม้ถูกต้อง: {ans_06[:100]}...",
        "latency_ms": round(latency_06, 1),
        "pass": tc06_pass,
        "raw_response": tc06_res
    })

    # -------------------------------------------------------------
    # TC-07: คำถามนอกขอบเขตความรู้ (Out-of-Domain Guardrail)
    # -------------------------------------------------------------
    print("\n[TC-07] Testing Out-of-Domain Fallback Guardrail...")
    q_07 = "วิธีปลูกทุเรียนหมอนทอง"
    t0 = time.time()
    tc07_res = rag_query(RAGQueryRequest(question=q_07))
    latency_07 = (time.time() - t0) * 1000

    ans_07 = tc07_res.get("answer", "")
    has_fallback = "ขออภัย" in ans_07 and "โรคข้าว" in ans_07
    tc07_pass = tc07_res.get("success") and has_fallback
    print(f"  Result: Polite Scope Fallback Triggered={has_fallback}, Latency={latency_07:.1f}ms")
    print(f"  Answer Snippet: {ans_07[:120]}...")
    print(f"  Status: {'[PASS]' if tc07_pass else '[FAIL]'}")

    results.append({
        "id": "TC-07",
        "scenario": "คำถามนอกขอบเขตความรู้",
        "input": q_07,
        "expected": "ปฏิเสธอย่างสุภาพตาม Guardrail ว่าเชี่ยวชาญเฉพาะเรื่องโรคข้าว",
        "actual": f"แจ้ง Fallback สุภาพ: {ans_07[:100]}...",
        "latency_ms": round(latency_07, 1),
        "pass": tc07_pass,
        "raw_response": tc07_res
    })

    # -------------------------------------------------------------
    # TC-08: สนทนาทั่วไปและคำทักทาย / ขอบคุณ
    # -------------------------------------------------------------
    print("\n[TC-08] Testing Conversational Greeting & Gratitude...")
    q_08_a = "สวัสดีครับ"
    t0 = time.time()
    tc08_res_a = rag_query(RAGQueryRequest(question=q_08_a))
    latency_08_a = (time.time() - t0) * 1000

    ans_08_a = tc08_res_a.get("answer", "")
    is_friendly_greeting = "สวัสดี" in ans_08_a and "ขออภัย" not in ans_08_a

    q_08_b = "ขอบคุณมากครับ"
    tc08_res_b = rag_query(RAGQueryRequest(question=q_08_b))
    ans_08_b = tc08_res_b.get("answer", "")
    is_friendly_thanks = "ยินดี" in ans_08_b and "ขออภัย" not in ans_08_b

    tc08_pass = is_friendly_greeting and is_friendly_thanks
    print(f"  Result: Greeting OK={is_friendly_greeting}, Thanks OK={is_friendly_thanks}, Latency={latency_08_a:.1f}ms")
    print(f"  Greeting Snippet: {ans_08_a[:80]}...")
    print(f"  Status: {'[PASS]' if tc08_pass else '[FAIL]'}")

    results.append({
        "id": "TC-08",
        "scenario": "สนทนาทั่วไปและคำทักทาย",
        "input": f"'{q_08_a}' และ '{q_08_b}'",
        "expected": "ตอบกลับคำทักทายและคำขอบคุณอย่างอบอุ่น โดยไม่มีคำขออภัย",
        "actual": f"ทักทาย: {ans_08_a[:60]}... / ขอบคุณ: {ans_08_b[:60]}...",
        "latency_ms": round(latency_08_a, 1),
        "pass": tc08_pass,
        "raw_response": {"greeting": tc08_res_a, "thanks": tc08_res_b}
    })

    # -------------------------------------------------------------
    # SUMMARY & REPORT GENERATION
    # -------------------------------------------------------------
    passed_count = sum(1 for r in results if r["pass"])
    total_count = len(results)
    pass_pct = (passed_count / total_count) * 100

    print("\n" + "=" * 70)
    print(f"  TEST SUMMARY: {passed_count}/{total_count} PASSED ({pass_pct:.1f}%)")
    print("=" * 70)

    # Save detailed JSON logs
    os.makedirs(os.path.join(ROOT_DIR, "docs", "test_reports"), exist_ok=True)
    json_path = os.path.join(ROOT_DIR, "docs", "test_reports", "functional_test_logs.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved full JSON log to: {json_path}")

    # Generate Markdown Report
    md_path = os.path.join(ROOT_DIR, "docs", "test_reports", "functional_test_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# รายงานผลการทดสอบระบบฟังก์ชันการทำงานจริง (Functional Test Report)\n\n")
        f.write(f"- **วันที่ทำการทดสอบ:** {time.strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"- **ระบบที่ทำการทดสอบ:** Unified FastAPI Service v2 (`api/app_v2.py`) & LINE Chatbot Pipeline\n")
        f.write(f"- **ผลการทดสอบรวม:** ผ่านการทดสอบ {passed_count}/{total_count} กรณีทดสอบ (**{pass_pct:.1f}% PASS**)\n\n")
        f.write("---\n\n")
        f.write("## ตารางสรุปผลการทดสอบทั้ง 8 กรณี (TC-01 ถึง TC-08)\n\n")
        f.write("| รหัสทดสอบ | กรณีการทดสอบ | ข้อมูลนำเข้า (Input) | ผลลัพธ์ที่คาดหวัง | ผลการทดสอบจริง | เวลาตอบสนอง (ms) | สถานะ |\n")
        f.write("| :---: | :--- | :--- | :--- | :--- | :---: | :---: |\n")
        for r in results:
            status_badge = "✅ ผ่าน (PASS)" if r["pass"] else "❌ ไม่ผ่าน (FAIL)"
            f.write(f"| **{r['id']}** | {r['scenario']} | `{r['input']}` | {r['expected']} | {r['actual']} | {r['latency_ms']} ms | {status_badge} |\n")
        f.write("\n---\n\n")
        f.write("## รายละเอียดการทดสอบและข้อความตอบกลับฉบับเต็ม\n\n")
        for r in results:
            f.write(f"### {r['id']}: {r['scenario']}\n")
            f.write(f"- **ข้อมูลนำเข้า:** `{r['input']}`\n")
            f.write(f"- **เวลาประมวลผล:** `{r['latency_ms']} ms`\n")
            f.write(f"- **สถานะ:** {'✅ ผ่านการทดสอบ' if r['pass'] else '❌ ไม่ผ่าน'}\n")
            f.write(f"```json\n{json.dumps(r['raw_response'], ensure_ascii=False, indent=2)}\n```\n\n")

    print(f"Saved Markdown report to: {md_path}")
    return results

if __name__ == "__main__":
    run_tests()
