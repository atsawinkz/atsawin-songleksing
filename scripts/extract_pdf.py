import pymupdf
import os

project_dir = r"c:\Users\COLORFUL\Downloads\rice_leaf_diseases_v5\Project"
output_dir = r"c:\Users\COLORFUL\Downloads\rice_leaf_diseases_v5\Project"

for pdf_name in ["Chapter_01.pdf", "Chapter_02.pdf", "Chapter_03.pdf"]:
    pdf_path = os.path.join(project_dir, pdf_name)
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n---PAGE BREAK---\n"
    doc.close()
    
    txt_name = pdf_name.replace(".pdf", ".txt")
    txt_path = os.path.join(output_dir, txt_name)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Extracted {pdf_name} -> {txt_name} ({len(text)} chars)")
