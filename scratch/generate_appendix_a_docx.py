import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def create_appendix_a():
    doc = docx.Document()
    
    # 1. Setup Margins (KU Thesis Standard: Left 1.5", Top 1.5", Right 1.0", Bottom 1.0")
    section = doc.sections[0]
    section.page_width = Inches(8.27)   # A4
    section.page_height = Inches(11.69) # A4
    section.top_margin = Inches(1.5)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.5)
    section.right_margin = Inches(1.0)
    
    # Normal style setup
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'TH Sarabun New'
    normal_style.font.size = Pt(16)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(4)
    normal_style.paragraph_format.space_before = Pt(0)
    
    # Helper to set run font
    def set_font(run, font_name='TH Sarabun New', size=16, bold=False, italic=False, color=RGBColor(0, 0, 0)):
        run.font.name = font_name
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
        run.font.color.rgb = color
        # Required for Thai fonts in Word
        rPr = run._r.get_or_add_rPr()
        rFonts = OxmlElement('w:rFonts')
        rFonts.set(qn('w:ascii'), font_name)
        rFonts.set(qn('w:hAnsi'), font_name)
        rFonts.set(qn('w:cs'), font_name)
        rPr.append(rFonts)

    def add_para(text, align=WD_ALIGN_PARAGRAPH.LEFT, bold=False, size=16, indent=0, space_before=0, space_after=4):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.space_after = Pt(space_after)
        if indent > 0:
            p.paragraph_format.first_line_indent = Inches(indent)
        run = p.add_run(text)
        set_font(run, size=size, bold=bold)
        return p

    def add_placeholder_box(fig_no, fig_title, fig_desc):
        # 1-cell table as a bordered placeholder box
        table = doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.cell(0, 0)
        cell.width = Inches(5.7)
        
        # Border
        tcPr = cell._element.get_or_add_tcPr()
        tcBorders = OxmlElement('w:tcBorders')
        for border_name in ['top', 'left', 'bottom', 'right']:
            edge = OxmlElement(f'w:{border_name}')
            edge.set(qn('w:val'), 'dashed')
            edge.set(qn('w:sz'), '8')
            edge.set(qn('w:space'), '0')
            edge.set(qn('w:color'), 'A0AEC0')
            tcBorders.append(edge)
        tcPr.append(tcBorders)
        
        # Shading
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), 'F8FAFC')
        tcPr.append(shd)
        
        # Content inside box
        p_box = cell.paragraphs[0]
        p_box.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_box.paragraph_format.space_before = Pt(28)
        p_box.paragraph_format.space_after = Pt(28)
        
        r1 = p_box.add_run("[ พื้นที่สำหรับวางรูปภาพ ]\n")
        set_font(r1, size=15, bold=True, color=RGBColor(71, 85, 105))
        
        r2 = p_box.add_run("(คลิกที่นี่หรือลบข้อความนี้เพื่อแทรกรูปภาพ screenshot)")
        set_font(r2, size=13, italic=True, color=RGBColor(148, 163, 184))
        
        # Figure caption
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(6)
        p_cap.paragraph_format.space_after = Pt(2)
        r_cap_label = p_cap.add_run(f"ภาพที่ {fig_no} ")
        set_font(r_cap_label, size=16, bold=True)
        r_cap_title = p_cap.add_run(fig_title)
        set_font(r_cap_title, size=16, bold=False)
        
        # Explanatory note
        p_note = doc.add_paragraph()
        p_note.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_note.paragraph_format.space_before = Pt(0)
        p_note.paragraph_format.space_after = Pt(12)
        r_note = p_note.add_run(f"(คำอธิบายรูป: {fig_desc})")
        set_font(r_note, size=14, italic=True, color=RGBColor(100, 116, 139))

    # ==================== PAGE 1: COVER PAGE ====================
    for _ in range(8):
        doc.add_paragraph()
        
    p_cov1 = doc.add_paragraph()
    p_cov1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cov1 = p_cov1.add_run("ภาคผนวก ก")
    set_font(r_cov1, size=18, bold=True)
    
    p_cov2 = doc.add_paragraph()
    p_cov2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cov2 = p_cov2.add_run("คู่มือการใช้งานระบบแชทบอทวิเคราะห์โรคข้าว")
    set_font(r_cov2, size=18, bold=True)
    
    doc.add_page_break()
    
    # ==================== PAGE 2: MANUAL BODY ====================
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(14)
    r_title = p_title.add_run("คู่มือการใช้งานระบบแชทบอทวิเคราะห์โรคข้าว")
    set_font(r_title, size=18, bold=True)
    
    add_para(
        "คู่มือการใช้งานระบบแชทบอทวิเคราะห์โรคข้าว แบ่งออกเป็น 2 ส่วน ดังนี้",
        indent=0.5, space_after=2
    )
    add_para("1) ส่วนสำหรับผู้ใช้งานทั่วไป (เกษตรกร): การใช้งานแชทบอทผ่านแอปพลิเคชัน LINE", indent=0.75, space_after=2)
    add_para("2) ส่วนสำหรับผู้ดูแลระบบ (Admin): การจัดการระบบทำงานอัตโนมัติ (n8n) และฐานข้อมูลความรู้ (Pinecone)", indent=0.75, space_after=12)
    
    # -------------------- ก.1 --------------------
    add_para("ก.1 ส่วนสำหรับเกษตรกรและผู้ใช้งานทั่วไป", bold=True, size=16, space_before=6, space_after=4)
    
    # ก.1.1
    add_para("ก.1.1 การเพิ่มเพื่อนและเริ่มต้นใช้งานแชทบอท", bold=True, size=16, indent=0.25, space_after=3)
    add_para(
        "ผู้ใช้สามารถเริ่มต้นใช้งานได้ง่าย ๆ โดยการเพิ่มเพื่อนใน LINE ผ่าน 2 วิธี ได้แก่ การสแกนคิวอาร์โค้ด (QR Code) ของแชทบอท หรือค้นหาด้วย LINE ID (@บอทวินิจฉัยโรคข้าว) เมื่อกดเพิ่มเพื่อนและเปิดเข้าไปในห้องแชท แชทบอทจะส่งข้อความทักทายต้อนรับ แนะนำตัว และบอกวิธีใช้งานเบื้องต้น พร้อมกับแสดงเมนูหลักที่ด้านล่างของหน้าจอทันที",
        indent=0.5, space_after=8
    )
    add_placeholder_box(
        "ก-1",
        "การเพิ่มเพื่อนและข้อความต้อนรับของแชทบอท",
        "หน้าจอ LINE ตอนกดเพิ่มเพื่อน และมีข้อความต้อนรับพร้อมเมนูด้านล่างปรากฏขึ้น"
    )
    
    # ก.1.2
    add_para("ก.1.2 การใช้งานเมนูหลัก (Rich Menu)", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "ที่แถบด้านล่างของหน้าจอแชทจะมีเมนูลัดให้กดใช้งานได้สะดวก แบ่งออกเป็น 3 ปุ่มหลัก ดังนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) ตรวจโรคข้าว: แนะนำวิธีการส่งรูปถ่ายใบข้าวเพื่อให้บอทช่วยตรวจ", indent=0.75, space_after=2)
    add_para("2) วิธีการใช้งาน: บอกขั้นตอนการถ่ายรูปใบข้าวให้ชัดเจน และตัวอย่างคำถามที่สามารถพิมพ์ถามบอทได้", indent=0.75, space_after=2)
    add_para("3) เกี่ยวกับระบบ: แสดงข้อมูลเกี่ยวกับโครงงาน และหน่วยงานอ้างอิงข้อมูล (กรมการข้าว)", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-2",
        "หน้าจอเมนูหลักด้านล่าง (Rich Menu)",
        "หน้าจอ LINE แสดงปุ่มเมนูด้านล่าง 3 ปุ่ม ได้แก่ ตรวจโรคข้าว, วิธีการใช้งาน, และเกี่ยวกับระบบ"
    )
    
    # ก.1.3
    add_para("ก.1.3 การส่งรูปภาพใบข้าวเพื่อตรวจโรค", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "เมื่อพบใบข้าวที่มีอาการผิดปกติในแปลงนา สามารถส่งรูปให้บอทตรวจได้ตามขั้นตอนนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) กดที่ไอคอน “กล้องถ่ายรูป” เพื่อถ่ายรูปใบข้าวในนาจริง หรือกดไอคอน “รูปภาพ” เพื่อเลือกรูปถ่ายที่มีอยู่ในเครื่อง", indent=0.75, space_after=2)
    add_para("2) คำแนะนำในการถ่ายรูป: ควรถ่ายรูปใบข้าวใกล้ ๆ (ระยะใกล้) ให้เห็นแผลหรือรอยโรคชัดเจน ให้ใบข้าวอยู่ตรงกลางภาพ มีแสงสว่างพอดี และหลีกเลี่ยงภาพเบลอ ภาพสั่น หรือภาพมุมกว้างทั้งแปลงนา", indent=0.75, space_after=2)
    add_para("3) กดปุ่ม “ส่ง” รูปภาพเข้าห้องแชท บอทจะส่งข้อความแจ้งว่ากำลังวิเคราะห์รูปภาพ ให้รอสักครู่", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-3",
        "การเลือกและส่งรูปภาพใบข้าวเข้าสู่ห้องแชท",
        "หน้าจอ LINE ตอนเลือกรูปถ่ายใบข้าวที่มีอาการผิดปกติและกดส่งเข้าไปในแชท"
    )
    
    # ก.1.4
    add_para("ก.1.4 การรับผลการตรวจและวิธีรักษาโรค", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "เมื่อบอทวิเคราะห์รูปภาพเสร็จแล้ว จะส่งข้อความบอกรายละเอียดกลับมา 3 ส่วน ดังนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) ผลการตรวจวินิจฉัย: บอกชื่อโรคข้าวที่ตรวจพบ (ทั้งภาษาไทยและภาษาอังกฤษ เช่น โรคไหม้ข้าว / Rice Blast) พร้อมบอกระดับความมั่นใจของระบบเป็นเปอร์เซ็นต์", indent=0.75, space_after=2)
    add_para("2) สาเหตุและอาการของโรค: อธิบายว่าโรคเกิดจากเชื้อราหรือแบคทีเรีย และมีลักษณะแผลบนใบอย่างไร", indent=0.75, space_after=2)
    add_para("3) วิธีรักษาและการดูแล: แนะนำวิธีจัดการแปลงนา และชนิดของยาหรือสารชีวภัณฑ์ที่แนะนำให้ใช้ พร้อมอัตราการผสมฉีดพ่นตามคู่มือของกรมการข้าว", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-4",
        "ข้อความผลการวินิจฉัยโรคและคำแนะนำการรักษา",
        "หน้าจอแสดงข้อความที่แชทบอทตอบกลับ บอกชื่อโรค เปอร์เซ็นต์ความมั่นใจ สาเหตุ และวิธีรักษา"
    )
    
    # ก.1.5
    add_para("ก.1.5 การพิมพ์ข้อความสอบถามข้อมูลโรคข้าว", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "นอกจากส่งรูปภาพแล้ว ผู้ใช้ยังสามารถพิมพ์ข้อความถามข้อสงสัยเกี่ยวกับโรคข้าวหรือการดูแลรักษาได้เลย เช่น “โรคขอบใบแห้งเกิดจากอะไร”, “ถ้าข้าวเป็นโรคไหม้ต้องใช้ยาตัวไหนฉีด”, หรือ “โรคใบจุดสีน้ำตาลมีวิธีป้องกันยังไง” บอทจะค้นหาข้อมูลจากคู่มือวิชาการของกรมการข้าว แล้วสรุปคำตอบที่เข้าใจง่ายและตรงจุดส่งกลับมาให้ทันที",
        indent=0.5, space_after=8
    )
    add_placeholder_box(
        "ก-5",
        "การพิมพ์ข้อความสอบถามข้อมูลโรคข้าว",
        "หน้าจอตอนพิมพ์ถามข้อสงสัยเรื่องโรคข้าว และบอทตอบคำแนะนำกลับมา"
    )
    
    # ก.1.6
    add_para("ก.1.6 การแจ้งเตือนกรณีส่งรูปไม่ชัดเจนหรือถามนอกเรื่อง", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "ระบบมีการแจ้งเตือนเพื่อช่วยแนะนำผู้ใช้งานในกรณีต่าง ๆ ดังนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) ถ้าส่งรูปอื่นที่ไม่ใช่ใบข้าว หรือภาพมองไม่เห็นรอยโรค: บอทจะตอบว่า “ตรวจไม่พบรอยโรคข้าวในภาพ กรุณาถ่ายภาพใบข้าวที่มีรอยโรคให้ชัดเจนในระยะใกล้แล้วส่งใหม่อีกครั้ง”", indent=0.75, space_after=2)
    add_para("2) ถ้าพิมพ์ถามเรื่องอื่นที่ไม่เกี่ยวกับข้าว: เช่น ถามสภาพอากาศ ราคาทอง หรือถามเรื่องพืชชนิดอื่น บอทจะแจ้งอย่างสุภาพว่า ระบบนี้เชี่ยวชาญเฉพาะเรื่องโรคข้าว จึงไม่สามารถตอบเรื่องอื่นได้", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-6",
        "การแจ้งเตือนเมื่อส่งภาพที่ไม่ใช่ใบข้าวหรือถามนอกเรื่อง",
        "หน้าจอแชทบอทตอบแจ้งเตือนเมื่อตรวจไม่พบโรคข้าว หรือพิมพ์ถามเรื่องที่ไม่เกี่ยวข้อง"
    )
    
    # -------------------- ก.2 --------------------
    add_para("ก.2 ส่วนสำหรับผู้ดูแลระบบ (Admin)", bold=True, size=16, space_before=10, space_after=4)
    
    # ก.2.1
    add_para("ก.2.1 การเข้าสู่ระบบจัดการ (n8n Login)", bold=True, size=16, indent=0.25, space_after=3)
    add_para(
        "ผู้ดูแลระบบสามารถเข้าใช้งานระบบควบคุมการทำงานอัตโนมัติได้ดังนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) เปิดเว็บเบราว์เซอร์ (เช่น Google Chrome) แล้วพิมพ์ลิงก์ของระบบ n8n (เช่น http://localhost:5678 หรือลิงก์ประจำเซิร์ฟเวอร์)", indent=0.75, space_after=2)
    add_para("2) กรอก Email และ Password ของผู้ดูแลระบบ", indent=0.75, space_after=2)
    add_para("3) กดปุ่ม “Sign in” เพื่อเข้าสู่หน้าแดชบอร์ด", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-7",
        "หน้าจอเข้าสู่ระบบจัดการ (n8n Login)",
        "หน้าจอเข้าสู่ระบบ n8n สำหรับกรอกอีเมลและรหัสผ่านของผู้ดูแลระบบ"
    )
    
    # ก.2.2
    add_para("ก.2.2 การตรวจสอบผังการทำงานของระบบ (Workflow Canvas)", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "เมื่อเข้าสู่ระบบแล้ว ให้เลือกเวิร์กโฟลว์ “Rice Leaf Disease Diagnosis Bot” หน้าจอจะแสดงขั้นตอนการทำงานของระบบที่เชื่อมต่อกันเป็นผังงาน (Canvas) ได้แก่:",
        indent=0.5, space_after=2
    )
    add_para("1) โหนดรับข้อมูลจาก LINE (LINE Webhook): ทำหน้าที่รับรูปภาพและข้อความที่ผู้ใช้ส่งเข้ามา", indent=0.75, space_after=2)
    add_para("2) โหนดแยกประเภท (Router): คัดแยกว่าเป็นรูปภาพหรือข้อความ", indent=0.75, space_after=2)
    add_para("3) โหนดเรียกโมเดล AI (Model API): ส่งรูปไปให้โมเดลตรวจจับโรคข้าว", indent=0.75, space_after=2)
    add_para("4) โหนดค้นหาข้อมูล (Pinecone Vector Search): ค้นหาวิธีรักษาโรคข้าวจากฐานข้อมูล", indent=0.75, space_after=2)
    add_para("5) โหนดสร้างคำตอบ (Google Gemini): เรียบเรียงคำแนะนำให้อ่านเข้าใจง่าย", indent=0.75, space_after=2)
    add_para("6) โหนดส่งข้อความตอบกลับ (LINE Reply): ส่งผลลัพธ์กลับไปที่แชทของผู้ใช้ (หมายเหตุ: ต้องตรวจดูสวิตช์มุมขวาบนให้เปิดเป็น “Active” เพื่อให้ระบบทำงานตลอดเวลา)", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-8",
        "หน้าจอผังการทำงานของแชทบอทบน n8n",
        "หน้าจอแสดงโหนดการทำงานต่าง ๆ ของระบบแชทบอทที่เชื่อมต่อกันบน n8n"
    )
    
    # ก.2.3
    add_para("ก.2.3 การดูประวัติการใช้งานและข้อผิดพลาด (Execution Logs)", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "ผู้ดูแลระบบสามารถตรวจสอบการทำงานย้อนหลังของบอทได้ดังนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) คลิกที่เมนู “Executions” ทางแถบด้านซ้าย", indent=0.75, space_after=2)
    add_para("2) หน้าจอจะแสดงรายการที่เคยประมวลผลทั้งหมด พร้อมบอกเวลา สถานะว่าผ่านหรือไม่ (Success หรือ Error) และเวลาที่ใช้ไป", indent=0.75, space_after=2)
    add_para("3) สามารถคลิกเข้าไปดูรายละเอียดของแต่ละรายการได้ว่า ผู้ใช้ส่งภาพหรือข้อความอะไรเข้ามา และแต่ละขั้นตอนประมวลผลได้ผลลัพธ์อย่างไร ช่วยให้ตรวจสอบและแก้ไขปัญหาได้ง่าย", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-9",
        "หน้าจอแสดงประวัติการทำงาน (Executions) บน n8n",
        "หน้าจอแสดงรายการประวัติการใช้งานย้อนหลัง และการกดดูข้อมูลในแต่ละโหนด"
    )
    
    # ก.2.4
    add_para("ก.2.4 การจัดการฐานข้อมูลความรู้ (Pinecone Vector Database)", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "ผู้ดูแลระบบสามารถตรวจดูฐานข้อมูลความรู้โรคข้าวได้ตามขั้นตอนนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) เข้าไปที่เว็บไซต์ Pinecone (https://app.pinecone.io) และลงชื่อเข้าใช้", indent=0.75, space_after=2)
    add_para("2) เลือกดู Index ประจำโครงงาน เช่น rice-disease-index", indent=0.75, space_after=2)
    add_para("3) ดูข้อมูลสำคัญ เช่น Total Vectors (จำนวนชิ้นข้อมูลความรู้ที่เก็บไว้ในระบบ), Dimension (ขนาดมิติของเวกเตอร์ เช่น 768 หรือ 1536) และสถิติการใช้งาน เช่น จำนวนครั้งที่มีการค้นหาข้อมูล", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-10",
        "หน้าจอจัดการฐานข้อมูลความรู้บน Pinecone",
        "หน้าจอแดชบอร์ด Pinecone แสดงข้อมูล Index และจำนวนข้อมูลที่จัดเก็บ"
    )
    
    # ก.2.5
    add_para("ก.2.5 การอัปเดตข้อมูลความรู้โรคข้าวเพิ่มเติม", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "หากมีข้อมูลโรคข้าวใหม่ หรือมีคู่มือวิชาการฉบับใหม่จากกรมการข้าว สามารถเพิ่มเข้าสู่ระบบได้ดังนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) นำไฟล์เอกสารฉบับใหม่ (ไฟล์ .pdf หรือ .txt) ไปวางไว้ในโฟลเดอร์ data/knowledge_base/", indent=0.75, space_after=2)
    add_para("2) เปิดโปรแกรม Terminal หรือ Command Prompt", indent=0.75, space_after=2)
    add_para("3) พิมพ์คำสั่งรันสคริปต์: python scripts/ingest_knowledge.py", indent=0.75, space_after=2)
    add_para("4) ระบบจะอ่านเอกสาร ตัดแบ่งข้อความ และแปลงเป็นข้อมูลเวกเตอร์อัปโหลดเข้า Pinecone ให้โดยอัตโนมัติ ทำให้บอทสามารถนำข้อมูลใหม่ไปใช้ตอบคำถามผู้ใช้ได้ทันที", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-11",
        "การรันคำสั่งอัปเดตข้อมูลความรู้โรคข้าว",
        "หน้าต่าง Terminal ตอนรันคำสั่งเพิ่มข้อมูลคู่มือโรคข้าวเข้าสู่ฐานข้อมูล"
    )
    
    # Output path
    out_docx = "Project/Appendix_A.docx"
    try:
        doc.save(out_docx)
        print(f"Successfully generated {out_docx}!")
    except PermissionError:
        out_docx = "Project/Appendix_A_updated.docx"
        doc.save(out_docx)
        print(f"Appendix_A.docx is open in Word. Saved to {out_docx} instead!")

if __name__ == "__main__":
    create_appendix_a()
