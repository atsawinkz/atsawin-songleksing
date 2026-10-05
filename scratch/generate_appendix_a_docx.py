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
    for _ in range(7):
        doc.add_paragraph()
        
    p_cov1 = doc.add_paragraph()
    p_cov1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cov1 = p_cov1.add_run("ภาคผนวก ก")
    set_font(r_cov1, size=18, bold=True)
    
    p_cov2 = doc.add_paragraph()
    p_cov2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cov2 = p_cov2.add_run("คู่มือการใช้งานระบบแชทบอทวินิจฉัยโรคข้าวและแนะนำวิธีการรักษา\nผ่านแอปพลิเคชันไลน์ด้วยเทคนิค RAG")
    set_font(r_cov2, size=18, bold=True)
    
    doc.add_page_break()
    
    # ==================== PAGE 2: MANUAL BODY ====================
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_after = Pt(14)
    r_title = p_title.add_run("คู่มือการใช้งานระบบแชทบอทวินิจฉัยโรคข้าวและแนะนำวิธีการรักษา\nผ่านแอปพลิเคชันไลน์ด้วยเทคนิค RAG")
    set_font(r_title, size=18, bold=True)
    
    add_para(
        "คู่มือการใช้งานระบบแชทบอทวินิจฉัยโรคข้าวและแนะนำวิธีการรักษา แบ่งออกเป็น 2 ส่วน ดังนี้",
        indent=0.5, space_after=2
    )
    add_para("1) ส่วนสำหรับผู้ใช้งานทั่วไป (เกษตรกร): การใช้งานแชทบอทผ่านบัญชี LINE Official Account (@325wtzzd)", indent=0.75, space_after=2)
    add_para("2) ส่วนสำหรับผู้ดูแลระบบ (Admin): การจัดการระบบเวิร์กโฟลว์อัตโนมัติ (n8n) และฐานข้อมูลความรู้เวกเตอร์ (Pinecone)", indent=0.75, space_after=12)
    
    # -------------------- ก.1 --------------------
    add_para("ก.1 ส่วนสำหรับเกษตรกรและผู้ใช้งานทั่วไป", bold=True, size=16, space_before=6, space_after=4)
    
    # ก.1.1
    add_para("ก.1.1 การเพิ่มเพื่อนและเริ่มต้นใช้งานแชทบอท", bold=True, size=16, indent=0.25, space_after=3)
    add_para(
        "ผู้ใช้สามารถเริ่มต้นใช้งานได้ง่าย ๆ โดยการเพิ่มเพื่อนใน LINE ผ่าน 2 วิธี ได้แก่ การสแกนคิวอาร์โค้ด (QR Code) ประจำบัญชีทางการ หรือค้นหาด้วย LINE ID: @325wtzzd (ชื่อบัญชี: บอทวินิจฉัยโรคข้าว หรือ Rice Disease AI) เมื่อกดเพิ่มเพื่อนและเปิดเข้าไปในห้องแชท แชทบอทจะส่งข้อความทักทายต้อนรับ แนะนำตัว และบอกวิธีใช้งานเบื้องต้น พร้อมกับแสดงแถบเมนูลัด (Rich Menu) 3 ปุ่มที่ด้านล่างของหน้าจอทันที",
        indent=0.5, space_after=8
    )
    add_placeholder_box(
        "ก-1",
        "การเพิ่มเพื่อนและข้อความต้อนรับเข้าสู่ระบบแชทบอท",
        "ภาพหน้าจอแอปพลิเคชัน LINE ขณะเพิ่มเพื่อนสำเร็จ มีข้อความต้อนรับอัตโนมัติปรากฏขึ้น และมีแถบ Rich Menu แสดงอยู่ที่ด้านล่างของหน้าจอ"
    )
    
    # ก.1.2
    add_para("ก.1.2 การใช้งานเมนูหลัก (Rich Menu)", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "ที่แถบด้านล่างของหน้าจอห้องแชทจะแสดงเมนูลัด (Rich Menu) ให้กดใช้งานได้อย่างสะดวกรวดเร็ว โดยแบ่งออกเป็น 3 ปุ่มคำสั่งหลัก ได้แก่:",
        indent=0.5, space_after=2
    )
    add_para("1) ถ่ายภาพตรวจโรคข้าว (TAKE DIAGNOSIS PHOTO): เมื่อกดปุ่ม ระบบจะเปิดกล้องถ่ายรูปบนโทรศัพท์มือถือ เพื่อให้เกษตรกรสามารถเล็งถ่ายภาพใบข้าวที่มีอาการผิดปกติจากแปลงนาส่งตรวจได้ทันที", indent=0.75, space_after=2)
    add_para("2) ส่งภาพตรวจโรคข้าว (SUBMIT DISEASE PHOTO): เมื่อกดปุ่ม ระบบจะเปิดคลังภาพ (Gallery) หรืออัลบั้มในโทรศัพท์ เพื่อให้เลือกรูปภาพใบข้าวที่เคยถ่ายเก็บไว้ส่งเข้ามาให้ระบบวิเคราะห์", indent=0.75, space_after=2)
    add_para("3) วิธีการใช้งานแชทบอท (CHATBOT USER GUIDE): เมื่อกดปุ่ม ระบบจะส่งข้อความแนะนำขั้นตอนการใช้งานระบบ 3 ขั้นตอน และเทคนิคการถ่ายภาพใบข้าวระยะใกล้ให้คมชัด พร้อมตัวอย่างคำถามที่สามารถพิมพ์ถามบอทได้", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-2",
        "หน้าจอแถบเมนูลัด (Rich Menu 3 ปุ่ม) ด้านล่างของห้องแชท LINE",
        "ภาพหน้าจอแสดงแถบ Rich Menu ด้านล่างของห้องแชท LINE ประกอบด้วย 3 ปุ่มคำสั่ง: ถ่ายภาพตรวจโรคข้าว, ส่งภาพตรวจโรคข้าว, และวิธีการใช้งานแชทบอท"
    )
    
    # ก.1.3
    add_para("ก.1.3 การส่งรูปภาพใบข้าวเพื่อตรวจวินิจฉัยโรค", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "เมื่อพบใบข้าวที่มีอาการผิดปกติในแปลงนา สามารถส่งรูปให้บอทตรวจได้ตามขั้นตอนนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) กดที่ปุ่ม “ถ่ายภาพตรวจโรคข้าว” เพื่อเปิดกล้องถ่ายภาพจริงในแปลงนา หรือกดปุ่ม “ส่งภาพตรวจโรคข้าว” เพื่อเลือกรูปภาพที่มีอยู่ในเครื่อง", indent=0.75, space_after=2)
    add_para("2) คำแนะนำในการถ่ายรูป: ควรถ่ายภาพใบข้าวในระยะใกล้ (ประมาณ 15–30 เซนติเมตร) ให้เห็นรอยโรคชัดเจนอยู่บริเวณกึ่งกลางภาพ มีแสงสว่างเพียงพอ หลีกเลี่ยงภาพเบลอ ภาพสั่น หรือการถ่ายย้อนแสงแดดจัด และหลีกเลี่ยงการถ่ายภาพมุมกว้างที่เห็นทั้งแปลงนา เพราะ AI จะมองไม่เห็นรายละเอียดของแผล", indent=0.75, space_after=2)
    add_para("3) กดปุ่ม “ส่ง” รูปภาพเข้าสู่ห้องแชท ระบบจะส่งข้อความแจ้งเตือนสถานะการประมวลผลให้ผู้ใช้ทราบว่ากำลังวิเคราะห์รูปภาพ", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-3",
        "การเลือกและส่งรูปภาพใบข้าวที่มีรอยโรคเข้าสู่ห้องแชท LINE",
        "ภาพหน้าจอขณะผู้ใช้เลือกภาพถ่ายใบข้าวที่มีอาการของโรคจากอัลบั้มหรือกล้อง และกดส่งเข้ามาในห้องแชท"
    )
    
    # ก.1.4
    add_para("ก.1.4 การรับผลการวินิจฉัยโรคและคำแนะนำการรักษา", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "เมื่อระบบประมวลผลภาพเสร็จสิ้น แชทบอทจะส่งข้อความตอบกลับในรูปแบบข้อความตัวหนังสือที่มีโครงสร้างชัดเจนและอ่านง่าย ประกอบด้วย 2 ส่วนสำคัญ ได้แก่:",
        indent=0.5, space_after=2
    )
    add_para("1) ผลการวินิจฉัยโรคข้าว: ระบุชื่อโรคข้าวที่ตรวจพบ (เช่น โรคใบจุดสีน้ำตาล, โรคไหม้ข้าว, โรคขอบใบแห้ง) พร้อมระบุระดับความมั่นใจของโมเดล (Confidence Score เช่น ความมั่นใจ 85.8%)", indent=0.75, space_after=2)
    add_para("2) คำแนะนำและการรักษา: สรุปวิธีปฏิบัติตามคู่มือวิชาการของกรมการข้าว กระทรวงเกษตรและสหกรณ์ ได้แก่ การเตรียมเมล็ดพันธุ์ก่อนปลูก, การจัดการแปลงนา, และการใช้สารเคมีป้องกันกำจัดโรคพืช (ชื่อสารเคมีที่แนะนำ เช่น ไตรไซคลาโซล, แมนโคเซบ, คาร์เบนดาซิม, คาซูกาไมซิน) พร้อมอัตราส่วนผสมที่ปลอดภัย", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-4",
        "ข้อความผลการวินิจฉัยโรคและคำแนะนำการรักษาที่บอทส่งตอบกลับผู้ใช้",
        "ภาพหน้าจอข้อความผลลัพธ์ที่แชทบอทส่งกลับมา แสดงชื่อโรค ค่าความมั่นใจ และคำแนะนำการรักษาตามคู่มือกรมการข้าว"
    )
    
    # ก.1.5
    add_para("ก.1.5 การพิมพ์ข้อความสอบถามข้อมูลโรคข้าวเพิ่มเติม", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "นอกจากส่งรูปภาพแล้ว เกษตรกรสามารถพิมพ์ข้อความสอบถามข้อสงสัยเกี่ยวกับโรคข้าว การดูแลรักษา หรือการใช้ยาได้โดยตรง เช่น “โรคใบไหม้รักษายังไง ใช้สารเคมีอะไร”, “โรคขอบใบแห้งเกิดจากอะไร”, หรือ “โรคจุดน้ำตาลเป็นยังไง” ระบบสืบค้นข้อมูลความรู้ (Hybrid RAG) จะค้นหาข้อมูลที่ตรงจุดจากฐานข้อมูลคู่มือวิชาการของกรมการข้าว และใช้ AI ช่วยเรียบเรียงคำตอบที่เข้าใจง่ายและตรงจุดส่งกลับมาให้ทันที",
        indent=0.5, space_after=8
    )
    add_placeholder_box(
        "ก-5",
        "การพิมพ์ข้อความสอบถามข้อมูลโรคข้าวและคำตอบจากระบบ",
        "ภาพหน้าจอการสนทนาที่ผู้ใช้พิมพ์คำถามเกี่ยวกับโรคข้าวหรือการใช้ยา และแชทบอทตอบกลับด้วยคำแนะนำที่ถูกต้องตามหลักวิชาการ"
    )
    
    # ก.1.6
    add_para("ก.1.6 การแจ้งเตือนกรณีส่งรูปไม่ชัดเจนหรืออยู่นอกขอบเขต (Fallback & Guardrails)", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "ระบบมีข้อความแจ้งเตือนเพื่อช่วยแนะนำผู้ใช้งานในกรณีต่าง ๆ ดังนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) ถ้าส่งรูปอื่นที่ไม่ใช่ใบข้าว หรือภาพมองไม่เห็นรอยโรค: บอทจะตอบว่า “ระบบตรวจไม่พบรอยโรคข้าวในภาพ กรุณาถ่ายภาพใบข้าวที่มีรอยโรคให้ชัดเจนในระยะใกล้และส่งใหม่อีกครั้ง”", indent=0.75, space_after=2)
    add_para("2) ถ้าพิมพ์ถามเรื่องอื่นที่ไม่เกี่ยวกับข้าว: เช่น ถามสภาพอากาศ ราคาทอง หรือถามเรื่องพืชชนิดอื่น ระบบ Guardrail จะตอบปฏิเสธอย่างสุภาพว่า “ขออภัยครับ ระบบนี้เป็น AI ผู้ช่วยวินิจฉัยและให้คำปรึกษาเรื่องโรคข้าวจากกรมการข้าวโดยเฉพาะ จึงสามารถตอบได้เฉพาะเรื่องโรคข้าวและการดูแลรักษาเท่านั้นครับ”", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-6",
        "การแจ้งเตือนเมื่อส่งภาพที่ไม่ใช่ใบข้าวหรือถามคำถามนอกเรื่อง",
        "ภาพหน้าจอแสดงข้อความปฏิเสธอย่างสุภาพของแชทบอทเมื่อผู้ใช้ส่งภาพวัตถุอื่นหรือพิมพ์คำถามที่ไม่เกี่ยวข้องกับเรื่องโรคข้าว"
    )
    
    # -------------------- ก.2 --------------------
    add_para("ก.2 ส่วนสำหรับผู้ดูแลระบบ (Admin)", bold=True, size=16, space_before=10, space_after=4)
    
    # ก.2.1
    add_para("ก.2.1 การเข้าสู่ระบบจัดการเวิร์กโฟลว์ (n8n Login)", bold=True, size=16, indent=0.25, space_after=3)
    add_para(
        "ผู้ดูแลระบบสามารถเข้าใช้งานระบบควบคุมการทำงานอัตโนมัติได้ดังนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) เปิดโปรแกรมเว็บเบราว์เซอร์ (เช่น Google Chrome) แล้วพิมพ์ URL ของระบบ n8n (เช่น http://158.108.113.124:5678 หรือ URL ประจำเซิร์ฟเวอร์)", indent=0.75, space_after=2)
    add_para("2) กรอก Email และ Password ของบัญชีผู้ดูแลระบบ", indent=0.75, space_after=2)
    add_para("3) กดปุ่ม “Sign in” เพื่อเข้าสู่หน้าแดชบอร์ดหลักของ n8n", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-7",
        "หน้าจอเข้าสู่ระบบจัดการ (n8n Login)",
        "ภาพหน้าจอ Login ของ n8n สำหรับให้ผู้ดูแลระบบกรอกอีเมลและรหัสผ่านเพื่อเข้าใช้งาน"
    )
    
    # ก.2.2
    add_para("ก.2.2 การตรวจสอบผังการทำงานของระบบ (Workflow Canvas)", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "เมื่อเข้าสู่ระบบแล้ว ให้เลือกเวิร์กโฟลว์ “Rice Disease LINE Chatbot” หน้าจอจะแสดงขั้นตอนการทำงานของระบบที่เชื่อมต่อกันเป็นผังงาน (Canvas) ได้แก่:",
        indent=0.5, space_after=2
    )
    add_para("1) LINE Webhook Node: ทำหน้าที่รับข้อมูลรูปภาพและข้อความที่ผู้ใช้ส่งเข้ามาจาก LINE", indent=0.75, space_after=2)
    add_para("2) Router Node: ทำหน้าที่คัดแยกว่าข้อมูลที่ส่งเข้ามาเป็นรูปภาพหรือข้อความพิมพ์สอบถาม", indent=0.75, space_after=2)
    add_para("3) HTTP Request Node (Model API): เชื่อมต่อไปยังเซิร์ฟเวอร์ AI เพื่อส่งภาพไปให้โมเดล RT-DETR-L v5 ตรวจจับรอยโรค", indent=0.75, space_after=2)
    add_para("4) Pinecone Vector Search Node: เชื่อมต่อค้นหาข้อมูลวิธีการรักษาโรคข้าวจากดัชนีฐานข้อมูล", indent=0.75, space_after=2)
    add_para("5) Google Gemini 2.5 Flash Node: เรียบเรียงคำตอบและคำแนะนำการรักษาให้อ่านเข้าใจง่าย", indent=0.75, space_after=2)
    add_para("6) LINE Reply (Text RAG) Node: ส่งข้อความตอบกลับไปยังห้องแชท LINE ของผู้ใช้งาน (หมายเหตุ: ผู้ดูแลระบบต้องตรวจดูสวิตช์มุมขวาบนให้เปิดเป็น “Active” เสมอ เพื่อให้ระบบทำงานได้ตลอด 24 ชั่วโมง)", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-8",
        "หน้าจอผังการทำงานของแชทบอท (Workflow Canvas) บนระบบ n8n",
        "ภาพหน้าจอผังการเชื่อมต่อโหนดต่าง ๆ บน n8n แสดงเส้นทางการรับส่งข้อมูลตั้งแต่ Webhook จนถึงการส่งข้อความตอบกลับ"
    )
    
    # ก.2.3
    add_para("ก.2.3 การดูประวัติการใช้งานและตรวจสอบข้อมูลเชิงลึก (Execution Logs)", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "ผู้ดูแลระบบสามารถตรวจสอบการทำงานย้อนหลังและการวิเคราะห์ปัญหา (Debugging) ได้ดังนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) คลิกที่แถบเมนู “Executions” ทางแถบด้านซ้ายของหน้าจอ", indent=0.75, space_after=2)
    add_para("2) ระบบจะแสดงตารางรายการประวัติการเรียกใช้งานทั้งหมด พร้อมแสดงสถานะการทำงาน (Success / Error), เวลาที่เริ่มต้นทำงาน (Start time), และระยะเวลาที่ใช้ในการประมวลผล (Duration)", indent=0.75, space_after=2)
    add_para("3) เมื่อคลิกที่รายการใดรายการหนึ่ง หน้าจอจะแสดงรายละเอียดข้อมูลนำเข้าและส่งออก (Input / Output Payload) ของแต่ละโหนดอย่างละเอียด เช่น โหนด LINE Reply (Text RAG) จะแสดงข้อความคำถาม, รหัสโรค (disease_id), คำตอบที่ AI สร้างขึ้น, เอกสารบริบทที่ดึงมา (contexts) และรหัสยืนยันการส่งข้อความสำเร็จ (sentMessages ID)", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-9",
        "หน้าจอแสดงประวัติการทำงาน (Executions) และการตรวจสอบ Payload บน n8n",
        "ภาพหน้าจอแสดงประวัติการทำงานย้อนหลังในหน้า Executions และการคลิกดูรายละเอียด Payload ของแต่ละโหนด"
    )
    
    # ก.2.4
    add_para("ก.2.4 การจัดการฐานข้อมูลความรู้เวกเตอร์ (Pinecone Vector Database)", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "ผู้ดูแลระบบสามารถตรวจสอบและจัดการดัชนีข้อมูลความรู้โรคข้าวได้ตามขั้นตอนนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) เข้าสู่เว็บไซต์คอนโซลของ Pinecone (https://app.pinecone.io) และลงชื่อเข้าใช้", indent=0.75, space_after=2)
    add_para("2) เลือกดูรายละเอียด Index ประจำโครงงาน ชื่อ rice-knowledge", indent=0.75, space_after=2)
    add_para("3) ตรวจสอบข้อมูลสถิติที่สำคัญ: Index Name: rice-knowledge (Type: Dense, Region: aws us-east-1), Record Count: จำนวนชิ้นข้อมูลความรู้โรคข้าว (Chunks) ที่จัดเก็บอยู่ในระบบ (เช่น 10 Chunks), Records: รายการข้อมูลความรู้ของแต่ละโรค เช่น bacterial_leaf_blight_chunk_0, leaf_blast_chunk_0, sheath_blight_chunk_0 และสถิติการใช้งาน เช่น จำนวนครั้งที่มีการค้นหาข้อมูล", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-10",
        "หน้าจอจัดการ Index rice-knowledge บน Pinecone Vector Database Console",
        "ภาพหน้าจอแดชบอร์ดบนเว็บไซต์ Pinecone แสดงสถานะ Index จำนวนเวกเตอร์ และมิติข้อมูลของฐานความรู้โรคข้าว"
    )
    
    # ก.2.5
    add_para("ก.2.5 การอัปเดตและเพิ่มเอกสารความรู้โรคข้าว (Knowledge Ingestion Script)", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "กรณีที่มีข้อมูลโรคข้าวใหม่ สารเคมีชนิดใหม่ หรือต้องการอัปเดตคู่มือวิชาการของกรมการข้าว ผู้ดูแลระบบสามารถเพิ่มข้อมูลเข้าสู่ระบบได้ดังนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) นำไฟล์เอกสารคู่มือฉบับใหม่ (ไฟล์ .pdf, .md หรือ .txt) ไปจัดเก็บไว้ในโฟลเดอร์คลังเอกสารของระบบ (data/knowledge_base/)", indent=0.75, space_after=2)
    add_para("2) เปิดโปรแกรม Terminal หรือ Command Prompt ในเครื่องเซิร์ฟเวอร์", indent=0.75, space_after=2)
    add_para("3) รันคำสั่งสคริปต์สำหรับการนำเข้าข้อมูลความรู้: python scripts/ingest_knowledge.py", indent=0.75, space_after=2)
    add_para("4) สคริปต์จะทำการอ่านเอกสาร ตัดแบ่งข้อความออกเป็นส่วนย่อย (Chunking), คำนวณค่า Vector Embeddings และอัปโหลดเวกเตอร์ชุดใหม่เข้าสู่ Pinecone Index rice-knowledge โดยอัตโนมัติ ทำให้บอทสามารถนำข้อมูลความรู้ใหม่มาใช้ตอบคำถามแก่เกษตรกรได้ทันทีโดยไม่ต้องทำการเทรนโมเดลใหม่", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-11",
        "หน้าต่างการรันสคริปต์อัปเดตเอกสารความรู้โรคข้าวเข้าสู่ Vector Database",
        "ภาพหน้าจอ Terminal ขณะรันสคริปต์ Ingestion แสดงขั้นตอนการอ่านไฟล์เอกสาร การทำ Chunking และการอัปโหลดเวกเตอร์ขึ้น Pinecone สำเร็จ"
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
