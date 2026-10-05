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
        "ส่วนของคู่มือการใช้งานระบบแชทบอทวิเคราะห์โรคข้าว แบ่งออกเป็น 2 ส่วน ประกอบด้วย",
        indent=0.5, space_after=2
    )
    add_para("1) ส่วนสำหรับผู้ใช้งานทั่วไป (เกษตรกร)", indent=0.75, space_after=2)
    add_para("2) ส่วนสำหรับผู้ดูแลระบบ (Admin)", indent=0.75, space_after=12)
    
    # -------------------- ก.1 --------------------
    add_para("ก.1 ส่วนสำหรับเกษตรกรและผู้ใช้งานทั่วไป", bold=True, size=16, space_before=6, space_after=4)
    
    # ก.1.1
    add_para("ก.1.1 การเพิ่มเพื่อนและเริ่มต้นใช้งานแชทบอท (Add Friend)", bold=True, size=16, indent=0.25, space_after=3)
    add_para(
        "ผู้ใช้งานสามารถเริ่มต้นใช้งานระบบได้โดยการเพิ่มเพื่อนในแอปพลิเคชัน LINE ผ่าน 2 ช่องทาง ได้แก่ การสแกนรหัสคิวอาร์ (QR Code) ประจำบัญชีทางการ หรือค้นหาด้วย LINE ID (@บอทวินิจฉัยโรคข้าว) เมื่อทำการเพิ่มเพื่อนสำเร็จและเข้าสู่ห้องแชท ระบบจะส่งข้อความต้อนรับอัตโนมัติ (Greeting Message) เพื่อแนะนำตัวและสรุปฟังก์ชันการทำงานเบื้องต้น พร้อมทั้งแสดงแถบเมนูด้านล่าง (Rich Menu) เพื่อให้ผู้ใช้เลือกเข้าถึงคำสั่งต่าง ๆ ได้ทันที",
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
        "แถบเมนูลัดที่ด้านล่างของหน้าจอห้องแชท (Rich Menu) ออกแบบมาเพื่อให้เกษตรกรสามารถกดใช้งานได้อย่างสะดวกรวดเร็ว โดยแบ่งออกเป็น 3 เมนูคำสั่งหลัก ได้แก่:",
        indent=0.5, space_after=2
    )
    add_para("1) ตรวจโรคข้าว: เมนูแนะนำการส่งภาพถ่ายใบข้าวเพื่อให้ระบบทำการวิเคราะห์", indent=0.75, space_after=2)
    add_para("2) วิธีการใช้งาน: เมนูแสดงคำแนะนำขั้นตอนการถ่ายภาพใบข้าวที่ถูกต้อง และตัวอย่างคำถามที่สามารถสอบถามบอทได้", indent=0.75, space_after=2)
    add_para("3) เกี่ยวกับระบบ / ข้อมูลอ้างอิง: เมนูแสดงข้อมูลรายละเอียดของโครงงาน และแหล่งอ้างอิงองค์ความรู้จากกรมการข้าว กระทรวงเกษตรและสหกรณ์", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-2",
        "หน้าจอเมนูหลัก (Rich Menu) ที่แถบด้านล่างของห้องแชท LINE",
        "ภาพหน้าจอแสดงแถบ Rich Menu ด้านล่างของห้องแชท LINE ซึ่งประกอบด้วย 3 ปุ่มคำสั่ง: ตรวจโรคข้าว, วิธีการใช้งาน, และเกี่ยวกับระบบ"
    )
    
    # ก.1.3
    add_para("ก.1.3 การส่งรูปภาพใบข้าวเพื่อตรวจวินิจฉัยโรค (Image Diagnosis)", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "เกษตรกรสามารถส่งภาพถ่ายใบข้าวที่มีอาการผิดปกติเพื่อตรวจวินิจฉัยได้ตามขั้นตอนดังนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) กดที่ไอคอน “กล้องถ่ายรูป” เพื่อถ่ายภาพใบข้าวจากแปลงนาโดยตรง หรือกดไอคอน “รูปภาพ” เพื่อเลือกภาพที่ถ่ายเก็บไว้ในคลังภาพ", indent=0.75, space_after=2)
    add_para("2) ข้อแนะนำในการถ่ายภาพ: ควรโฟกัสภาพถ่ายระยะใกล้ (Close-up) บริเวณรอยโรคบนแผ่นใบข้าว ให้เห็นแผลชัดเจน มีแสงสว่างเพียงพอ และหลีกเลี่ยงภาพเบลอหรือภาพมุมกว้างทั้งแปลงนา", indent=0.75, space_after=2)
    add_para("3) กดปุ่ม “ส่ง” (Send) เพื่อส่งภาพเข้าสู่ห้องแชท ระบบจะส่งข้อความแจ้งเตือนสถานะการประมวลผลให้ผู้ใช้ทราบว่าระบบกำลังวิเคราะห์ภาพ", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-3",
        "การเลือกและส่งรูปภาพใบข้าวที่มีรอยโรคเข้าสู่ห้องแชท LINE",
        "ภาพหน้าจอขณะผู้ใช้เลือกภาพถ่ายใบข้าวที่มีอาการของโรคจากอัลบั้มหรือกล้อง และกดส่งเข้ามาในห้องแชท"
    )
    
    # ก.1.4
    add_para("ก.1.4 การรับผลการวินิจฉัยโรคและคำแนะนำการรักษา (Diagnosis & Advisory Result)", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "เมื่อระบบประมวลผลเสร็จสิ้น แชทบอทจะส่งข้อความตอบกลับในรูปแบบข้อความตัวหนังสือที่มีโครงสร้างชัดเจนและอ่านง่าย ประกอบด้วยข้อมูล 3 ส่วนสำคัญ ได้แก่:",
        indent=0.5, space_after=2
    )
    add_para("1) ผลการวินิจฉัยโรค: แสดงชื่อโรคข้าวที่ตรวจพบ (ทั้งภาษาไทยและภาษาอังกฤษ เช่น โรคไหม้ข้าว / Rice Blast) พร้อมระบุระดับความเชื่อมั่นของโมเดล (Confidence Score คิดเป็นเปอร์เซ็นต์)", indent=0.75, space_after=2)
    add_para("2) สาเหตุและลักษณะอาการ: อธิบายสาเหตุของโรค (เชื้อราหรือแบคทีเรีย) และลักษณะแผลที่สังเกตได้บนแผ่นใบ", indent=0.75, space_after=2)
    add_para("3) แนวทางการจัดการและรักษา: คำแนะนำการปฏิบัติตามหลักวิชาการของกรมการข้าว เช่น การควบคุมระดับน้ำในนา, การงดใส่ปุ๋ยไนโตรเจนชั่วคราว, ชนิดของสารชีวภัณฑ์หรือสารเคมีที่แนะนำ พร้อมอัตราการผสมฉีดพ่นที่ถูกต้องและปลอดภัย", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-4",
        "ผลการวินิจฉัยโรคข้าวและคำแนะนำการรักษาที่บอทส่งตอบกลับผู้ใช้",
        "ภาพหน้าจอข้อความผลลัพธ์ที่แชทบอทส่งกลับมา แสดงชื่อโรค ค่าความเชื่อมั่น สาเหตุของโรค และวิธีการรักษาจากคู่มือกรมการข้าว"
    )
    
    # ก.1.5
    add_para("ก.1.5 การพิมพ์ข้อความสอบถามข้อมูลโรคข้าวเพิ่มเติม (Text Query / Conversational Q&A)", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "นอกจากการส่งภาพแล้ว ผู้ใช้งานสามารถพิมพ์ข้อความสอบถามข้อสงสัยเกี่ยวกับโรคข้าว การดูแลรักษา หรือการใช้ยาได้โดยตรง เช่น “โรคขอบใบแห้งเกิดจากอะไร”, “ถ้าข้าวเป็นโรคไหม้ต้องใช้ยาตัวไหนฉีด”, หรือ “โรคใบจุดสีน้ำตาลป้องกันยังไง” ระบบสืบค้นข้อมูลความรู้ (RAG) จะค้นหาข้อมูลที่ตรงจุดจากฐานข้อมูลวิชาการของกรมการข้าว และใช้ปัญญาประดิษฐ์เรียบเรียงคำตอบเป็นภาษาธรรมชาติที่กระชับและเข้าใจง่ายตอบกลับผู้ใช้ทันที",
        indent=0.5, space_after=8
    )
    add_placeholder_box(
        "ก-5",
        "การพิมพ์ข้อความสอบถามข้อมูลโรคข้าวและคำตอบจากระบบ",
        "ภาพหน้าจอการสนทนาที่ผู้ใช้พิมพ์คำถามเกี่ยวกับโรคข้าวหรือการใช้ยา และแชทบอทตอบกลับด้วยคำแนะนำที่ถูกต้องตามหลักวิชาการ"
    )
    
    # ก.1.6
    add_para("ก.1.6 การแจ้งเตือนกรณีภาพถ่ายไม่ชัดเจนหรืออยู่นอกขอบเขต (Fallback & Guardrails)", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "ระบบมีกลไกป้องกันความผิดพลาดเพื่อความปลอดภัยของเกษตรกร โดยแบ่งเป็น 2 กรณี:",
        indent=0.5, space_after=2
    )
    add_para("1) กรณีส่งภาพที่ไม่ใช่ใบข้าว หรือภาพที่ตรวจไม่พบรอยโรค: ระบบจะตอบกลับอย่างสุภาพว่า “ระบบตรวจไม่พบรอยโรคข้าวในภาพ กรุณาถ่ายภาพใบข้าวที่มีรอยโรคให้ชัดเจนในระยะใกล้และส่งใหม่อีกครั้ง”", indent=0.75, space_after=2)
    add_para("2) กรณีพิมพ์คำถามที่ไม่เกี่ยวข้องกับโรคข้าว: เช่น ถามสภาพอากาศ ราคาทองคำ หรือถามเรื่องพืชชนิดอื่น ระบบ Guardrail จะตอบปฏิเสธอย่างสุภาพว่า ระบบนี้เป็น AI ผู้ช่วยวินิจฉัยและให้คำปรึกษาเฉพาะเรื่องโรคข้าว จึงไม่สามารถตอบคำถามนอกขอบเขตได้", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-6",
        "การแจ้งเตือนกรณีส่งภาพที่ไม่ใช่ใบข้าวหรือคำถามนอกขอบเขต",
        "ภาพหน้าจอแสดงข้อความปฏิเสธอย่างสุภาพของแชทบอทเมื่อผู้ใช้ส่งภาพวัตถุอื่นหรือพิมพ์คำถามที่ไม่เกี่ยวข้องกับเรื่องโรคข้าว"
    )
    
    # -------------------- ก.2 --------------------
    add_para("ก.2 ส่วนสำหรับผู้ดูแลระบบ (Admin)", bold=True, size=16, space_before=10, space_after=4)
    
    # ก.2.1
    add_para("ก.2.1 การเข้าสู่ระบบจัดการเวิร์กโฟลว์ (n8n Workflow Automation)", bold=True, size=16, indent=0.25, space_after=3)
    add_para(
        "ผู้ดูแลระบบสามารถเข้าจัดการระบบควบคุมการทำงานอัตโนมัติได้ตามขั้นตอนดังนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) เปิดโปรแกรมเว็บเบราว์เซอร์ (เช่น Google Chrome) แล้วพิมพ์ URL ของระบบ n8n (เช่น http://localhost:5678 หรือ URL ประจำเซิร์ฟเวอร์)", indent=0.75, space_after=2)
    add_para("2) กรอก Email และ Password ของบัญชีผู้ดูแลระบบ", indent=0.75, space_after=2)
    add_para("3) กดปุ่ม “Sign in” เพื่อเข้าสู่หน้าแดชบอร์ดหลักของ n8n", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-7",
        "หน้าจอเข้าสู่ระบบ (Sign In) ของระบบ n8n Workflow Automation",
        "ภาพหน้าจอ Login ของ n8n สำหรับให้ผู้ดูแลระบบกรอกอีเมลและรหัสผ่านเพื่อเข้าใช้งาน"
    )
    
    # ก.2.2
    add_para("ก.2.2 การตรวจสอบผังการทำงานและการเชื่อมต่อระบบ (Workflow Canvas Overview)", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "เมื่อเข้าสู่ระบบ ผู้ดูแลระบบสามารถตรวจสอบและจัดการผังเวิร์กโฟลว์การทำงานของระบบแชทบอทได้ดังนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) เลือกเวิร์กโฟลว์โครงงาน “Rice Leaf Disease Diagnosis Bot”", indent=0.75, space_after=2)
    add_para("2) หน้าจอจะแสดงผังการเชื่อมต่อโหนดต่าง ๆ (Canvas) ได้แก่ LINE Webhook Node, Router Node, HTTP Request Node (Model API), Pinecone Vector Search Node, Google Gemini 2.5 Flash Node, และ LINE Reply Node", indent=0.75, space_after=2)
    add_para("3) ตรวจสอบสถานะการเปิดใช้งานเวิร์กโฟลว์ โดยปุ่มสวิตช์มุมขวาบนต้องอยู่ในสถานะ “Active” เพื่อให้ระบบทำงานได้ตลอดเวลา", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-8",
        "ผังการทำงานภาพรวม (Workflow Canvas) บนระบบ n8n",
        "ภาพหน้าจอผังการเชื่อมต่อโหนดต่าง ๆ บน n8n แสดงเส้นทางการรับส่งข้อมูลตั้งแต่ Webhook จนถึงการส่งข้อความตอบกลับ"
    )
    
    # ก.2.3
    add_para("ก.2.3 การตรวจสอบประวัติการทำงานและข้อผิดพลาด (Execution Logs)", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "ผู้ดูแลระบบสามารถตรวจสอบสถานะการทำงานย้อนหลังและการวิเคราะห์ปัญหา (Debugging) ได้ดังนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) คลิกที่แถบเมนู “Executions” ทางแถบด้านซ้ายของหน้าจอ", indent=0.75, space_after=2)
    add_para("2) ระบบจะแสดงตารางรายการประวัติการเรียกใช้งานทั้งหมด พร้อมแสดงสถานะการทำงาน (Success / Error), เวลาที่เริ่มต้นทำงาน (Start time), และระยะเวลาที่ใช้ในการประมวลผล (Duration)", indent=0.75, space_after=2)
    add_para("3) เมื่อคลิกที่รายการใดรายการหนึ่ง หน้าจอจะแสดงรายละเอียดข้อมูลนำเข้าและส่งออก (Input / Output Payload) ของแต่ละโหนดอย่างละเอียด ทำให้ผู้ดูแลระบบสามารถตรวจสอบได้ทันทีว่าเกิดปัญหาที่ขั้นตอนใด", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-9",
        "หน้าจอแสดงรายการประวัติการทำงาน (Execution Logs) บนระบบ n8n",
        "ภาพหน้าจอแสดงประวัติการทำงานย้อนหลังในหน้า Executions และการคลิกดูรายละเอียด Payload ของแต่ละโหนด"
    )
    
    # ก.2.4
    add_para("ก.2.4 การจัดการฐานข้อมูลความรู้เวกเตอร์ (Pinecone Vector Database)", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "ผู้ดูแลระบบสามารถตรวจสอบและจัดการดัชนีข้อมูลความรู้ (Vector Index) ได้ดังนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) เข้าสู่เว็บไซต์คอนโซลของ Pinecone (https://app.pinecone.io) และลงชื่อเข้าใช้", indent=0.75, space_after=2)
    add_para("2) เลือกดูรายละเอียด Index ประจำโครงงาน เช่น rice-disease-index", indent=0.75, space_after=2)
    add_para("3) ตรวจสอบข้อมูลสถิติที่สำคัญ ได้แก่ Total Vectors (จำนวนชิ้นส่วนข้อความ), Dimension (มิติเวกเตอร์), Metric (Cosine Similarity), และกราฟสถิติการใช้งาน (Usage & Request Latency)", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-10",
        "หน้าจอจัดการ Index บน Pinecone Vector Database Console",
        "ภาพหน้าจอแดชบอร์ดบนเว็บไซต์ Pinecone แสดงสถานะ Index จำนวนเวกเตอร์ และมิติข้อมูลของฐานความรู้โรคข้าว"
    )
    
    # ก.2.5
    add_para("ก.2.5 การอัปเดตและเพิ่มเอกสารความรู้โรคข้าว (Knowledge Ingestion Script)", bold=True, size=16, indent=0.25, space_before=6, space_after=3)
    add_para(
        "กรณีที่มีข้อมูลโรคข้าวใหม่ สารเคมีชนิดใหม่ หรือต้องการอัปเดตคู่มือวิชาการของกรมการข้าว ผู้ดูแลระบบสามารถเพิ่มข้อมูลเข้าสู่ระบบได้ดังนี้:",
        indent=0.5, space_after=2
    )
    add_para("1) นำไฟล์เอกสารคู่มือฉบับใหม่ (ไฟล์ .pdf หรือ .txt) ไปจัดเก็บไว้ในโฟลเดอร์คลังเอกสารของระบบ (data/knowledge_base/)", indent=0.75, space_after=2)
    add_para("2) เปิดโปรแกรม Terminal หรือ Command Prompt ในเครื่องเซิร์ฟเวอร์", indent=0.75, space_after=2)
    add_para("3) รันคำสั่งสคริปต์สำหรับการนำเข้าข้อมูลความรู้ เช่น python scripts/ingest_knowledge.py", indent=0.75, space_after=2)
    add_para("4) สคริปต์จะทำการอ่านเอกสาร ตัดแบ่งข้อความออกเป็นส่วนย่อย (Chunking), คำนวณค่า Vector Embeddings ด้วยโมเดลภาษา และอัปโหลดเวกเตอร์ชุดใหม่เข้าสู่ Pinecone Index โดยอัตโนมัติ ทำให้ระบบ RAG นำความรู้ใหม่ไปใช้ตอบคำถามได้ทันทีโดยไม่ต้องเทรนโมเดลใหม่", indent=0.75, space_after=8)
    add_placeholder_box(
        "ก-11",
        "หน้าต่างการรันสคริปต์อัปเดตเอกสารความรู้โรคข้าวเข้าสู่ Vector Database",
        "ภาพหน้าจอ Terminal ขณะรันสคริปต์ Ingestion แสดงขั้นตอนการอ่านไฟล์เอกสาร การทำ Chunking และการอัปโหลดเวกเตอร์ขึ้น Pinecone สำเร็จ"
    )
    
    # Output path
    out_docx = "Project/Appendix_A.docx"
    doc.save(out_docx)
    print(f"Successfully generated {out_docx}!")

if __name__ == "__main__":
    create_appendix_a()
