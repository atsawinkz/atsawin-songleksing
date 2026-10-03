import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

out_dir = 'docs/thesis/figures'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'fig_4_9_segmentation_ablation.png')

# Paths for Scald and Spot pairs
natural_dir = 'data/labeled_datacentric-blb/test/images/'
seg_dir = 'experiments/segmentation_study/dataset_segmented/test/images/'

scald_file = 'leaf_scald_097_jpg.rf.ebfccd17c3d4128ce8507b9d0fc8cad4.jpg'
spot_file = 'brown_spot_038_jpg.rf.86213da3f47416fb4d4500aa007f835a.jpg'

# Load images
scald_nat = cv2.imread(os.path.join(natural_dir, scald_file))
scald_seg = cv2.imread(os.path.join(seg_dir, scald_file))

spot_nat = cv2.imread(os.path.join(natural_dir, spot_file))
spot_seg = cv2.imread(os.path.join(seg_dir, spot_file))

# Resize image panels
panel_w = 460
panel_h = 460

scald_nat_pil = Image.fromarray(cv2.cvtColor(cv2.resize(scald_nat, (panel_w, panel_h), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB))
scald_seg_pil = Image.fromarray(cv2.cvtColor(cv2.resize(scald_seg, (panel_w, panel_h), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB))
spot_nat_pil = Image.fromarray(cv2.cvtColor(cv2.resize(spot_nat, (panel_w, panel_h), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB))
spot_seg_pil = Image.fromarray(cv2.cvtColor(cv2.resize(spot_seg, (panel_w, panel_h), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB))

# Dimensions
pad_x = 45
pad_y = 30
header_h = 90
footer_h = 80
row_h = panel_h + 30
card_w = 640

canvas_w = pad_x * 2 + panel_w * 2 + 25 + card_w + 25 # 90 + 920 + 25 + 640 + 25 = 1700
canvas_h = header_h + pad_y + row_h * 2 + pad_y + footer_h # 90 + 30 + 490*2 + 30 + 80 = 1210

canvas = Image.new('RGB', (canvas_w, canvas_h), color=(248, 250, 252))
draw = ImageDraw.Draw(canvas)

# Fonts
font_title = ImageFont.truetype('C:/Windows/Fonts/leelawdb.ttf', 30)
font_sub = ImageFont.truetype('C:/Windows/Fonts/tahomabd.ttf', 18)
font_row_title = ImageFont.truetype('C:/Windows/Fonts/tahomabd.ttf', 20)
font_panel_badge = ImageFont.truetype('C:/Windows/Fonts/tahomabd.ttf', 14)
font_card_head = ImageFont.truetype('C:/Windows/Fonts/tahomabd.ttf', 21)
font_card_metric = ImageFont.truetype('C:/Windows/Fonts/leelawdb.ttf', 32)
font_card_sub = ImageFont.truetype('C:/Windows/Fonts/tahomabd.ttf', 15)
font_card_desc = ImageFont.truetype('C:/Windows/Fonts/tahoma.ttf', 15)
font_footer = ImageFont.truetype('C:/Windows/Fonts/tahomabd.ttf', 16)

# Header Banner
draw.rectangle([0, 0, canvas_w, header_h], fill=(30, 41, 59))
draw.text((canvas_w // 2, 32), "การเปรียบเทียบผลการตัดพื้นหลังภาพแปลงนาด้วย MobileSAM (Ablation Study Visuals)", fill=(255, 255, 255), font=font_title, anchor="mm")
draw.text((canvas_w // 2, 68), "การศึกษาผลกระทบทางสัณฐานวิทยา: ความแตกต่างระหว่างรอยโรคขนาดใหญ่ (Macro-lesion) และรอยโรคขนาดเล็ก (Micro-lesion)", fill=(148, 163, 184), font=font_sub, anchor="mm")

# --- ROW 1: Leaf Scald (Macro-Lesion Benefit) ---
y1 = header_h + pad_y
x1_nat = pad_x
x1_seg = x1_nat + panel_w + 25
x1_card = x1_seg + panel_w + 25

# Panel 1: Scald Natural
canvas.paste(scald_nat_pil, (x1_nat, y1))
draw.rectangle([x1_nat - 1, y1 - 1, x1_nat + panel_w, y1 + panel_h], outline=(203, 213, 225), width=2)
draw.rounded_rectangle([x1_nat + 12, y1 + 12, x1_nat + 220, y1 + 44], radius=6, fill=(30, 41, 59))
draw.text((x1_nat + 116, y1 + 28), "ภาพถ่ายแปลงนาจริง", fill=(255, 255, 255), font=font_panel_badge, anchor="mm")

# Panel 2: Scald Segmented
canvas.paste(scald_seg_pil, (x1_seg, y1))
draw.rectangle([x1_seg - 1, y1 - 1, x1_seg + panel_w, y1 + panel_h], outline=(203, 213, 225), width=2)
draw.rounded_rectangle([x1_seg + 12, y1 + 12, x1_seg + 240, y1 + 44], radius=6, fill=(16, 185, 129))
draw.text((x1_seg + 126, y1 + 28), "ตัดพื้นหลังด้วย MobileSAM", fill=(255, 255, 255), font=font_panel_badge, anchor="mm")

# Panel 3: Scald Impact Card (Green theme)
draw.rounded_rectangle([x1_card, y1, x1_card + card_w, y1 + panel_h], radius=10, fill=(240, 253, 244), outline=(134, 239, 172), width=2)
draw.text((x1_card + 25, y1 + 25), "กรณีรอยโรคขนาดใหญ่: โรคใบขอบกาบแห้ง (Leaf Scald)", fill=(21, 128, 61), font=font_card_head)

# Metric pill
draw.rounded_rectangle([x1_card + 25, y1 + 65, x1_card + card_w - 25, y1 + 135], radius=8, fill=(220, 252, 231), outline=(74, 222, 128), width=1)
draw.text((x1_card + 45, y1 + 83), "mAP@0.5:", fill=(22, 101, 52), font=font_card_sub)
draw.text((x1_card + 140, y1 + 80), "54.50%  ->  65.67%", fill=(21, 128, 61), font=font_card_metric)
draw.text((x1_card + 480, y1 + 80), "(+11.17%)", fill=(22, 163, 74), font=font_card_metric)

scald_desc = (
    "- กลไกที่ได้ประโยชน์: แผลใบขอบกาบแห้งมีลักษณะเป็นปื้นคลื่นยาวตามขอบใบ\n"
    "- ผลการตัดฉากหลัง: ขจัดกอข้าวข้างเคียงและวัชพืชพื้นน้ำ (Visual Distractors) ออกไปได้หมด\n"
    "- การเรียนรู้ของโมเดล: โมเดลสามารถจับรูปทรงขอบแผลสัณฐานวิทยาของใบเดี่ยวได้ชัดเจนขึ้น\n"
    "- บทวิเคราะห์: ส่งผลให้ค่าความแม่นยำเฉลี่ย mAP@0.5 เพิ่มขึ้นอย่างมีนัยสำคัญ (+11.17%)\n"
    "  เช่นเดียวกับโรคกาบใบแห้ง (+4.43%) และโรคขอบใบแห้ง (+5.82%)"
)
draw.text((x1_card + 25, y1 + 155), scald_desc, fill=(20, 83, 45), font=font_card_desc, spacing=8)


# --- ROW 2: Brown Spot (Micro-Lesion Collapse) ---
y2 = y1 + row_h + 15
x2_nat = pad_x
x2_seg = x2_nat + panel_w + 25
x2_card = x2_seg + panel_w + 25

# Panel 1: Spot Natural
canvas.paste(spot_nat_pil, (x2_nat, y2))
draw.rectangle([x2_nat - 1, y2 - 1, x2_nat + panel_w, y2 + panel_h], outline=(203, 213, 225), width=2)
draw.rounded_rectangle([x2_nat + 12, y2 + 12, x2_nat + 220, y2 + 44], radius=6, fill=(30, 41, 59))
draw.text((x2_nat + 116, y2 + 28), "ภาพถ่ายแปลงนาจริง", fill=(255, 255, 255), font=font_panel_badge, anchor="mm")

# Panel 2: Spot Segmented
canvas.paste(spot_seg_pil, (x2_seg, y2))
draw.rectangle([x2_seg - 1, y2 - 1, x2_seg + panel_w, y2 + panel_h], outline=(203, 213, 225), width=2)
draw.rounded_rectangle([x2_seg + 12, y2 + 12, x2_seg + 240, y2 + 44], radius=6, fill=(225, 29, 72))
draw.text((x2_seg + 126, y2 + 28), "ตัดพื้นหลังด้วย MobileSAM", fill=(255, 255, 255), font=font_panel_badge, anchor="mm")

# Panel 3: Spot Impact Card (Red theme)
draw.rounded_rectangle([x2_card, y2, x2_card + card_w, y2 + panel_h], radius=10, fill=(254, 242, 242), outline=(252, 165, 165), width=2)
draw.text((x2_card + 25, y2 + 25), "กรณีรอยโรคขนาดเล็ก: โรคใบจุดสีน้ำตาล (Brown Spot)", fill=(185, 28, 28), font=font_card_head)

# Metric pill
draw.rounded_rectangle([x2_card + 25, y2 + 65, x2_card + card_w - 25, y2 + 135], radius=8, fill=(254, 226, 226), outline=(248, 113, 113), width=1)
draw.text((x2_card + 45, y2 + 83), "mAP@0.5:", fill=(153, 27, 27), font=font_card_sub)
draw.text((x2_card + 140, y2 + 80), "79.90%  ->  64.71%", fill=(185, 28, 28), font=font_card_metric)
draw.text((x2_card + 480, y2 + 80), "(-15.19%)", fill=(220, 38, 38), font=font_card_metric)

spot_desc = (
    "- กลไกที่เกิดปัญหา: แผลใบจุดสีน้ำตาลมีขนาดเล็กมาก (Punctate Lesions 1-5 มิลลิเมตร)\n"
    "- การสูญเสียบริบทแวดล้อม: แผลจุดเล็กต้องอาศัยคอนทราสต์กับเนื้อใบข้าวสีเขียวรอบแผล\n"
    "- ผลกระทบจากการ Mask: ขอบใบไม้ถูกตัดแหว่งเป็นสีดำทึบ ทำให้โมเดลสับสนระหว่างขอบมืดกับรอยโรค\n"
    "- ความแม่นยำร่วงลงอย่างรุนแรง: ค่า mAP@0.5 ดิ่งลงถึง -15.19% (เหลือเพียง 64.71%)\n"
    "- ปัญหาเวลาประมวลผล: SAM เพิ่มเวลาทำงานเป็น 100-170 ms (ช้ากว่าโมเดลภาพจริง 15-20 เท่า)"
)
draw.text((x2_card + 25, y2 + 155), spot_desc, fill=(127, 29, 29), font=font_card_desc, spacing=8)


# --- FOOTER BANNER: Architecture Decision ---
footer_y = canvas_h - footer_h - 15
draw.rounded_rectangle([pad_x, footer_y, canvas_w - pad_x, footer_y + footer_h], radius=8, fill=(15, 23, 42), outline=(51, 65, 85), width=2)
decision_title = "ข้อสรุปการเลือกสถาปัตยกรรม (Architecture Decision): เลือกระบบตรวจจับภาพจริงโดยตรง (Direct Detection v5)"
decision_sub = "ไม่ตัดพื้นหลัง เนื่องจากรักษาความแม่นยำของโรคแผลเล็กได้สูงกว่า (79.9%), ประมวลผลเร็วระดับ Real-Time เพียง 6.8 ms และประหยัดทรัพยากรเซิร์ฟเวอร์"
draw.text((canvas_w // 2, footer_y + 25), decision_title, fill=(56, 189, 248), font=font_footer, anchor="mm")
draw.text((canvas_w // 2, footer_y + 53), decision_sub, fill=(226, 232, 240), font=font_card_desc, anchor="mm")

# Save
canvas.save(out_path, quality=95)
print(f"Successfully generated {out_path} ({canvas_w}x{canvas_h})")
