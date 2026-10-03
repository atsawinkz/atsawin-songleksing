import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Paths
img1_path = 'data/labeled/train/images/bacterial_leaf_blight_547_jpg.rf.ba5245f8f779bcf4dec3f537cfc30836.jpg'
lbl1_path = 'data/labeled/train/labels/bacterial_leaf_blight_547_jpg.rf.ba5245f8f779bcf4dec3f537cfc30836.txt'
img2_path = 'data/labeled_datacentric-blb/train/images/bacterial_leaf_blight_547_jpg.rf.de18ac1f7d812c89d51be0f4fa228f72.jpg'
lbl2_path = 'data/labeled_datacentric-blb/train/labels/bacterial_leaf_blight_547_jpg.rf.de18ac1f7d812c89d51be0f4fa228f72.txt'

out_dir = 'docs/thesis/figures'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'fig_4_4_blb_datacentric_comparison.png')

# Load images
im1 = cv2.imread(img1_path)
im2 = cv2.imread(img2_path)

# Resize both to standard 720x720 for crisp rendering
target_size = 720
im1 = cv2.resize(im1, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
im2 = cv2.resize(im2, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

# Draw boxes on im1 (Red / Coral)
with open(lbl1_path) as f:
    boxes1 = [line.strip().split() for line in f if line.strip()]

for i, b in enumerate(boxes1):
    cls, cx, cy, bw, bh = map(float, b)
    x1 = int((cx - bw / 2) * target_size)
    y1 = int((cy - bh / 2) * target_size)
    x2 = int((cx + bw / 2) * target_size)
    y2 = int((cy + bh / 2) * target_size)
    cv2.rectangle(im1, (x1, y1), (x2, y2), (30, 40, 225), 2)

# Draw boxes on im2 (Green / Emerald)
with open(lbl2_path) as f:
    boxes2 = [line.strip().split() for line in f if line.strip()]

for i, b in enumerate(boxes2):
    cls, cx, cy, bw, bh = map(float, b)
    x1 = int((cx - bw / 2) * target_size)
    y1 = int((cy - bh / 2) * target_size)
    x2 = int((cx + bw / 2) * target_size)
    y2 = int((cy + bh / 2) * target_size)
    cv2.rectangle(im2, (x1, y1), (x2, y2), (35, 175, 35), 3)

# Convert to PIL
pil1 = Image.fromarray(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
pil2 = Image.fromarray(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))

# Create Composite Canvas: width = 1600, height = 1040
canvas_w = 1600
canvas_h = 1040

canvas = Image.new('RGB', (canvas_w, canvas_h), color=(248, 250, 252))
draw = ImageDraw.Draw(canvas)

# Fonts
font_title = ImageFont.truetype('C:/Windows/Fonts/leelawdb.ttf', 30)
font_sub = ImageFont.truetype('C:/Windows/Fonts/tahomabd.ttf', 19)
font_card_title = ImageFont.truetype('C:/Windows/Fonts/tahomabd.ttf', 20)
font_card_desc = ImageFont.truetype('C:/Windows/Fonts/tahoma.ttf', 16)
font_badge = ImageFont.truetype('C:/Windows/Fonts/tahomabd.ttf', 15)

# Title Banner
draw.rectangle([0, 0, canvas_w, 88], fill=(30, 41, 59))
draw.text((canvas_w // 2, 30), "การเปรียบเทียบฉลากข้อมูลโรคขอบใบแห้ง (Bacterial Leaf Blight: BLB)", fill=(255, 255, 255), font=font_title, anchor="mm")
draw.text((canvas_w // 2, 66), "Data-Centric AI: การแก้ไขปัญหา Label Fragmentation สู่ Continuous Morphological Annotation", fill=(148, 163, 184), font=font_sub, anchor="mm")

# Left Image Position: x=55, y=110
x_left = 55
x_right = target_size + 125
y_top = 110

# Paste images
canvas.paste(pil1, (x_left, y_top))
canvas.paste(pil2, (x_right, y_top))

# Draw border around images
draw.rectangle([x_left - 1, y_top - 1, x_left + target_size, y_top + target_size], outline=(203, 213, 225), width=2)
draw.rectangle([x_right - 1, y_top - 1, x_right + target_size, y_top + target_size], outline=(203, 213, 225), width=2)

# Overlay Badges on Top-Left of each image
# Left badge (Red)
badge1_w = 340
draw.rounded_rectangle([x_left + 15, y_top + 15, x_left + 15 + badge1_w, y_top + 52], radius=6, fill=(220, 38, 38))
draw.text((x_left + 15 + badge1_w // 2, y_top + 33), "ฉลากเดิม: ซอยกรอบย่อย 17 กรอบ", fill=(255, 255, 255), font=font_badge, anchor="mm")

# Right badge (Green)
badge2_w = 380
draw.rounded_rectangle([x_right + 15, y_top + 15, x_right + 15 + badge2_w, y_top + 52], radius=6, fill=(22, 163, 74))
draw.text((x_right + 15 + badge2_w // 2, y_top + 33), "ฉลากใหม่ (Data-Centric): กรอบต่อเนื่อง 2 กรอบ", fill=(255, 255, 255), font=font_badge, anchor="mm")

# Footer Cards
card_y = y_top + target_size + 15
card_h = 160

# Left Card (Problem & Result)
draw.rounded_rectangle([x_left, card_y, x_left + target_size, card_y + card_h], radius=8, fill=(254, 242, 242), outline=(252, 165, 165), width=1)
draw.text((x_left + 20, card_y + 20), "(ก) ก่อนปรับปรุง (Version 1-4): Fragmented Bounding Boxes", fill=(185, 28, 28), font=font_card_title)
desc1 = "- ปัญหา: กรอบสี่เหลี่ยมซอยย่อยซ้อนทับกันมากเกินไป (5,922 กล่องทั่วทั้งชุดข้อมูล)\n- ผลกระทบ: โมเดลสับสนในการตรวจหาขอบเขตแผลแท้จริง เกิด False Positive ซ้ำซ้อน\n- ประสิทธิภาพโมเดล: mAP@0.5 = 7.42% (ตรวจจับแทบไม่ได้ผล)"
draw.text((x_left + 20, card_y + 55), desc1, fill=(70, 20, 20), font=font_card_desc, spacing=6)

# Right Card (Solution & Result)
draw.rounded_rectangle([x_right, card_y, x_right + target_size, card_y + card_h], radius=8, fill=(240, 253, 244), outline=(134, 239, 172), width=1)
draw.text((x_right + 20, card_y + 20), "(ข) หลังปรับปรุง (Version 5 Data-Centric AI): Continuous Bounding Boxes", fill=(21, 128, 61), font=font_card_title)
desc2 = "- การแก้ไข: ยุบรวมกล่องย่อยเป็นกรอบเดียวต่อเนื่องตามแนวสัณฐานวิทยาของแผลจริง\n- ผลลัพธ์ข้อมูล: ลดความซ้ำซ้อนของกรอบลง 68.5% (เหลือ 1,863 กล่องที่สะอาดและแม่นยำ)\n- ประสิทธิภาพโมเดล: mAP@0.5 พุ่งขึ้นเป็น 69.50% (+62.1% Absolute Increase)"
draw.text((x_right + 20, card_y + 55), desc2, fill=(20, 70, 30), font=font_card_desc, spacing=6)

# Save
canvas.save(out_path, quality=95)
print(f"Successfully generated refined {out_path} ({canvas_w}x{canvas_h})")
