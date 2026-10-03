from ultralytics import RTDETR
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

out_dir = 'docs/thesis/figures'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'fig_4_6_detection_gallery.png')

model_path = 'training/runs/detect/runs/rtdetr_rice_v5/weights/best.pt'
model = RTDETR(model_path)

disease_info = [
    {
        'id': 'BLB',
        'en': 'Bacterial Leaf Blight',
        'th': 'โรคขอบใบแห้ง',
        'path': 'data/labeled_datacentric-blb/test/images/bacterial_leaf_blight_118_jpg.rf.8b2ae0a86439458896d3f12172289420.jpg',
        'color': (37, 99, 235), # Blue
        'bgr': (235, 99, 37),
        'map': 'mAP@0.5: 69.5%',
        'conf_str': 'Top Conf: 86.0%'
    },
    {
        'id': 'Brown Spot',
        'en': 'Brown Spot',
        'th': 'โรคใบจุดสีน้ำตาล',
        'path': 'data/labeled_datacentric-blb/test/images/brown_spot_009_jpg.rf.b7aadcd3b5b44b11999dba3b731e33b7.jpg',
        'color': (217, 119, 6), # Amber
        'bgr': (6, 119, 217),
        'map': 'mAP@0.5: 79.9%',
        'conf_str': 'Top Conf: 85.8%'
    },
    {
        'id': 'Leaf Blast',
        'en': 'Leaf Blast',
        'th': 'โรคใบไหม้',
        'path': 'data/labeled_datacentric-blb/test/images/leaf_blast_028_jpg.rf.3db33ed7f62be24ff8299b31ad63cd0a.jpg',
        'color': (220, 38, 38), # Red
        'bgr': (38, 38, 220),
        'map': 'mAP@0.5: 77.6%',
        'conf_str': 'Top Conf: 85.9%'
    },
    {
        'id': 'Leaf Scald',
        'en': 'Leaf Scald',
        'th': 'โรคใบขอบกาบแห้ง',
        'path': 'data/labeled_datacentric-blb/test/images/leaf_scald_015_jpg.rf.1e255ae997404ae696db35e4b2fa8c8d.jpg',
        'color': (147, 51, 234), # Purple
        'bgr': (234, 51, 147),
        'map': 'mAP@0.5: 54.5%',
        'conf_str': 'Top Conf: 86.2%'
    },
    {
        'id': 'Sheath Blight',
        'en': 'Sheath Blight',
        'th': 'โรคกาบใบแห้ง',
        'path': 'data/labeled_datacentric-blb/test/images/sheath_blight_031_jpg.rf.a021ca9fc3d909cd52927aa46f851c36.jpg',
        'color': (13, 148, 136), # Teal
        'bgr': (136, 148, 13),
        'map': 'mAP@0.5: 71.3%',
        'conf_str': 'Top Conf: 89.9%'
    }
]

# Run inference and draw boxes with custom high-aesthetic rendering
img_card_w = 540
img_card_h = 540

rendered_images = []
for d in disease_info:
    res = model.predict(d['path'], conf=0.45, imgsz=800, verbose=False)[0]
    orig_img = cv2.imread(d['path'])
    orig_img = cv2.resize(orig_img, (img_card_w, img_card_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Sort boxes by confidence descending
    sorted_boxes = sorted(res.boxes, key=lambda b: float(b.conf[0]), reverse=True)
    
    for i, box in enumerate(sorted_boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()
        # Scale to img_card_w/h
        x1 = max(0, min(img_card_w - 2, int(xyxy[0] / 640 * img_card_w)))
        y1 = max(0, min(img_card_h - 2, int(xyxy[1] / 640 * img_card_h)))
        x2 = max(0, min(img_card_w - 2, int(xyxy[2] / 640 * img_card_w)))
        y2 = max(0, min(img_card_h - 2, int(xyxy[3] / 640 * img_card_h)))
        
        # Draw box
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), d['bgr'], 3 if i < 3 else 2)
        
        # Only put text label on top detections to prevent extreme clutter if > 4 boxes
        if len(sorted_boxes) <= 4 or i < 3 or (d['id'] == 'Brown Spot' and i < 4):
            label_text = f"{model.names[cls_id]} {conf:.2f}"
            font_scale = 0.48
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Smart boundary handling
            # If x1 + tw + 8 > img_card_w, pull back
            lx = x1
            if lx + tw + 8 > img_card_w:
                lx = max(0, img_card_w - tw - 10)
                
            # If y1 is near top (less than 25 px), draw label INSIDE box
            if y1 < 25:
                ly_box_top = y1
                ly_box_bottom = y1 + th + 8
                ly_text = y1 + th + 4
            else:
                ly_box_top = y1 - th - 8
                ly_box_bottom = y1
                ly_text = y1 - 4
                
            cv2.rectangle(orig_img, (lx, ly_box_top), (lx + tw + 8, ly_box_bottom), d['bgr'], -1)
            cv2.putText(orig_img, label_text, (lx + 4, ly_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
    pil_im = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    rendered_images.append(pil_im)

# Canvas Layout: 3 columns x 2 rows
pad_x = 40
pad_y = 35
col_w = img_card_w
row_h = img_card_h + 90
header_h = 92

canvas_w = pad_x * 4 + col_w * 3
canvas_h = header_h + pad_y * 3 + row_h * 2

canvas = Image.new('RGB', (canvas_w, canvas_h), color=(248, 250, 252))
draw = ImageDraw.Draw(canvas)

# Fonts
font_title = ImageFont.truetype('C:/Windows/Fonts/leelawdb.ttf', 30)
font_sub = ImageFont.truetype('C:/Windows/Fonts/tahomabd.ttf', 18)
font_card_head = ImageFont.truetype('C:/Windows/Fonts/tahomabd.ttf', 18)
font_card_sub = ImageFont.truetype('C:/Windows/Fonts/tahoma.ttf', 15)
font_badge = ImageFont.truetype('C:/Windows/Fonts/tahomabd.ttf', 14)
font_summary_title = ImageFont.truetype('C:/Windows/Fonts/leelawdb.ttf', 24)
font_metric_num = ImageFont.truetype('C:/Windows/Fonts/leelawdb.ttf', 28)
font_metric_lbl = ImageFont.truetype('C:/Windows/Fonts/tahoma.ttf', 14)

# Header Banner
draw.rectangle([0, 0, canvas_w, header_h], fill=(30, 41, 59))
draw.text((canvas_w // 2, 32), "ตัวอย่างผลการตรวจจับรอยโรคข้าวครบทั้ง 5 โรค (Detection Gallery)", fill=(255, 255, 255), font=font_title, anchor="mm")
draw.text((canvas_w // 2, 68), "ผลการอนุมานบนชุดข้อมูลทดสอบจริง (Test Set) ด้วยโมเดล RT-DETR-L v5 (Real-Time Detection Transformer)", fill=(148, 163, 184), font=font_sub, anchor="mm")

# Place 5 disease cards
positions = [
    (0, 0), # Row 0, Col 0: BLB
    (0, 1), # Row 0, Col 1: Brown Spot
    (0, 2), # Row 0, Col 2: Leaf Blast
    (1, 0), # Row 1, Col 0: Leaf Scald
    (1, 1), # Row 1, Col 1: Sheath Blight
]

for idx, (r, c) in enumerate(positions):
    info = disease_info[idx]
    x = pad_x + c * (col_w + pad_x)
    y = header_h + pad_y + r * (row_h + pad_y)
    
    # Image
    canvas.paste(rendered_images[idx], (x, y))
    draw.rectangle([x - 1, y - 1, x + col_w, y + img_card_h], outline=(203, 213, 225), width=2)
    
    # Top class badge on image
    draw.rounded_rectangle([x + 12, y + 12, x + 240, y + 44], radius=6, fill=info['color'])
    draw.text((x + 126, y + 28), f"คลาส {idx+1}: {info['en']}", fill=(255, 255, 255), font=font_badge, anchor="mm")
    
    # Bottom info pill
    pill_y = y + img_card_h + 10
    draw.rounded_rectangle([x, pill_y, x + col_w, pill_y + 70], radius=8, fill=(255, 255, 255), outline=(226, 232, 240), width=1)
    
    # Left color strip
    draw.rounded_rectangle([x + 2, pill_y + 2, x + 10, pill_y + 68], radius=3, fill=info['color'])
    
    # Text
    draw.text((x + 22, pill_y + 12), f"{idx+1}. {info['th']} ({info['en']})", fill=(15, 23, 42), font=font_card_head)
    desc_txt = f"{info['map']}  |  {info['conf_str']}  |  IoU Threshold: 0.50"
    draw.text((x + 22, pill_y + 40), desc_txt, fill=(100, 116, 139), font=font_card_sub)

# Slot 6 (Row 1, Col 2): Executive Model Summary Card
slot6_x = pad_x + 2 * (col_w + pad_x)
slot6_y = header_h + pad_y + 1 * (row_h + pad_y)
slot6_h = img_card_h + 80

draw.rounded_rectangle([slot6_x, slot6_y, slot6_x + col_w, slot6_y + slot6_h], radius=12, fill=(15, 23, 42), outline=(51, 65, 85), width=2)

# Slot 6 Header
draw.text((slot6_x + col_w // 2, slot6_y + 35), "สรุปสมรรถนะ RT-DETR-L v5", fill=(255, 255, 255), font=font_summary_title, anchor="mm")
draw.text((slot6_x + col_w // 2, slot6_y + 65), "Real-Time Detection Transformer (Final Model)", fill=(148, 163, 184), font=font_badge, anchor="mm")

# Divider
draw.line([slot6_x + 30, slot6_y + 85, slot6_x + col_w - 30, slot6_y + 85], fill=(51, 65, 85), width=1)

# Metrics Grid (2 columns x 2 rows)
metric_boxes = [
    {"val": "70.56%", "lbl": "mAP@0.5 (Overall)", "color": (34, 197, 94), "bx": 0, "by": 0},
    {"val": "35.54%", "lbl": "mAP@0.5:0.95", "color": (56, 189, 248), "bx": 1, "by": 0},
    {"val": "73.51%", "lbl": "Precision (P)", "color": (251, 191, 36), "bx": 0, "by": 1},
    {"val": "70.17%", "lbl": "Recall (R)", "color": (168, 85, 247), "bx": 1, "by": 1},
]

mb_w = (col_w - 80) // 2
mb_h = 90
grid_start_y = slot6_y + 105

for mb in metric_boxes:
    mx = slot6_x + 30 + mb['bx'] * (mb_w + 20)
    my = grid_start_y + mb['by'] * (mb_h + 15)
    draw.rounded_rectangle([mx, my, mx + mb_w, my + mb_h], radius=8, fill=(30, 41, 59), outline=(51, 65, 85), width=1)
    draw.text((mx + mb_w // 2, my + 32), mb['val'], fill=mb['color'], font=font_metric_num, anchor="mm")
    draw.text((mx + mb_w // 2, my + 65), mb['lbl'], fill=(148, 163, 184), font=font_metric_lbl, anchor="mm")

# Bottom Highlights
info_y = grid_start_y + 2 * (mb_h + 15) + 15
highlights = [
    "- ความเร็วประมวลผล (GPU Latency): 6.8 ms (~147 FPS)",
    "- ความละเอียดภาพนำเข้า: 800 x 800 px (Letterbox)",
    "- ฮาร์ดแวร์ฝึกฝน: NVIDIA GeForce RTX 4090 24GB",
    "- การแก้ไขฉลาก BLB: ยุบรวมกล่องต่อเนื่อง (Data-Centric)",
    "- สถาปัตยกรรม: Transformer Vision Backbone + CCFM",
    "- ความพร้อมใช้งาน: ทำงานจริงบน LINE Official Account"
]

for i, hl in enumerate(highlights):
    draw.text((slot6_x + 30, info_y + i * 30), hl, fill=(226, 232, 240), font=font_card_sub)

# Save
canvas.save(out_path, quality=95)
print(f"Successfully generated refined {out_path} ({canvas_w}x{canvas_h})")
