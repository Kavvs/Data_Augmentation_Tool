import os
import xml.etree.ElementTree as ET
import shutil
import random

# === CONFIGURATION ===
ANNOTATIONS_DIR = r"C:\Users\Kavya M\Downloads\Data_Augmentation_Tool\annotations"
LABELS_TRAIN_DIR = r"C:\Users\Kavya M\Downloads\Data_Augmentation_Tool\dataset\labels\train"
LABELS_VAL_DIR = r"C:\Users\Kavya M\Downloads\Data_Augmentation_Tool\dataset\labels\val"


SPLIT_RATIO = 0.8  # 80% train, 20% val
SEED = 42

# === Ensure output folders exist ===
os.makedirs(LABELS_TRAIN_DIR, exist_ok=True)
os.makedirs(LABELS_VAL_DIR, exist_ok=True)

# === Class dictionary ===
CLASS_MAP = {
    "defective": 0,
    "good": 1
}

def convert_xml_to_yolo(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)

    yolo_lines = []

    for obj in root.iter('object'):
        cls = obj.find('name').text.lower()
        if cls not in CLASS_MAP:
            continue

        cls_id = CLASS_MAP[cls]
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))

        # Normalize
        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return yolo_lines

# === Main Split and Convert ===
all_xmls = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.xml')]
random.shuffle(all_xmls)

# === FIXED SHUFFLE ===
random.seed(SEED)
all_xmls.sort()  
random.shuffle(all_xmls)

split_index = int(len(all_xmls) * SPLIT_RATIO)
train_xmls = all_xmls[:split_index]
val_xmls = all_xmls[split_index:]

def save_txt(yolo_data, target_txt_path):
    with open(target_txt_path, 'w') as f:
        for line in yolo_data:
            f.write(line + '\n')

for xml_file in train_xmls:
    full_path = os.path.join(ANNOTATIONS_DIR, xml_file)
    yolo_data = convert_xml_to_yolo(full_path)
    base_name = os.path.splitext(xml_file)[0]
    save_txt(yolo_data, os.path.join(LABELS_TRAIN_DIR, base_name + ".txt"))

for xml_file in val_xmls:
    full_path = os.path.join(ANNOTATIONS_DIR, xml_file)
    yolo_data = convert_xml_to_yolo(full_path)
    base_name = os.path.splitext(xml_file)[0]
    save_txt(yolo_data, os.path.join(LABELS_VAL_DIR, base_name + ".txt"))

print(f"✅ Converted and split {len(train_xmls)} train + {len(val_xmls)} val labels.")
