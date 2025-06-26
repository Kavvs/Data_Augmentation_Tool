import os
import shutil
import random

# === CONFIGURATION ===
SOURCE_DIR = r"C:\Users\Kavya M\Downloads\Data_Augmentation_Tool\all_images"
TRAIN_DIR = r"C:\Users\Kavya M\Downloads\Data_Augmentation_Tool\dataset\images\train"
VAL_DIR = r"C:\Users\Kavya M\Downloads\Data_Augmentation_Tool\dataset\images\val"
SPLIT_RATIO = 0.8  # 80% train, 20% val
SEED = 42

# === ENSURE DESTINATIONS EXIST ===
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# === GET ALL IMAGE FILES ===
image_extensions = (".jpg", ".jpeg", ".png")
all_images = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(image_extensions)]
random.shuffle(all_images)

# === FIXED SHUFFLE ===
random.seed(SEED)
all_images.sort()  
random.shuffle(all_images)

# === SPLIT ===
split_index = int(len(all_images) * SPLIT_RATIO)
train_images = all_images[:split_index]
val_images = all_images[split_index:]

# === COPY FILES ===
def copy_files(file_list, destination):
    for filename in file_list:
        src_path = os.path.join(SOURCE_DIR, filename)
        dst_path = os.path.join(destination, filename)
        shutil.copy2(src_path, dst_path)

# === PERFORM COPY ===
copy_files(train_images, TRAIN_DIR)
copy_files(val_images, VAL_DIR)

print(f"✅ Split complete: {len(train_images)} train images / {len(val_images)} val images.")
