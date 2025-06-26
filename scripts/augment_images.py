import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

# --------------------- Albumentations (light-safe transforms only) ------------------------
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.ShiftScaleRotate(
        shift_limit=0.02, scale_limit=0.03, rotate_limit=3,
        border_mode=cv2.BORDER_REFLECT_101, p=0.4
    )
])

# --------------------- Remove white borders / background ------------------------
def crop_white_borders(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)  # white areas
    contours, _ = cv2.findContours(~thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        return image[y:y+h, x:x+w]
    return image

# --------------------- Grayscale + Enhance + Denoise + Sharpen ------------------------
def preprocess_image(img):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Slight sharpening (high-pass filter)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # Convert to 3 channels
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

# --------------------- Augmentation Runner ------------------------
def augment_images(input_path, output_path, target_count):
    os.makedirs(output_path, exist_ok=True)
    image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = []

    for file in image_files:
        full_path = os.path.join(input_path, file)
        img = cv2.imread(full_path)

        if img is not None:
            cropped = crop_white_borders(img)
            preprocessed = preprocess_image(cropped)
            images.append((file, preprocessed))
        else:
            print(f"❌ Skipped unreadable image: {full_path}")

    if not images:
        print(f"❌ No valid images found in {input_path}")
        return

    total_augmented = 0
    img_idx = 0
    pbar = tqdm(total=target_count, desc=f"Augmenting {os.path.basename(output_path)}")

    while total_augmented < target_count:
        filename, base_img = images[img_idx % len(images)]
        augmented = transform(image=base_img)['image']

        base_name = os.path.splitext(filename)[0]
        save_name = f"{base_name}_aug_{total_augmented}.jpg"
        save_path = os.path.join(output_path, save_name)

        # Save with 100% JPEG quality
        cv2.imwrite(save_path, augmented, [cv2.IMWRITE_JPEG_QUALITY, 100])

        total_augmented += 1
        img_idx += 1
        pbar.update(1)

    pbar.close()

# --------------------- Run on both classes ------------------------
augment_images("../original_images/defective", "../augmented_dataset/defective", 50)
augment_images("../original_images/good", "../augmented_dataset/good", 50)
