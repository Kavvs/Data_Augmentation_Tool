from PIL import Image
import os

input_folder = r"C:\Users\Kavya M\Downloads\Data_Augmentation_Tool\Data_augmentation\input"
output_folder = r"C:\Users\Kavya M\Downloads\Data_Augmentation_Tool\Data_augmentation\output"
target_size = (224, 224)

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(input_folder, filename)
        try:
            with Image.open(img_path) as img:
                resized = img.resize(target_size, Image.Resampling.LANCZOS)
                resized.save(os.path.join(output_folder, filename))
        except Exception as e:
            print(f"Error resizing {filename}: {e}")

print("✅ All images resized to 224x224.")
