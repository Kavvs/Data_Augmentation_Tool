# 🔍 Defect Detection using YOLOv8

This project is an AI-powered solution to automatically detect **defects in visual assets** (such as machine parts, industrial components, or manufactured products) using the YOLOv8 object detection framework.

Unlike generic classification tools, this system precisely **localizes defects** within the image and distinguishes them from "good" parts using bounding boxes. This is especially valuable in quality control pipelines where accuracy, speed, and transparency are critical.

---

## 💡 Project Highlights

### 🧠 Powered by YOLOv8
- Utilizes **YOLOv8** (You Only Look Once) — the latest version of the popular real-time object detection model by Ultralytics.
- Supports lightweight (`yolov8n.pt`) to high-accuracy (`yolov8l.pt`) model variants.

### 🔍 Binary Object Detection
- Classes:
  - `good` – Non-defective parts.
  - `defect` – Faulty parts.
- Each object is marked using precise bounding boxes, ensuring that even partial defects are detected.

### 🧰 Complete Custom Pipeline
- From raw images ➡ annotation ➡ preprocessing ➡ augmentation ➡ training.
- Includes **custom label generation** and auto-labeling scripts for repetitive use cases.
- Converts manual annotations (.xml) into YOLO-readable `.txt` files.

### 🛠️ Optimized Preprocessing
- Converts all images to **grayscale**, enhances contrast using **CLAHE**, denoises while preserving edges with **bilateral filtering**, and sharpens the final result.
- Applies intelligent cropping to remove unwanted white borders and maximize the usable area.

### ⚙️ Robust Augmentation
- Uses **Albumentations** to safely apply flipping, rotation, and geometric tweaks without corrupting annotations or reducing label fidelity.

---

## 🔍 Why This Project Stands Out

- ✅ **Clean training pipeline** tailored for industrial image datasets.
- ✅ **Annotation auto-generator** for consistent labels in high-volume datasets.
- ✅ **Balanced train/val splitting** that preserves reproducibility with optional shuffling seed.
- ✅ **Edge-preserving preprocessing** designed for noisy or artifact-heavy datasets.
- ✅ **Scalable** to other binary or multi-class object detection tasks with minimal code changes.

---

## 🚀 Use Cases

- Automated quality inspection in factories
- Surface defect detection in metal, glass, fabric, or PCB boards
- Packaging validation in consumer goods
- Visual anomaly detection in any grayscale or RGB imagery

---

## 🔧 Tech Stack

- **Python 3.9**
- **OpenCV** for preprocessing
- **Albumentations** for augmentation
- **Ultralytics YOLOv8** for object detection
- **LabelImg** (or programmatic XML generators) for annotation

---

## 🙋‍♀️ Developed by

**Kavya M**  
AI Developer Intern @ ITC Infotech  
Feel free to connect for feedback, collaboration, or contributions!

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).

