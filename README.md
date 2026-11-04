# C389 Project 1: People Counting Assistant
## Twist 3 — Distracted Pedestrian Detection

A fine-tuned computer vision solution built for **C389 AI Solution Design and Implementation (Project 1)**. This project extends a baseline walkway monitoring assistant into **Twist 3: Distracted Pedestrian Detection**, providing real-time, single-frame directional pedestrian counts alongside mobile phone distraction signals.

---

## 📌 Project Overview

Monitoring pedestrian traffic flow and identifying potential distraction hazards (e.g., pedestrians using mobile phones while walking) is critical for walkway safety. This system fine-tunes a **YOLO26 Nano (`yolo26n.pt`)** object detection model on a custom-built corridor dataset to detect directional movement and active phone usage without relying on complex, continuous frame tracking.

### Key Capabilities
- **Directional Pedestrian Counting:** Classifies pedestrians walking left vs. right across the frame.
- **Distraction Detection:** Identifies mobile phone usage paired with pedestrian movement.
- **Robust Deployment Service:** Features an end-to-end Gradio web service with strict input validation (`validate_image`) to handle corrupted or empty uploads gracefully.
- **Privacy-By-Design:** Operates strictly on single-frame detection without storing facial identities or personal data.

---

## 🏷️ Class Definitions

The model is trained on a 4-class target schema exported from Roboflow:

| Class ID | Class Name | Description |
| :---: | :--- | :--- |
| **0** | `person_left` | Pedestrian moving towards the left of the frame |
| **1** | `person_right` | Pedestrian moving towards the right of the frame |
| **2** | `phone_using_left` | Pedestrian holding/using a phone while moving left |
| **3** | `phone_using_right` | Pedestrian holding/using a phone while moving right |

> **Crucial Rule:** Horizontal flip augmentation (`fliplr`) is **strictly set to `0.0`** during training. Mirrored images invert walking direction without changing label associations, which would teach the model incorrect directional features.

---

## ⚙️ Model Architecture & Training Parameters

- **Pretrained Backbone:** `yolo26n.pt` (YOLO26 Nano variant)
- **Input Resolution:** `640 x 640`
- **Batch Size:** `16`
- **Epochs:** `100`
- **Optimizer:** `AdamW` (`lr0=0.01`, `lrf=0.01`, `momentum=0.937`, `weight_decay=0.0005`)
- **Augmentation Constraints:** `fliplr=0.0` (Horizontal flip disabled), mosaic & HSV active
- **Hardware Target:** NVIDIA Tesla T4 GPU (Google Colab runtime)

---

## 📁 Repository Structure

```
.
├── artifacts/
│   └── CA1_Twist3/                  # Production export folder
│       ├── best.pt                  # Top-performing weights (validated by mAP score)
│       └── baseline_metrics.json    # Metric benchmarks and configuration state
├── runs/
│   └── CA1_Twist3_Organised/        # Experiment outputs
│       ├── run1/                    # Training logs, weights, confusion matrix, PR curves
│       └── run2/                    # Subsequent experiment iterations
├── notebooks/
│   └── CA1_Twist3_Training.ipynb    # Clean 43-cell fine-tuning and evaluation notebook
├── app/
│   └── app.py                       # Gradio service script with validate_image()
├── experiments_summary.csv          # Master tracking log (overall & per-class metrics)
├── README.md                        # Project documentation
```

---

## 📊 Experiment Tracking & Evaluation

Every training iteration dynamically logs full overall and per-class metrics to `experiments_summary.csv`:

| Run Folder | Best Epoch | Overall Precision | Overall Recall | Overall mAP50 | Overall mAP50-95 | mAP50 (person_left) | mAP50 (person_right) | mAP50 (phone_using_left) | mAP50 (phone_using_right) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `run1` | 98 | 0.564 | 0.661 | **0.853** | 0.623 | 0.841 | 0.844 | 0.185 | 0.489 |

### Evaluated Metrics
- **Train vs. Validation Loss Curves:** Monitored via `box_loss` and `cls_loss` to identify optimal stopping points and prevent overfitting.
- **Confusion Matrix & Precision-Recall Curves:** Evaluated per class to detect direction confusion or recall drops on small objects (`phone`).
- **Error Review:** Qualitative inspection using `random.choice()` on differing test split predictions against annotations.

---

## 🚀 Quick Start & Installation

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
git clone <your-repository-url>
cd C389-Twist3-Distracted-Pedestrian
pip install -r requirements.txt
```

### 2. Dataset Download & Path Normalisation
In Google Colab or your local setup, ensure your Roboflow dataset is downloaded into `PROJECT_ROOT/dataset`. The notebook automatically runs `normalise_data_yaml(DATASET_DIR)` to convert relative paths into absolute paths.

### 3. Model Fine-Tuning
Open `notebooks/CA1_Twist3_Training.ipynb` in Colab with a T4 GPU runtime and run cells top-to-bottom:
- **Section 1–2:** Setup, directory creation under `runs/CA1_Twist3_Organised/`, and Roboflow download check.
- **Section 3–5:** Pretrained baseline comparison, hyperparameter verification (`fliplr=0.0`), and 100-epoch training.
- **Section 6–8:** Per-class evaluation, random error visualisation, and confidence thresholding (`CONFIDENCE_THRESHOLD = 0.40`).
- **Section 9:** Smart export to `artifacts/CA1_Twist3/best.pt` (replaces previous checkpoint only if the new mAP score is higher).

### 4. Running the Web Service
Launch the Gradio web interface locally or on Colab:
```bash
python app/app.py
```
The interface accepts image uploads, runs input validation, and returns structured per-image counts for pedestrian direction and phone distraction.

---

## 🛡️ Privacy & Ethical Considerations

1. **Facial Anonymity & Data Capture:** Images were captured in campus corridors across multiple viewing angles (side-on and 45°) adhering to standard public space recording protocols.
2. **No Identity Persistence:** Detection runs purely per frame. No tracking ID or biometric data is stored or processed across frames.
3. **Graceful Error Handling:** Uploaded images are validated before inference. Invalid, corrupted, or non-image files return structured JSON error codes rather than raising uncaught server exceptions.

---

## 👨‍💻 Course Information & Acknowledgements

- **Module:** C389 AI Solution Design and Implementation
- **Project:** Project 1 — People Counting Assistant (Twist 3)
- **Built With:** Ultralytics YOLO, Roboflow, PyTorch, and Gradio.
