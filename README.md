# People Counting Assistant (Twist 3: Distracted Pedestrian Detection) — Streamlit Edition

A production-ready computer vision solution fine-tuned to detect, classify, and count directional pedestrian traffic and distracted mobile phone usage on campus walkways. Built with **Ultralytics YOLO26** and deployed as an interactive dashboard using **Streamlit**.

---

## 📌 Project Overview & Twist 3 Design

Baseline vision assistants detect generic pedestrians (`person`), but fail to differentiate direction of travel or identify distracted behaviors. **Twist 3** extends the baseline model by training a 4-class unified object detector:

| Class ID | Class Name | Description |
| :---: | :--- | :--- |
| `0` | **`person_left`** | Pedestrian walking towards the left (non-distracted) |
| `1` | **`person_right`** | Pedestrian walking towards the right (non-distracted) |
| `2` | **`phone_using_left`** | Pedestrian walking towards the left while using a mobile phone |
| `3` | **`phone_using_right`** | Pedestrian walking towards the right while using a mobile phone |

---

## 📊 Final Model Training & Validation Results (`run7`)

The model was fine-tuned using `yolo26n.pt` on Roboflow Dataset v10 (**617 Train**, **174 Validation**, **167 Test** images) for 100 epochs using the `AdamW` optimizer (`lr0=0.001`, `cos_lr=True`).

### 1. Overall Test Set Performance (167 Held-Out Images)

- **Overall mAP@0.5:** **`0.930` (93.0%)**
- **Overall mAP@0.5:0.95:** **`0.791` (79.1%)**
- **Overall Precision:** **`0.879` (87.9%)**
- **Overall Recall:** **`0.903` (90.3%)**

### 2. Per-Class Performance Breakdown

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`phone_using_right`** | `0.895` | `0.949` | **`0.972` (97.2%)** | `0.878` | Top Performer |
| **`phone_using_left`** | `0.889` | `0.946` | **`0.937` (93.7%)** | `0.819` | Excellent |
| **`person_left`** | `0.874` | `0.858` | **`0.910` (91.0%)** | `0.730` | Passed 90% Target |
| **`person_right`** | `0.856` | `0.860` | **`0.901` (90.1%)** | `0.735` | Passed 90% Target |

> **Class Performance Gap:** The gap between the best (`0.972`) and worst (`0.901`) class is only **`0.071`** (well below the `0.10` imbalance threshold). This proves all 4 categories were learned equally well without class imbalance issues.

### 3. Loss Convergence & Training Dynamics

- **Final Epoch Losses:** `train/box_loss`: `0.694` | `val/box_loss`: `0.677` | `train/cls_loss`: `0.512` | `val/cls_loss`: `0.575`
- **Peak Validation Epoch:** Epoch 90 (mAP@0.5: `0.917`, mAP@0.5:0.95: `0.760`).
- **Overfitting Diagnosis:** Validation box and class losses closely tracked training losses without diverging upward at late epochs, confirming healthy generalization.
- **Mandatory Augmentation Rule:** `fliplr` was fixed at `0.0`. Horizontal flipping mirrors the image while retaining labels, which corrupts directional feature learning.

---

## 🚀 Streamlit Application Architecture (`app.py`)

The application is optimized for deployment on **Streamlit Community Cloud** (Linux headless runtime).

### Key Features & Design Rules:

1. **Performance Optimization:**
   - Pre-loads model weights (`models/best.pt`) using `@st.cache_resource` to minimize cold-start latency.
   - Non-blocking OpenCV/PIL image processing for real-time inference rendering.

2. **Sidebar Controls:**
   - **Model Info Card:** Version `v1.0.0`, Architecture `YOLO26n`.
   - **Confidence Threshold Slider:** Default `0.40` (Range: `0.10`–`0.90`).
   - **NMS IoU Threshold Slider:** Default `0.45` (Range: `0.20`–`0.80`).
   - **Class Visibility Toggles:** Toggle display of individual classes and confidence score labels.

3. **KPI Summary Cards:**
   - **Total Pedestrians Detected:** Aggregate count.
   - **Walking Left Count:** `person_left` + `phone_using_left`.
   - **Walking Right Count:** `person_right` + `phone_using_right`.
   - **Distraction Rate (%):** Colored badges (Green < 15%, Yellow 15–30%, Red > 30%).

4. **Visual Output & Analytics:**
   - **Split-Screen View:** Column 1 shows annotated bounding box overlays; Column 2 displays class distribution breakdowns and confidence averages.
   - **Inference Timing Indicator:** Reports exact per-image inference speed in seconds.

5. **Developer API & JSON Export:**
   - Dedicated expander for **Raw JSON Output** conforming to REST API standards.
   - Download button (`st.download_button`) for exporting structured JSON metrics.

---

## 📂 Repository Structure

```
├── app.py                      # Main Streamlit application file
├── models/
│   └── best.pt                 # Fine-tuned YOLO26n model weights (run7)
├── requirements.txt            # Application dependencies
└── README.md                   # Repository documentation
```

---

## 🛠️ Installation & Execution

1. Install required Python packages:
   `pip install -r requirements.txt`
2. Run the Streamlit web dashboard:
   `streamlit run app.py`
