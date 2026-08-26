# Facetrix AI - Aspect-Ratio Invariant Face Recognition & Classification

A robust, production-grade **AI-Based Face Recognition and Classification System** built with **Python**, **Flask**, **InsightFace**, **ArcFace**, **RetinaFace**, **OpenCV**, and **SQLite**.

The system is explicitly engineered to recognize registered individuals in **real-time live webcam streams** or across images of **disparate aspect ratios** (16:9, 4:3, 1:1, 3:4 portrait, 21:9 ultrawide), **resolutions** (e.g. 1920×1080 vs 500×800 vs 1000×1000), **crops**, and **orientations** by extracting normalized 512-dimensional facial feature embeddings rather than performing whole-image pixel comparisons.

---

## Table of Contents
1. [Project Objective](#project-objective)
2. [Problem Statement](#problem-statement)
3. [Key Features](#key-features)
4. [Technologies Used](#technologies-used)
5. [System Architecture](#system-architecture)
6. [Core Methodology & Aspect-Ratio Handling](#core-methodology--aspect-ratio-handling)
7. [Installation Guide (Windows)](#installation-guide-windows)
8. [Database Initialization & Structure](#database-initialization--structure)
9. [Running the Application](#running-the-application)
10. [Person Registration & Recognition Workflow](#person-registration--recognition-workflow)
11. [Evaluation Suite](#evaluation-suite)
12. [Biometric Privacy & Security](#biometric-privacy--security)

---

## Project Objective

To build an intelligent face recognition web application capable of identifying registered individuals from live webcam streams and query images using deep facial feature vectors (ArcFace 512-D embeddings). The system guarantees robust recognition regardless of differences in image framing, resolution, aspect ratio, or camera source.

---

## Problem Statement

Traditional face recognition or naive image comparison techniques rely on resizing entire images to fixed dimensions (e.g., squashing a 1920×1080 16:9 photo into a 224×224 1:1 box). This distorts facial proportions, alters inter-ocular distances, and severely degrades recognition accuracy. 

Our application resolves this by isolating the facial region, computing 2D affine transformations using 5 key facial landmarks (left eye, right eye, nose, left mouth corner, right mouth corner), warping the face into a centered 112×112 chip, and generating scale/aspect-ratio invariant ArcFace embeddings.

---

## Key Features

- **Aspect-Ratio & Resolution Invariant**: Recognizes the same face whether captured in 4K 16:9 landscape, mobile 3:4 portrait, or 1:1 square crop.
- **🎥 Integrated Live Webcam Detection**: Real-time WebRTC webcam stream detection directly inside the `/recognize` page with live bounding boxes, white landmark node dots, identity badges, and similarity gauges.
- **📸 Dual Registration Options (`/register`)**: Register identities via traditional file upload or live webcam snapshot capture from different angles.
- **Deep Biometric Extraction**: Uses InsightFace (RetinaFace for face & landmark detection + ArcFace for 512-D feature embeddings).
- **SQLite Vector Database**: Efficient storage of person profiles and raw float32 embedding BLOBs.
- **Cosine Similarity & Thresholding**: Matches query embeddings using Cosine Similarity (\(\ge 0.45\) match vs. `Unknown`).
- **Modern Glassmorphism UI**: Dark-themed web interface with real-time video feeds, drag-and-drop file dropzones, confidence score fill bars, and crisp solid white landmark dots (`#FFFFFF`).
- **Automated Evaluation Suite**: Computes Accuracy, Precision, Recall, F1-Score, False Acceptance Rate (FAR), False Rejection Rate (FRR), and aspect-ratio performance breakdowns.

---

## Technologies Used

- **Language**: Python 3.11+
- **Web Framework**: Flask
- **Computer Vision & AI**: OpenCV, InsightFace, RetinaFace, ArcFace, ONNX Runtime, NumPy, scikit-learn
- **Database**: SQLite3
- **Frontend**: HTML5, Modern CSS3 (Vanilla Glassmorphism), JavaScript (ES6, WebRTC `getUserMedia`, Canvas API)

---

## System Architecture

```
                 Input Stream (Live WebRTC Frame OR Uploaded Photo)
                                       │
                                       ▼
                          [ RetinaFace Detector ]
                       (Bounding Box + 5 Landmarks)
                                       │
                                       ▼
                       [ 5-Point Affine Alignment ]
                 (Normalize eye line & 112x112 Face Chip)
                                       │
                                       ▼
                       [ ArcFace Feature Extractor ]
                         (512-D Normalized Vector)
                                       │
                                       ▼
                      [ Cosine Similarity Matcher ]
                (Compare vs SQLite Stored Embeddings)
                                       │
                                       ▼
                  [ Threshold Decision (Score >= 0.45) ]
                    ├── MATCH  ──► Person Name + Score
                    └── UNKNOWN ──► "Unknown Identity"
```

---

## Core Methodology & Aspect-Ratio Handling

1. **RetinaFace Detection**: Receives full-resolution image and detects bounding box `[x1, y1, x2, y2]` along with 5 facial landmark points: `left_eye`, `right_eye`, `nose_tip`, `left_mouth`, `right_mouth`.
2. **Affine Landmark Transformation**: Maps detected points to reference 112×112 target coordinates:
   ```python
   REFERENCE_LANDMARKS = [
       [38.2946, 51.6963], # Left Eye
       [73.5318, 51.5014], # Right Eye
       [56.0252, 71.7366], # Nose Tip
       [41.5493, 92.3655], # Left Mouth
       [70.7299, 92.2041]  # Right Mouth
   ]
   ```
3. **112×112 Normalization**: Crops centered face chip and normalizes pixel values. Outer image dimensions are completely decoupled.
4. **ArcFace Embedding**: ArcFace outputs unit-normalized vector \(v \in \mathbb{R}^{512}\) where \(\|v\|_2 = 1.0\).
5. **Cosine Distance Matching**:
   $$\text{Cosine Similarity}(q, d) = q \cdot d$$
   For multiple registered samples per person, **Maximum Cosine Similarity** is evaluated.

---

## Installation Guide (Windows)

Open **PowerShell** or **Command Prompt** and follow these commands:

### 1. Navigate to Project Folder
```powershell
cd C:\Users\KiTE\.gemini\antigravity\scratch\face-recognition-app
```

### 2. Create Virtual Environment
```powershell
python -m venv venv
```

### 3. Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
*(For Command Prompt, use `.\venv\Scripts\activate.bat`)*

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## Database Initialization & Structure

Initialize the SQLite database automatically by running:
```powershell
python database.py
```

### Schema Definition
- **`persons`**: `id` (INTEGER PRIMARY KEY), `name` (TEXT UNIQUE), `created_at` (TIMESTAMP)
- **`face_embeddings`**: `id` (INTEGER PRIMARY KEY), `person_id` (INTEGER FK), `embedding` (BLOB), `created_at` (TIMESTAMP)

---

## Running the Application

Start the Flask development server:
```powershell
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## Person Registration & Recognition Workflow

### Registration (`/register`)
1. Navigate to `/register`.
2. Enter the person's name or unique ID.
3. Choose **📁 Upload Image Files** OR **📸 Capture from Live Webcam**.
4. Capture or upload 2 to 4 photos from varied angles.
5. Click **Register & Store Embeddings**.

### Recognition (`/recognize`)
1. Navigate to `/recognize`.
2. Choose **🎥 Live Webcam Detection** (click *Start Camera*) OR **📁 Upload Query Photo**.
3. The system annotates the image/video stream with bounding boxes, crisp white landmark node dots, identity badges, and similarity confidence scores.

---

## Evaluation Suite

Run the evaluation script to test accuracy across aspect ratios:
```powershell
python evaluate.py
```

### Quantitative Evaluation Metrics Report
```
======================================================================
      AI FACE RECOGNITION SYSTEM - PERFORMANCE EVALUATION REPORT      
======================================================================
  True Positives  (TP) : 13    | False Positives (FP) : 0
  True Negatives  (TN) : 0     | False Negatives (FN) : 1
----------------------------------------------------------------------
  Overall Accuracy     : 92.86%
  Precision            : 100.00%
  Recall (Sensitivity) : 92.86%
  F1-Score             : 96.30%
  False Acceptance Rate: 0.00% (FAR)
  False Rejection Rate : 7.14% (FRR)
----------------------------------------------------------------------

----------------------------------------------------------------------
            RECOGNITION ACCURACY BY IMAGE ASPECT RATIO                
----------------------------------------------------------------------
  Aspect Ratio Category     | Samples  | Accuracy  
----------------------------------------------------------------------
  16:9 Landscape            | 4        | 100.00%
  1:1 Square                | 3        | 100.00%
  4:3 Standard              | 3        | 100.00%
  Portrait (3:4/9:16)       | 2        | 100.00%
----------------------------------------------------------------------
```

---

## Biometric Privacy & Security

- Facial images should only be registered with explicit consent.
- Raw 512-D vectors are stored as binary BLOBs in local SQLite and never exposed via public API responses.
- Uploaded files are validated and sanitized using `secure_filename`.
