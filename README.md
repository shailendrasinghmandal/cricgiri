# Cricket Ball Analytics System using YOLOv8 and Kalman Tracking

## Production API (Render — permanent URL)

The FastAPI analytics API is ready to deploy from this repo.

| Step | Action |
|------|--------|
| 1 | Open **[Deploy to Render](https://render.com/deploy?repo=https://github.com/shailendrasinghmandal/cricket_project)** |
| 2 | Sign in with **GitHub** (same account: `shailendrasinghmandal`) |
| 3 | Click **Deploy Blueprint** — uses root `render.yaml` |
| 4 | Confirm **Free** plan (no payment) — or **Starter** ($7/mo) for always-on client demos |
| 5 | Wait ~15–20 min for Docker build (PyTorch + YOLO) |

**Free plan caveats:** service sleeps after 15 min idle (~1 min wake-up); 512 MB RAM may be tight for YOLO; uploads are not persisted across restarts.

**Permanent URLs after deploy:**

- Health: `https://cricgiri-analytics-api.onrender.com/health`
- Swagger UI: `https://cricgiri-analytics-api.onrender.com/docs`
- Analyze: `POST https://cricgiri-analytics-api.onrender.com/analyze`

Full guide: [`deploy/DEPLOYMENT.md`](deploy/DEPLOYMENT.md) · Client doc: [`docs/API_SHARE_DOCUMENT.md`](docs/API_SHARE_DOCUMENT.md)

---

## Overview

This project is a Computer Vision based Cricket Ball Analytics System developed using Python, YOLOv8, OpenCV, and Kalman Filtering.

The system takes a short cricket bowling video as input and performs:

* Cricket ball detection
* Smooth ball tracking
* Trajectory visualization
* Speed estimation
* Swing classification
* Analytics visualization
* Final summary generation

The system is designed for internship-level sports analytics demonstrations and works on portrait-format 30 FPS cricket delivery videos.

---

# Project Objective

The objective of this project is to build a lightweight sports analytics pipeline capable of:

1. Detecting a cricket ball in video frames
2. Tracking the ball smoothly across frames
3. Reducing false detections
4. Estimating approximate delivery speed
5. Analyzing delivery trajectory
6. Classifying swing direction
7. Producing a professional analytics output video

---

# Technologies Used

| Technology    | Purpose                            |
| ------------- | ---------------------------------- |
| Python        | Main programming language          |
| YOLOv8        | Cricket ball object detection      |
| OpenCV        | Video processing and visualization |
| NumPy         | Numerical computations             |
| Kalman Filter | Smooth predictive tracking         |
| CUDA / GPU    | Faster inference and training      |
| Ultralytics   | YOLOv8 framework                   |

---

# System Architecture

```text
Input Video
     ↓
YOLOv8 Ball Detection
     ↓
False Positive Filtering
     ↓
Kalman Filter Tracking
     ↓
Trajectory Generation
     ↓
Speed Estimation
     ↓
Swing Classification
     ↓
Analytics HUD Rendering
     ↓
Final Summary Screen
     ↓
Output Video
```

---

# Features Implemented

## 1. Cricket Ball Detection

A custom-trained YOLOv8 model is used to detect the cricket ball from each video frame.

The model was trained on cricket-ball datasets and fine-tuned specifically for cricket delivery videos.

### Detection Features

* Ball-only detection
* Confidence threshold filtering
* Bounding box generation
* Real-time inference

---

## 2. False Positive Reduction

The system removes incorrect detections such as:

* Stumps
* Shoes
* Reflections
* Lighting artifacts
* Large incorrect objects

This is achieved using:

* Bounding box size filtering
* Distance-based tracking validation
* Confidence filtering

---

## 3. Kalman Filter Based Tracking

A Kalman Filter is used for:

* Smoothing noisy detections
* Predicting ball movement
* Handling temporary detection loss

### Tracking States

The tracker supports:

* TRACKING
* PREDICTING
* COMPLETED

This improves robustness during motion blur and missed detections.

---

## 4. Trajectory Visualization

The detected ball path is visualized using a fading trajectory.

### Visualization Features

* Smooth trajectory rendering
* Gradient coloring
* Velocity arrows
* Motion stabilization

This improves visual interpretation of the delivery path.

---

## 5. Speed Estimation

The delivery speed is estimated using frame displacement and video FPS.

The following formula is used:

```text
speed = distance / time
```

### Speed Estimation Pipeline

1. Calculate pixel displacement between frames
2. Convert pixels to approximate meters
3. Use FPS to compute velocity
4. Apply rolling-average smoothing
5. Display final stable delivery speed

### Notes

* Speed estimation is approximate
* Calibration depends on camera angle
* Designed for prototype-level analytics

---

## 6. Swing Classification

Instead of seam-based RPM estimation, the system performs trajectory-based swing analysis.

### Swing Types

* Left Swing
* Right Swing
* Straight Delivery

### Method Used

The system analyzes:

* Horizontal trajectory displacement
* Direction consistency
* Curvature trend over time

This method is more stable for 30 FPS videos.

---

## 7. Analytics HUD

A professional analytics overlay is displayed on the video.

### HUD Displays

* Delivery speed
* Swing classification
* Tracking status
* Speed bar

---

## 8. Final Summary Screen

At the end of processing, the system displays a final analytics summary screen containing:

* Final delivery speed
* Delivery type
* Completion status

This improves the professional presentation quality of the system.

---

# Dataset

The project uses cricket-ball datasets containing:

* Cricket delivery frames
* Ball annotations
* Bounding box labels

The dataset is used to train a custom YOLOv8 model.

---

# Model Training

The YOLOv8 model was trained using:

```bash
python train.py
```

### Training Configuration

* Custom cricket-ball dataset
* GPU acceleration enabled
* Multiple epochs
* YOLOv8 architecture

---

# Project Structure

```text
cricket_project/
│
├── dataset/
├── videos/
├── output/
├── runs/
├── track_ball.py
├── train.py
├── requirements.txt
└── README.md
```

---

# How to Run

## Step 1 — Activate Virtual Environment

```bash
venv\Scripts\activate
```

## Step 2 — Run Analytics System

```bash
python track_ball.py
```

---

# Output

The processed analytics video is automatically saved as:

```text
output/ball_analytics.mp4
```

---

# Performance Observations

The system performs well for:

* Clear cricket delivery videos
* Moderate lighting conditions
* Single-ball tracking
* Short-duration bowling clips

---

# Limitations

## 1. Approximate Speed Estimation

The current implementation uses pixel-to-meter calibration.

True physical speed requires:

* Camera calibration
* Perspective correction
* Multi-camera geometry

---

## 2. Swing Estimation is Trajectory-Based

The system does not estimate true seam RPM.

Reason:

* 30 FPS videos lack sufficient rotational detail
* Cricket ball appears motion blurred
* Reliable seam tracking is difficult in low-resolution footage

Trajectory-based swing analysis was chosen for stability and realism.

---

## 3. Single Object Tracking

The current implementation tracks only the cricket ball.

Other entities such as:

* Bowler
* Batsman
* Bat

are not analyzed.

---

# Future Improvements

Possible future enhancements include:

* Bounce point detection
* Delivery length classification
* Multi-object tracking
* Real camera calibration
* Streamlit web application
* Real-time live camera processing
* Bat-ball interaction analysis

---

# Conclusion

This project demonstrates the application of:

* Computer Vision
* Deep Learning
* Object Detection
* Motion Tracking
* Sports Analytics
* Video Processing

for cricket ball analytics.

The system successfully combines YOLOv8 detection, Kalman filtering, trajectory analysis, speed estimation, and swing classification into a complete sports analytics prototype.

---

# Author

Shailendra Singh Mandal

Academic deep learning / Sports Analytics Project
