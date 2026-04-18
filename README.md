# 🏃‍♂️ Stride Vision – Running Gait Analysis with Computer Vision

This project analyzes running form from 2D video using computer vision and pose estimation.
It extracts stride-level biomechanical features and provides interpretable feedback on running form through an interactive dashboard.

---

## 🚀 Minimum Viable Product (MVP)

**Input**

* 2D video clip of a runner (side/front/rear view)

**Output**

* Pose overlay video (skeleton tracking)
* Stride-level biomechanical features
* Running form predictions (e.g., overstride, trunk lean, etc.)
* Interactive visualizations and metrics in a dashboard

---

## 🧠 Core Pipeline

1. Extract pose landmarks from video using MediaPipe
2. Normalize joint coordinates
3. Detect strides from heel motion
4. Compute stride-level biomechanical features
5. Aggregate predictions across the run
6. Visualize results in a Streamlit dashboard

---

## 📁 Project Structure

```bash
Stride_Vision_Dashboard/
├── app.py                    # Streamlit dashboard
├── requirements.txt         # Python dependencies
├── packages.txt             # System dependencies (e.g., ffmpeg)
├── models/                  # Trained ML models + feature configs
├── pipeline/                # Core processing pipeline
│   ├── DataProcessingPipeline.py
│   └── StrideRangeAnalysis.py
├── static/                  # Assets (logo, images)
└── cache_videos/            # Temporary processed videos (ignored in git)
```

---

## 📊 Features Extracted

Examples of stride-level metrics:

* Cadence & stride time
* Foot-to-hip distance at contact (overstride)
* Knee angle at contact
* Trunk angle (mean & max)
* Vertical oscillation (hip movement)
* Step width / symmetry (multi-view)

---

## 🎯 Goal

To provide an **accessible, low-cost alternative to lab-based gait analysis** by using only standard video input and delivering **interpretable, actionable feedback** for runners.

---

## ⚙️ Tech Stack

* Python (pandas, numpy, scikit-learn)
* MediaPipe (pose estimation)
* OpenCV / ffmpeg (video processing)
* Streamlit (dashboard)
* Plotly (visualizations)

---

## 🧪 Status

* ✅ Pose extraction + pipeline working
* ✅ Feature engineering implemented
* ✅ Dashboard MVP complete
* 🚧 Model refinement and multi-view integration ongoing

---

## 📌 Notes

* `cache_videos/` is used for temporary outputs and is not persisted
* Models are stored locally in `models/`
* Deployment requires `ffmpeg` (see `packages.txt`)

---

## 👤 Author

Paulina Eguibar
Senior Project – Computer Science & Data Science
University of Mississippi
