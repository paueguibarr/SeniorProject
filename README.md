# CSCI 487 – Running Gait Analysis with Computer Vision

This project analyzes running form from 2D video using human pose estimation and gait metric extraction.
It produces interpretable metrics (cadence, stride timing, asymmetry, variability) and a visualization dashboard.

## MVP
- Input: 2D video clip of a runner 
- Output:
  - joint trajectories (hips/knees/ankles/etc.)
  - gait metrics and risk indicators (rule-based)
  - skeleton overlay and plots in a dashboard

## Repo Structure
- `src/pose/` – pose extraction (MediaPipe/OpenPose/Sapiens experiments)
- `src/metrics/` – gait detection and metric computation
- `src/visualization/` – dashboard and visualizations 
- `data/` – videos/frames 
- `reports/` – figures and results

## Overview
1. Extract frames from a video
2. Run pose estimation to generate keypoints
3. Compute gait metrics
4. Launch dashboard
