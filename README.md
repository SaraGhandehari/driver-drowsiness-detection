# Driver Drowsiness Detection System 🚗💤

A real-time computer vision system for detecting driver drowsiness and fatigue using facial landmark analysis, built with OpenCV, Dlib, and Flask.

## 📌 Overview

Driver drowsiness is a major cause of road accidents. This project provides a real-time monitoring system that detects:

Eye closure (drowsiness)
Yawning (fatigue indicator)

The system processes video streams from a webcam or uploaded video and triggers alerts when risky behavior is detected.

## 🔍 Features

* Real-time webcam-based monitoring
* Eye closure detection (EAR - Eye Aspect Ratio)
* Yawning detection using lip distance
* Alert system (audio warning)
* Video upload support
* Lightweight Flask web interface

## 🧠 How It Works

The system uses:

* **Dlib** for facial landmark detection (68-point model)
* **OpenCV** for video processing
* **EAR (Eye Aspect Ratio)** to detect eye closure
* **Lip distance analysis** to detect yawning
* Temporal thresholds to reduce false positives

## ▶️ Usage

```bash
python app.py
```

Then open:

```
http://localhost:5000
```

## 📁 Requirements

* Python 3.8+
* OpenCV
* Dlib
* NumPy
* SciPy
* Flask
* Pygame

## 📦 Model File

Download the dlib landmark model:

```
model/ shape_predictor_68_face_landmarks.dat
```

## 📊 Future Improvements

* Deep learning-based detection (CNN / LSTM)
* Mobile deployment
* Driver behavior analytics dashboard
* Cloud logging system
