---
title: PulmoScanAI
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# PulmoScanAI - AI Lung Cancer Detection System

An advanced web-based application for detecting lung cancer from histopathology images using a deep learning CNN model with feature-based analysis.

## Features

- **Real-time AI Analysis**: Uses TensorFlow/Keras deep learning model
- **Feature-based Detection**: Analyzes darkness, purple staining, and edge density
- **Beautiful UI**: Modern, responsive design with animated backgrounds
- **Drag & Drop Upload**: Easy image upload with preview
- **Confidence Score**: Displays detection confidence percentage
- **CORS Enabled**: Seamless frontend-backend communication

## How It Works

1. **Upload Image**: Drag & drop a histopathology image
2. **CNN Processing**: Model analyzes tissue patterns
3. **Feature Analysis**: Evaluates darkness, staining, and texture
4. **Result**: Shows diagnosis with confidence score
   - **Green**: Normal tissue detected
   - **Red**: Cancer detected

## API Endpoints

### Health Check
```
GET /api/health
```

### Prediction
```
POST /api/predict
Content-Type: multipart/form-data
```

**Request**: Image file in multipart form data
**Response**:
```json
{
  "is_cancer": false,
  "confidence": 0.92,
  "diagnosis": "No Cancer Found",
  "confidence_percentage": 92.0
}
```

## Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: 150×150 RGB images
- **Output**: 3-class classification (Adenocarcinoma, Normal, Squamous Cell Carcinoma)
- **Framework**: TensorFlow 2.13.0 / Keras

## Technical Stack

- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Backend**: Python Flask with Flask-CORS
- **ML Framework**: TensorFlow 2.x / Keras
- **Image Processing**: OpenCV, Pillow, NumPy

## Project Structure

```
├── app.py                    # Flask backend server
├── best_lung_model.h5       # Trained CNN model
├── PulmoScanAI.html         # Web frontend
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container configuration
└── README.md               # This file
```

## License

© 2025 PulmoScanAI • Next-Gen AI Pathology Platform
