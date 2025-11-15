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
├── app.py                      # Flask backend server
├── best_lung_model.h5         # Trained CNN model
├── PulmoScanAI.html           # Web frontend
├── lung-cancer-new.ipynb      # Jupyter notebook for model training
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
└── README.md                  # This file
```

## File Descriptions

### `app.py`
The main Flask backend server that:
- Loads the trained CNN model (`best_lung_model.h5`)
- Provides REST API endpoints for health checks and predictions
- Handles image preprocessing (resizing to 150×150, normalization)
- Performs feature-based analysis (darkness, purple staining, edge density)
- Returns cancer detection results with confidence scores
- Enables CORS for frontend-backend communication

### `best_lung_model.h5`
Pre-trained Keras CNN model file containing:
- Model architecture (layers, weights, biases)
- Trained on histopathology images
- 3-class classification: Adenocarcinoma, Normal, Squamous Cell Carcinoma
- Input: 150×150×3 RGB images
- Output: Probability distribution over 3 classes

### `PulmoScanAI.html`
Modern, responsive web interface featuring:
- Drag-and-drop image upload functionality
- Real-time image preview
- Animated gradient background
- Results display with confidence scores
- Color-coded diagnosis (Green = Normal, Red = Cancer)
- Mobile-responsive design

### `lung-cancer-new.ipynb`
Jupyter notebook containing:
- Data exploration and preprocessing steps
- Model architecture design and training code
- Dataset preparation and augmentation
- Model evaluation metrics and visualizations
- Training history and performance analysis
- Used to create the `best_lung_model.h5` file

### `requirements.txt`
Python dependencies including:
- `flask` - Web framework for the backend server
- `flask-cors` - Cross-Origin Resource Sharing support
- `tensorflow` - Deep learning framework for model inference
- `opencv-python` - Image processing and feature extraction
- `pillow` - Image loading and manipulation
- `numpy` - Numerical operations and array handling

### `Dockerfile`
Container configuration for deploying the application:
- Base image setup with Python runtime
- Dependency installation
- Application file copying
- Port exposure (7860)
- Container startup command

## Installation & Setup

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/AyushmanGupta21/PulmoScanAI.git
   cd PulmoScanAI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask server**
   ```bash
   python app.py
   ```

4. **Open the web interface**
   - Open `PulmoScanAI.html` in your browser
   - Or navigate to `http://localhost:7860`

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t pulmoscanai .
   ```

2. **Run the container**
   ```bash
   docker run -p 7860:7860 pulmoscanai
   ```

3. **Access the application**
   - Navigate to `http://localhost:7860` in your browser

## Usage

1. Open the PulmoScanAI web interface
2. Drag and drop a histopathology image or click to upload
3. Wait for the AI analysis to complete
4. View the diagnosis result with confidence score
5. Green result indicates normal tissue, red indicates cancer detected

## Model Training

To retrain or modify the model:
1. Open `lung-cancer-new.ipynb` in Jupyter Notebook
2. Follow the notebook cells to prepare your dataset
3. Run the training cells to create a new model
4. Save the trained model as `best_lung_model.h5`
5. Replace the existing model file with your new model

## License

© 2025 PulmoScanAI • Next-Gen AI Pathology Platform
