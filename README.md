# AI-Age-Gender-Detection

An AI-powered Age and Gender Detection System built using **Python, Computer Vision, OpenCV, Streamlit, and Deep Learning**.

The system can detect faces from different input sources and estimate the **age** and **gender** of detected individuals. It provides a simple interactive web interface for testing images, webcam input, and web-based image sources.

---

## 📌 Project Overview

**AI-Age-Gender-Detection** is a computer vision application designed to analyze human faces and predict their age and gender using deep learning-based models.

The project combines:

* Face detection using OpenCV
* Deep learning-based age and gender prediction
* Image processing
* Multiple image input methods
* Interactive Streamlit web interface
* Automated testing scripts

The application is designed to provide a practical demonstration of AI-based facial analysis.

---

## ✨ Features

* 👤 **Face Detection**

  * Detects human faces from input images.
  * Uses an OpenCV-based face detection model.

* 🎂 **Age Detection**

  * Estimates the approximate age of detected faces.
  * Uses a Vision Transformer (ViT)-based age prediction model.

* 🚻 **Gender Detection**

  * Predicts the gender of detected individuals.

* 📷 **Multiple Input Methods**

  * Upload an image
  * Use a live webcam
  * Process pre-tested images
  * Provide an image through a web link

* 🖥️ **Interactive Web Interface**

  * Built using Streamlit.
  * Provides an easy-to-use interface for running predictions.

* 🧪 **Testing**

  * Includes separate test scripts for:

    * Face detection
    * Model testing
    * Inference
    * Full pipeline testing

* 📊 **Performance Documentation**

  * Includes model performance comparison information and training logs.

---

## 🛠️ Technologies Used

| Technology               | Purpose                             |
| ------------------------ | ----------------------------------- |
| Python                   | Core programming language           |
| Streamlit                | Interactive web application         |
| OpenCV                   | Face detection and image processing |
| PyTorch                  | Deep learning model inference       |
| Vision Transformer (ViT) | Age/Gender prediction               |
| NumPy                    | Numerical processing                |
| Pillow                   | Image processing                    |
| Git & GitHub             | Version control and project hosting |

---

## 📁 Project Structure

```text
AI-Age-Gender-Detection/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── assets/
│   ├── face2.jpg
│   ├── face3.jpg
│   ├── group1.jpg
│   ├── group2.jpg
│   ├── group3.jpg
│   ├── photo.avif
│   └── url_test_*.jpg
│
├── docs/
│   ├── demo/
│   │   └── Working Video.mp4
│   │
│   └── screenshots/
│       ├── LiveWebCame_InputMethod.png
│       ├── PreTested_InputMethod.png
│       ├── UploadingImages_InputMethod.png
│       ├── UserInterface_part1.png
│       ├── UserInterface_part2.png
│       └── WebLink_InputMethod.png
│
├── models/
│   ├── age_gender_vit/
│   │   ├── config.json
│   │   ├── model.py
│   │   ├── preprocessor_config.json
│   │   ├── training_logs.json
│   │   ├── performance_comparison.png
│   │   └── README.md
│   │
│   └── face_detector/
│       ├── opencv_face_detector.pbtxt
│       └── opencv_face_detector_uint8.pb
│
├── utils/
│   └── detector.py
│
├── test_face_detection.py
├── test_full_pipeline.py
├── test_inference.py
└── test_model.py
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Zeba-Phool2228/CrixSoft_AI-Age-Gender-Detection.git
```

### 2. Navigate to the Project

```bash
cd CrixSoft_AI-Age-Gender-Detection
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```powershell
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

After starting the application, Streamlit will provide a local URL that can be opened in a web browser.

---

## 🧪 Running Tests

The project includes multiple test scripts.

### Face Detection Test

```bash
python test_face_detection.py
```

### Model Test

```bash
python test_model.py
```

### Inference Test

```bash
python test_inference.py
```

### Full Pipeline Test

```bash
python test_full_pipeline.py
```

---

## 📷 Input Methods

The application supports multiple ways of providing images:

### 1. Upload Image

Upload an image directly through the web interface.

### 2. Live Webcam

Capture an image using a connected webcam.

### 3. Pre-Tested Images

Use images already available in the project assets for testing.

### 4. Web Image Link

Provide a direct image URL for processing.

---

## 📊 Documentation

Project screenshots are available in:

```text
docs/screenshots/
```

The working demonstration video is available in:

```text
docs/demo/Working Video.mp4
```

---

## 🤖 AI Pipeline

The overall processing pipeline can be summarized as:

```text
Input Image
     ↓
Face Detection
     ↓
Face Extraction
     ↓
Image Preprocessing
     ↓
Age & Gender Prediction
     ↓
Prediction Results
     ↓
Streamlit Interface
```

---

## 📌 Model Information

The project uses two major components:

### Face Detection Model

An OpenCV-based face detector is used to locate faces within input images.

### Age & Gender Model

A Vision Transformer (ViT)-based deep learning model is used for age and gender prediction.

The large model weight file is intentionally excluded from the Git repository because of its large size. The required model configuration and supporting files are included in the repository.

---

## 🔐 Repository & Security

Sensitive information and unnecessary generated files are excluded through `.gitignore`.

The repository does not include:

* API keys
* Passwords
* Authentication tokens
* Python cache files
* Model cache files
* Large excluded model weights

---

## 👩‍💻 Project

**Project Name:** AI-Age-Gender-Detection

**Repository:** CrixSoft_AI-Age-Gender-Detection

**Domain:** Artificial Intelligence / Computer Vision / Deep Learning

**Interface:** Streamlit Web Application

---

## 📄 License

This project is intended for educational, research, and demonstration purposes.
