# ArogyaAI

ArogyaAI is an AI-powered healthcare application focused on **skin disease detection** using deep learning. The project uses a **MobileNetV2**-based convolutional neural network trained on skin images to predict disease classes and provide fast, automated assistance.

---

## 📌 Motivation

Skin diseases are common, but access to dermatologists can be limited. ArogyaAI aims to:

* Assist in **early-stage screening**
* Reduce dependency on immediate specialist access
* Demonstrate practical use of **deep learning in healthcare**

> ⚠️ This project is for **educational and research purposes only** and should not replace professional medical diagnosis.

---

## 🚀 Features

* Skin disease classification using **MobileNetV2**
* Pre-trained and fine-tuned deep learning model
* Python-based inference pipeline
* Jupyter Notebook for training and experimentation
* Ready-to-use `.h5` trained model
* Simple application interface

---

## 🧠 Model Architecture

* Base Model: **MobileNetV2** (Transfer Learning)
* Input Shape: `224 x 224 x 3`
* Loss Function: Categorical Crossentropy
* Optimizer: Adam
* Output: Multi-class skin disease prediction

MobileNetV2 is chosen because:

* Lightweight and fast
* Suitable for deployment on low-resource devices
* High accuracy with fewer parameters

---

## 📁 Project Structure

```
ArogyaAI/
│
├── app.py                  # Main application file
├── arogya-ai.ipynb         # Training and experimentation notebook
├── skin_mobilenetv2.h5     # Trained deep learning model
├── sample__s/              # Sample skin images
└── README.md               # Project documentation
```

---

## 🛠️ Technologies Used

* Python 3.8+
* TensorFlow / Keras
* NumPy
* OpenCV
* Flask (if app is web-based)
* Jupyter Notebook

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Vivekpadavale11/ArogyaAI.git
cd ArogyaAI
```

### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install tensorflow numpy opencv-python flask
```

---

## ▶️ How to Run

### Run the application

```bash
python app.py
```

### Example Usage

1. Start the application
2. Upload a skin image
3. The model predicts the disease class
4. Output is displayed on screen

### Screenshots

*Add screenshots here:*

* Home page
* Image upload page
* Prediction result page

```bash
python app.py
```

### Run the notebook

```bash
jupyter notebook arogya-ai.ipynb
```

---

## 🧪 Dataset

* Skin disease image dataset
* Images resized to `224x224`
* Data preprocessing includes:

  * Normalization
  * Augmentation (optional)

> Dataset source can be replaced or extended for better accuracy.

---

## 📊 Results

* High accuracy on validation data
* Fast inference time
* Suitable for real-time prediction

(Exact metrics depend on dataset and training configuration.)

---

## 🔮 Future Improvements

* Add more disease classes
* Improve dataset size and quality
* Add confidence score to predictions
* Deploy using Flask / FastAPI
* Convert model to TensorFlow Lite for mobile use

---

## 📚 Use Cases

* Academic projects
* AI/ML learning
* Healthcare research prototypes
* Early disease screening tools

---

## 👤 Author

**Vivek Padavale**
Student | AI & ML Enthusiast

GitHub: [https://github.com/Vivekpadavale11](https://github.com/Vivekpadavale11)

---

## 📜 License

This project is open-source and available for educational use.

---

## ⚠️ Disclaimer

ArogyaAI does **not** provide medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.
