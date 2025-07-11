# 🎭 FACIAL-EMOTION-RECOGNITION-YKG

## 🚀 AI-Powered Facial Emotion Recognition System  
An advanced Deep Learning solution for **real-time facial emotion detection**, built with **CNN & TensorFlow**, featuring an intuitive **Tkinter GUI** for seamless user interaction.

---

📌 **Developed by Yuvraj Kumar Gond** during his **AI/ML Internship at ShadowFox Company**, a multinational firm based in **Bengaluru & Sydney**.

> 🗂️ `t1.py` – Main Application Code  
> 🧠 `t11.py` – Model Training & Deep Learning Scripts  

---

## 📌 About the Project

The **Facial Emotion Recognition System** is designed to accurately detect and classify human emotions using state-of-the-art **deep learning**. It utilizes **Convolutional Neural Networks (CNN)** trained on the **FER 2013 dataset**, allowing recognition from both static images and real-time webcam feeds.

### 🔥 Unique Features

- 🧠 **Self-Learning Capability** – Continuously improves with new training data  
- 🖥️ **GUI Interface** – Clean, interactive, and built using Tkinter  
- 🎥 **Real-Time Detection** – Optimized for webcam-based emotion capture  
- ⚙️ **Automated Preprocessing** – Uses image augmentation & grayscale conversion  
- 🛡️ **Overfitting Prevention** – Includes Dropout, EarlyStopping, and ModelCheckpoint  
- 📊 **High Accuracy & Speed** – Built for scalable and fast real-time classification  

---

## 🎯 Potential Use Cases

✅ Emotion tracking in **AI assistants** & **chatbots**  
✅ **Mental health monitoring** through facial analytics  
✅ **Sentiment analysis** in customer experience tools  
✅ **Interactive gaming & VR** based on real-time emotions  
✅ AI-driven **learning platforms** or classroom mood detectors  

---

## 🔬 How the Model Works

### 1️⃣ Image Preprocessing  
- Resizing and grayscale conversion  
- Image augmentation (flip, zoom, rotation, brightness)  
- Normalization for deep learning input  

### 2️⃣ Model Training  
- Built with **Convolutional Neural Networks (CNN)**  
- Layers: Conv2D → MaxPooling → Flatten → Dense → Dropout  
- Optimizer: **Adam**  
- Loss Function: **Categorical Crossentropy**  
- Metrics: Accuracy, Precision, Recall  

### 3️⃣ Emotion Classification  
- Final layer uses **Softmax** for multi-class output  
- Emotions classified include:  
  😃 Happy 😢 Sad 😠 Angry 😐 Neutral 😲 Surprise 🤢 Disgust 😨 Fear  

### 4️⃣ Continuous Learning  
- Supports **incremental training** with new labeled data  
- Just drop images into folders and retrain  

---

## 🛠 Tech Stack & Tools Used

| Component              | Tool/Library                  |
|------------------------|-------------------------------|
| Programming Language   | Python 🐍                     |
| Deep Learning Framework| TensorFlow & Keras ⚡         |
| Dataset                | FER 2013 (Kaggle) 📊          |
| Image Preprocessing    | OpenCV, NumPy 📷              |
| Model Architecture     | CNN (Convolutional Neural Network) |
| GUI Interface          | Tkinter 🎨                   |
| Performance Metrics    | Classification Report, Confusion Matrix 📈 |
| Model Format           | `.keras` for model storage   |

---

## 🎨 GUI & ML Design Concept

### 🔹 Graphical User Interface (GUI)
- Built with **Tkinter** for a clean, interactive look  
- Allows:
  - 📷 Image Upload
  - 🎥 Webcam Capture
  - 🧠 Emotion Prediction Display
  - 📚 Option to Retrain with New Images

### 🔹 Machine Learning Architecture
- Uses deep **CNN** with:
  - Conv2D, MaxPooling2D, Dropout, Flatten, Dense  
  - Final Softmax for emotion classification  
- Trained with:
  - 📊 80% Training, 20% Validation  
  - 🔁 Batch Size: 32  
  - ⏱️ Epochs: 50 (with EarlyStopping)

---

## 📊 Model Performance & Evaluation

✅ **Test Accuracy:** 🚀 _Insert Final Test Accuracy Here_  
✅ **Validation Loss:** 📉 _Insert Final Loss Value Here_  

### Evaluation Metrics:
- 📋 **Classification Report** – Precision, Recall, F1-Score  
- 🔄 **Confusion Matrix** – Visual emotion prediction accuracy  

---

## 🚀 How to Use the Model

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-github-username/Facial-Emotion-Recognition.git
cd Facial-Emotion-Recognition
2️⃣ Install Dependencies
bash
Copy
Edit
pip install tensorflow numpy opencv-python scikit-learn matplotlib pillow
3️⃣ Run the Model for Emotion Detection
bash
Copy
Edit
python emotion_detection.py
4️⃣ Train the Model with New Data (Optional)
Add new images to the train/ directory

Then run:

bash
Copy
Edit
python train_model.py
5️⃣ Launch the GUI for Emotion Detection
bash
Copy
Edit
python gui_emotion_recognition.py
🤝 Acknowledgment
💡 Concept & Development: Yuvraj Kumar Gond
🤖 AI/ML Support: ChatGPT
📂 Dataset Used: FER 2013 from Kaggle
👨‍🏫 Internship Mentor: (Insert Mentor’s Name)
🏢 Internship Company: ShadowFox Company (Bengaluru & Sydney)

📞 Connect With Me!
💼 LinkedIn: Yuvraj Kumar Gond

📧 Email: yuviig456@gmail.com

⭐ If you find this project helpful, don’t forget to star the repo on GitHub!
