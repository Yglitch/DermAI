# 🩹 DermAI – Wound Detection using Machine Learning

DermAI is a **machine learning-powered application** that detects different types of wounds from skin images, including:

- **Cuts**
- **Burns**
- **Abrasions**
- **Normal Skin**

The project is built with **Python, OpenCV, Scikit-learn, and Streamlit**, and includes both **image upload** and **live webcam detection** features.

---

## 📸 Features

- **Upload an image** for wound detection
- **Real-time webcam detection**
- Supports **four classes**: `cut`, `burn`, `abrasions`, `normal skin`
- **RandomForestClassifier** for classification
- **Class-balanced training** to reduce bias
- Saves predictions and displays live results on webcam feed

---

## 🏗 Project Structure

DermAI/
├─ Dermai.py # Streamlit app for wound detection
├─ train_model.py # Script to train a balanced RandomForest model
├─ model.pkl # Trained RandomForest model
├─ dataset/ # Dataset folder (not uploaded to GitHub)
│ ├─ cut/
│ ├─ burn/
│ ├─ abrasions/
│ └─ normal skin/
└─ README.md

yaml
Copy
Edit

---

## ⚡ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/YOUR_USERNAME/DermAI.git
cd DermAI
Create a virtual environment (optional but recommended):


python -m venv menv
menv\Scripts\activate   # Windows
source menv/bin/activate # Linux/Mac

Install dependencies
pip install -r requirements.txt

Run the Streamlit App:
#streamlit run Dermai.py

🏋️ Training a New Model

Prepare your dataset in the following structure:

dataset/
 ├─ cut/
 ├─ burn/
 ├─ abrasions/
 └─ normal skin/
Run the balanced training script:

python train_model.py
After training, a new model.pkl will be generated for the Streamlit app.

📊 Model Performance
Uses RandomForestClassifier with class_weight='balanced' to handle imbalanced datasets.

Prints classification report & confusion matrix for per-class accuracy.

Can achieve 70–90% accuracy depending on dataset size and balance.



📌 Future Improvements
Use deep learning (CNN) for better accuracy
Implement data augmentation for small datasets
Add automatic wound severity detection

Store prediction history in CSV for analysis

👨‍💻 Author
Yash Rana
GitHub: @Yglitch

