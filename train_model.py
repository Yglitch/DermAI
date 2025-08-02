import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


# Dataset Path & Classes

DATASET_DIR = "dataset"  # Make sure folder structure is correct
CLASSES = ['Cut', 'Burns', 'Abrasions', 'normal']
IMG_SIZE = 64  # resize to 64x64

X, y = [], []

print("📥 Loading dataset...")


# Load Dataset

for label, class_name in enumerate(CLASSES):
    folder = os.path.join(DATASET_DIR, class_name)
    if not os.path.exists(folder):
        print(f"⚠️ Warning: Folder {folder} not found. Skipping...")
        continue

    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"📊 {class_name}: {len(image_files)} images")

    for filename in image_files:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        features = img.flatten()  # 64*64*3 = 12288
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"✅ Dataset loaded: {X.shape[0]} images, feature size {X.shape[1]}")

if len(X) == 0:
    raise ValueError("❌ No images found in the dataset folders. Please check dataset path and files.")


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Train RandomForest (with class balance)

print("🚀 Training RandomForest model with balanced class weights...")
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)


# Evaluate Model

accuracy = model.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {accuracy*100:.2f}%")

y_pred = model.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASSES))

print("📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Save Model

joblib.dump(model, "model.pkl")
print("💾 Model saved as model.pkl")
