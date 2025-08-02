import cv2
import numpy as np
import joblib
import csv
import os
from datetime import datetime

classes = ['Cut', 'Burns', 'Abrasions' , 'normal' ]

# Load trained model
model = joblib.load("model.pkl")

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def predict_image(image_path):
    features = extract_features(image_path)
    prediction = model.predict([features])[0]
    return classes[prediction]

# Example usage
image_path = "sample.jpg"  # Replace with your image
predicted_class = predict_image(image_path)
print(f"Predicted Wound Type: {predicted_class}")

# Save to CSV
csv_file = "predictions.csv"
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["Timestamp", "Image", "Prediction"])
    writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_path, predicted_class])

print("Prediction saved to predictions.csv")
