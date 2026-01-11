import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from data_pipeline import load_dataset

test_ds = load_dataset("C:\\Users\\msthe\\Downloads\\archive (1)\\Test\\Test", shuffle=False)

model = tf.keras.models.load_model("best_model.h5")

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    preds = (preds > 0.5).astype(int)
    y_true.extend(labels.numpy())
    y_pred.extend(preds.flatten())

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Healthy", "Diseased"]))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))
