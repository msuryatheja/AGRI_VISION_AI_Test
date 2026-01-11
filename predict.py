import argparse
import tensorflow as tf
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
args = parser.parse_args()

model = tf.keras.models.load_model("best_model.h5")

img = tf.io.read_file(args.image)
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.resize(img, (224, 224))
img = img / 255.0
img = tf.expand_dims(img, axis=0)

prediction = model.predict(img)[0][0]

label = "Diseased" if prediction > 0.5 else "Healthy"
confidence = prediction if prediction > 0.5 else 1 - prediction

print(f"Prediction: {label}")
print(f"Confidence: {confidence * 100:.2f}%")
