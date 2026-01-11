├── data_pipeline.py       # Dataset loading & preprocessing
├── model_train.py         # Model training & fine-tuning
├── evaluate_model.py      # Model evaluation & metrics
├── prediction.py          # Single image inference
├── best_model.h5          # Best saved model
├── final_model.h5         # Final trained model


1. Dataset Handling
Images are organized into class-wise folders:
I)   Healthy
II)  Rust
III) Powdery
Rust and Powdery are merged into one Diseased (label = 1) class.
Dataset is loaded using a custom TensorFlow data pipeline (tf.data API).

Preprocessing steps:
I)   Resize images to 224 × 224
II)  Normalize pixel values to [0, 1]
III) Optional shuffling and augmentation
IV) Efficient batching, caching, and prefetching for performance

2. Data Augmentation:
To improve generalization, augmentation is applied during training:
I)   Horizontal flipping
II)  Random brightness adjustment
III) Random contrast variation
IV)  Rotation, zoom, and contrast layers inside the model

3. Model Architecture:
I)   Base Model: EfficientNetB0 (pre-trained on ImageNet)
II)  Strategy: Transfer Learning + Fine-Tuning
III) Architecture:
     a) Data augmentation layer
     b) Frozen EfficientNetB0 feature extractor
     c) Global Average Pooling
     d) Batch Normalization
     e) Dense (256 units, ReLU)
     f) Dropout (0.5)
     g) Output layer (Sigmoid for binary classification)
4. Training Strategy:
I)   Loss Function: Binary Crossentropy
II)  Optimizer: Adam
III) Evaluation Metrics:
     I)   Accuracy
     II)  Precision
     III) Recall
   <img width="615" height="173" alt="report" src="https://github.com/user-attachments/assets/e60aa963-3d40-4975-b176-b034a87dd792" />

Class Imbalance Handling: Class weights computed using training labels
Callbacks Used:
I)   EarlyStopping
II)  ReduceLROnPlateau
III) ModelCheckpoint (saves best model as best_model.h5)
Training Phases:
I)  Feature extraction with frozen base model
II) Fine-tuning last 50 layers of EfficientNetB0
5. Model Evaluation:
I)  Evaluation is performed on a separate test dataset
II) Metrics reported:
     a)Precision
     b)Recall
     c)F1-score
     d) Confusion Matrix
III) Predictions are thresholded at 0.5
6. Inference:
A command-line script allows prediction on a single image:
prediction.py --image path/to/image.jpg"
"python <img width="1365" height="718" alt="Screenshot 2026-01-11 193324" src="https://github.com/user-attachments/assets/db8f1246-7561-4a85-826e-eb0f5b81d1c7" />
