Data Structure:
├── data_pipeline.py       # Dataset loading & preprocessing</br>
├── model_train.py         # Model training & fine-tuning</br>
├── evaluate_model.py      # Model evaluation & metrics</br>
├── prediction.py          # Single image inference</br>
├── best_model.h5          # Best saved model</br>
├── final_model.h5         # Final trained model</br>


1. Dataset Handling</br>
Images are organized into class-wise folders:</br>
I)   Healthy</br>
II)  Rust</br>
III) Powdery</br>
Rust and Powdery are merged into one Diseased (label = 1) class.</br>
Dataset is loaded using a custom TensorFlow data pipeline (tf.data API).</br>

Preprocessing steps:</br>
I)   Resize images to 224 × 224</br>
II)  Normalize pixel values to [0, 1]</br>
III) Optional shuffling and augmentation</br>
IV) Efficient batching, caching, and prefetching for performance</br>

2. Data Augmentation:</br>
To improve generalization, augmentation is applied during training:</br>
I)   Horizontal flipping</br>
II)  Random brightness adjustment</br>
III) Random contrast variation</br>
IV)  Rotation, zoom, and contrast layers inside the model</br>

3. Model Architecture:</br>
I)   Base Model: EfficientNetB0 (pre-trained on ImageNet)</br>
II)  Strategy: Transfer Learning + Fine-Tuning</br>
III) Architecture:</br>
     &nbsp;a) Data augmentation layer</br>
     b) Frozen EfficientNetB0 feature extractor</br>
     c) Global Average Pooling</br>
     d) Batch Normalization</br>
     e) Dense (256 units, ReLU)</br>
     f) Dropout (0.5)</br>
     g) Output layer (Sigmoid for binary classification)</br>
4. Training Strategy:</br>
I)   Loss Function: Binary Crossentropy</br>
II)  Optimizer: Adam</br>
III) Evaluation Metrics:</br>
     I)   Accuracy</br>
     II)  Precision</br>
     III) Recall</br>
   <img width="615" height="173" alt="report" src="https://github.com/user-attachments/assets/e60aa963-3d40-4975-b176-b034a87dd792" />

Class Imbalance Handling: Class weights computed using training labels</br>
Callbacks Used:</br>
I)   EarlyStopping</br>
II)  ReduceLROnPlateau</br>
III) ModelCheckpoint (saves best model as best_model.h5)</br>
Training Phases:</br>
I)  Feature extraction with frozen base model</br>
II) Fine-tuning last 50 layers of EfficientNetB0</br>
5. Model Evaluation:</br>
I)  Evaluation is performed on a separate test dataset</br>
II) Metrics reported:</br>
     a)Precision</br>
     b)Recall</br>
     c)F1-score</br>
     d) Confusion Matrix</br>
III) Predictions are thresholded at 0.5</br>
6. Inference:</br>
A command-line script allows prediction on a single image:</br>
"python prediction.py --image path/to/image.jpg"</br>
 <img width="1365" height="718" alt="Screenshot 2026-01-11 193324" src="https://github.com/user-attachments/assets/db8f1246-7561-4a85-826e-eb0f5b81d1c7" />
