import tensorflow as tf
import os
import logging

logging.basicConfig(level=logging.INFO)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

CLASS_MAP = {
    "Healthy": 0,
    "Rust": 1,
    "Powdery": 1
}

def load_dataset(data_dir, shuffle=True, augment=False):
    images = []
    labels = []

    for class_name, label in CLASS_MAP.items():
        class_dir = os.path.join(data_dir, class_name)

        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Directory not found: {class_dir}")

        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(class_dir, img_name)
            images.append(img_path)
            labels.append(label)

    logging.info(f"Loaded {len(images)} images from {data_dir}")

    ds = tf.data.Dataset.from_tensor_slices((images, labels))

    def process(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0

        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, 0.9, 1.1)

        return img, label

    ds = ds.map(process, num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)

    return ds
