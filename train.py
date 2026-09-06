import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20

def get_dataset_path():
    paths = [
        pathlib.Path('dataset') / 'flower_photos',
        pathlib.Path.home() / '.keras' / 'datasets' / 'flower_photos',
    ]
    for p in paths:
        if p.exists():
            if (p / 'daisy').exists():
                return p
            elif (p / 'flower_photos' / 'daisy').exists():
                return p / 'flower_photos'

    print("Downloading dataset...")
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    p = pathlib.Path(data_dir)
    return p / 'flower_photos' if (p / 'flower_photos').exists() else p

if __name__ == '__main__':
    data_dir = get_dataset_path()
    print(f"Loading data from: {data_dir.resolve()}")

    # Load splits with explicit deterministic seed & alphabetical class indexing
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"Verified Class Order: {class_names}")

    # Normalize pixels [0, 1] & optimize pipeline
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(tf.data.AUTOTUNE)

    # Build reliable Convolutional Neural Network
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(class_names), activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop]
    )

    # Save Model Artifacts
    os.makedirs('model', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    model.save('model/flower_rnn.keras')
    print("Model successfully saved to 'model/flower_rnn.keras'.")

    # Evaluate Confusion Matrix
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues')
    plt.title('Validation Confusion Matrix')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved to 'outputs/confusion_matrix.png'.")