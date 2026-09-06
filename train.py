import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

# Config
DATASET_DIR = 'dataset'
CATEGORIES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
IMG_SIZE = (64, 64)
NUM_CLASSES = len(CATEGORIES)
EPOCHS = 20
BATCH_SIZE = 32

os.makedirs('model', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# 1. Load & Preprocess Images
print("Loading images...")
X, y = [], []

for label_idx, category in enumerate(CATEGORIES):
    folder_path = os.path.join(DATASET_DIR, category)
    image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))
    
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(IMG_SIZE)
            img_array = np.array(img) / 255.0  # Normalize
            
            # Reshape into sequence (64 timesteps, 192 features)
            sequence = img_array.reshape(IMG_SIZE[0], IMG_SIZE[1] * 3)
            
            X.append(sequence)
            y.append(label_idx)
        except Exception:
            pass

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
y_cat = to_categorical(y, num_classes=NUM_CLASSES)

# 2. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)

# 3. Build Bidirectional LSTM Model
model = Sequential([
    Input(shape=(64, 192)),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 4. Model Training
print("\nTraining upgraded Bidirectional LSTM model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# 5. Save Model
model_path = os.path.join('model', 'flower_rnn.keras')
model.save(model_path)
print(f"\nModel saved successfully at {model_path}")

# 6. Evaluation Plots
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nUpdated Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=CATEGORIES))

# Plot Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CATEGORIES)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Bidirectional LSTM Flower Classification Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join('outputs', 'confusion_matrix.png'))
plt.close()