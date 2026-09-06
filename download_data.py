import os
import shutil
import tensorflow as tf

# Get the extracted directory from Keras cache
cache_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets')
extracted_dir = os.path.join(cache_dir, 'flower_photos')

TARGET_DIR = 'dataset'
os.makedirs(TARGET_DIR, exist_ok=True)

# Map original archive folder names to target folder names
folder_mapping = {
    'daisy': 'daisy',
    'dandelion': 'dandelion',
    'roses': 'rose',
    'rose': 'rose',
    'sunflowers': 'sunflower',
    'sunflower': 'sunflower',
    'tulips': 'tulip',
    'tulip': 'tulip'
}

if os.path.exists(extracted_dir):
    for folder in os.listdir(extracted_dir):
        src = os.path.join(extracted_dir, folder)
        if os.path.isdir(src) and folder in folder_mapping:
            dst = os.path.join(TARGET_DIR, folder_mapping[folder])
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"Mapped and copied: {folder} -> {folder_mapping[folder]}")

print("\nDataset ready with classes:")
print(os.listdir(TARGET_DIR))