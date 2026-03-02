# === DATA LOADER DENGAN K-FOLD CROSS VALIDATION ===
# Tujuan: Menyiapkan dataset untuk training model MobileNetV2

import os
import numpy as np
import cv2
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical

# === KONFIGURASI ===
dataset_dir = r"C:\Users\Lenovo\Klasifikasi_ikan\datasets\augmented_output"
img_height, img_width = 224, 224
num_classes = 2
k_folds = 5  # jumlah fold untuk cross-validation

# === MUAT DATASET ===
data = []
labels = []

class_names = ['segar', 'tidak_segar']

print("🔍 Memuat dataset...")
for label, class_name in enumerate(class_names):
    class_folder = os.path.join(dataset_dir, class_name)
    files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for file in files:
        img_path = os.path.join(class_folder, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Gagal membaca {file}")
            continue

        img = cv2.resize(img, (img_width, img_height))
        img = img / 255.0  # normalisasi ke 0–1
        data.append(img)
        labels.append(label)

data = np.array(data, dtype=np.float32)
labels = np.array(labels)

print(f"✅ Dataset dimuat: {data.shape[0]} gambar ({len(class_names)} kelas).")

# === ONE-HOT ENCODING LABEL ===
labels_cat = to_categorical(labels, num_classes=num_classes)

# === MEMBUAT K-FOLD SPLIT ===
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_indices = []

for train_idx, val_idx in skf.split(data, labels):
    fold_indices.append((train_idx, val_idx))

# === SIMPAN INDEKS KE FILE ===
np.save("kfold_indices.npy", np.array(fold_indices, dtype=object), allow_pickle=True)
print(f"📁 Indeks K-Fold disimpan sebagai 'kfold_indices.npy' ({k_folds} fold).")

# === CEK HASIL ===
for i, (train_idx, val_idx) in enumerate(fold_indices):
    print(f"Fold {i+1}: Train={len(train_idx)} | Validasi={len(val_idx)}")

print("\n🎯 Data Loader dengan K-Fold siap digunakan untuk training model.")
