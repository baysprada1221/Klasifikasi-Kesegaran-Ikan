# === AUGMENTASI CITRA IKAN TIDAK SEGAR ===

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === KONFIGURASI FOLDER ===
input_folder = r"C:\Users\Lenovo\Klasifikasi_ikan\datasets\resized_output\tidak_segar"
output_folder = r"C:\Users\Lenovo\Klasifikasi_ikan\datasets\augmented_output\tidak_segar"

os.makedirs(output_folder, exist_ok=True)

# === PARAMETER AUGMENTASI ===
datagen = ImageDataGenerator(
    rotation_range=20,              # Memutar gambar secara acak hingga ±20 derajat.
    width_shift_range=0.1,          # Menggeser gambar secara horizontal hingga 10% dari lebar gambar.
    height_shift_range=0.1,         # Menggeser gambar secara vertikal hingga 10% dari tinggi gambar.
    zoom_range=0.1,                 # Memperbesar atau memperkecil gambar hingga 10%.
    brightness_range=(0.8, 1.2),    # Mengubah tingkat kecerahan antara 80%–120% dari gambar asli.
    horizontal_flip=True,           # Membalik gambar secara horizontal (kiri ↔ kanan).
    fill_mode='nearest'             # Mengisi area kosong akibat rotasi/pergeseran dengan nilai piksel terdekat.
)

target_count = 500  # total citra hasil augmentasi
files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
num_files = len(files)

print(f"🔹 Kelas: Tidak Segar | Ditemukan {num_files} gambar asli.")
per_image_aug = int(np.ceil(target_count / num_files))
total_generated = 0

for file in files:
    img_path = os.path.join(input_folder, file)
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Gagal membaca {file}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    aug_iter = datagen.flow(img, batch_size=1)

    for i in range(per_image_aug):
        if total_generated >= target_count:
            break
        aug_img = next(aug_iter)[0].astype(np.uint8)
        save_path = os.path.join(output_folder, f"aug_tidaksegar_{total_generated+1:03d}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
        total_generated += 1

    if total_generated >= target_count:
        break

print(f"✅ Total augmentasi kelas TIDAK SEGAR: {total_generated} citra disimpan di {output_folder}")
