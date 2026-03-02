import cv2
import os
import numpy as np

# === KONFIGURASI FOLDER Yang digunakan ===
base_input_folder = r"C:\Users\Lenovo\Klasifikasi_ikan\datasets\augmented_output"
base_output_folder = r"C:\Users\Lenovo\Klasifikasi_ikan\datasets\normalized_output"

categories = ["segar", "tidak_segar"]

# Buat folder output utama dan subfolder untuk kelas
for category in categories:
    input_path = os.path.join(base_input_folder, category)
    output_path = os.path.join(base_output_folder, category)
    os.makedirs(output_path, exist_ok=True)

    files = [f for f in os.listdir(input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\n📂 Memproses {len(files)} gambar pada kelas '{category}'...")

    for i, filename in enumerate(files, start=1):
        img_path = os.path.join(input_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Gagal membaca {filename}, dilewati.")
            continue

        # === NORMALISASI secara MANUAL ===
        img_normalized = img.astype(np.float32) / 255.0  # ubah ke float dan dilakukan normalisasi

        # Konversi kembali ke bentuk yang bisa disimpan
        img_to_save = np.clip(img_normalized * 255, 0, 255).astype(np.uint8)

        # Simpan ke folder baru
        save_path = os.path.join(output_path, filename)
        cv2.imwrite(save_path, img_to_save)

        if i % 50 == 0 or i == len(files):
            print(f"✅ {i}/{len(files)} gambar dari '{category}' telah dinormalisasi.")

print("\n🎯 Proses normalisasi selesai untuk semua kelas!")
