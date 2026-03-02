import cv2
import os

# === Konfigurasi Folder Yang digunakan ===
input_folder = r"C:\Users\Lenovo\Klasifikasi_ikan\datasets\roi_output_tdksegar"
output_folder = r"C:\Users\Lenovo\Klasifikasi_ikan\datasets\resized_output"

# Memastikan agar terdapat folder untuk menyimpan hasil resize
os.makedirs(output_folder, exist_ok=True)

# Tentukan ukuran baru untuk citra (standar MobileNetV2)
target_size = (224, 224)

# Ambil semua file gambar
files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"🔍 Ditemukan {len(files)} gambar untuk di-resize.\n")

for filename in files:
    img_path = os.path.join(input_folder, filename)
    image = cv2.imread(img_path)

    if image is None:
        print(f"⚠️ Gagal membaca {filename}, dilewati.")
        continue

    # Resize gambar
    resized_img = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Simpan hasil resize ke folder baru
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, resized_img)

    print(f"✅ {filename} → diresize menjadi {target_size}")

print("\n🎯 Semua gambar selesai diresize dan disimpan di folder:")
print(output_folder)
