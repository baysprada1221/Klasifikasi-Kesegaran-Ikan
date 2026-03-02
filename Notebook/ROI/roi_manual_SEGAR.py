import cv2
import os

# === Konfigurasi Folder Yang digunakan ===
input_folder = r"C:\Users\Lenovo\Klasifikasi_ikan\datasets\segar"
output_folder = r"C:\Users\Lenovo\Klasifikasi_ikan\datasets\roi_output_segar"

os.makedirs(output_folder, exist_ok=True)

# Ambil maksimal 50 gambar
files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:50]
print(f"🔍 Ditemukan {len(files)} gambar untuk diproses.\n")

for filename in files:
    image_path = os.path.join(input_folder, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"⚠️ Gagal membaca {filename}, dilewati.")
        continue

    while True:
        # Pilih ROI manual
        cv2.namedWindow("Pilih area kepala ikan", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pilih area kepala ikan", 800, 600)
        roi = cv2.selectROI("Pilih area kepala ikan", image)
        cv2.destroyAllWindows()

        if roi == (0, 0, 0, 0):
            print(f"⏭️ Dilewati: {filename}\n")
            break

        x, y, w, h = roi
        cropped = image[int(y):int(y+h), int(x):int(x+w)]

        # Tampilkan hasil crop
        cv2.imshow("Hasil Crop", cropped)
        print(f"\n🔍 Apakah hasil crop {filename} sudah bagus?")
        print("Tekan [Y] untuk simpan, [N] untuk ulang, [ESC] untuk keluar.")

        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key in [27, ord('q')]:  # ESC atau Q → keluar program
            print("\n🚪 Proses dibatalkan oleh pengguna.")
            exit()

        elif key in [ord('y'), ord('Y')]:
            save_path = os.path.join(output_folder, f"roi_{filename}")
            cv2.imwrite(save_path, cropped)
            print(f"✅ Disimpan: {save_path}\n")
            break  # lanjut ke gambar berikutnya

        elif key in [ord('n'), ord('N')]:
            print("🔁 Ulangi pemilihan ROI...\n")
            continue  # ulang untuk gambar yang sama

print("\n🎯 Semua gambar selesai diproses.")
