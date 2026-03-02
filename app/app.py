import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# === 1. Konfigurasi dan Load Semua Model K-Fold ===
MODEL_DIR = r"C:\Users\Lenovo\Klasifikasi_ikan\hasil_model\final_training\models"
MODEL_FILES = [
    "final_model_fold_1.keras",
    "final_model_fold_2.keras",
    "final_model_fold_3.keras",
    "final_model_fold_4.keras",
    "final_model_fold_5.keras"
]

@st.cache_resource  # caching agar model tidak di-load berulang
def load_models():
    models = []
    for model_file in MODEL_FILES:
        path = os.path.join(MODEL_DIR, model_file)
        try:
            model = tf.keras.models.load_model(path)
            models.append(model)
            st.write(f"✅ Berhasil memuat {model_file}")
        except Exception as e:
            st.error(f"Gagal memuat {model_file}: {e}")
    return models

models = load_models()

# === 2. Label dan Threshold ===
class_labels = ['Segar', 'Tidak Segar']
threshold_not_fish = 0.70  # jika confidence < 70%, dianggap bukan ikan ekor kuning

# === 3. UI Streamlit ===
st.title("🐟 Deteksi Kesegaran Ikan Ekor Kuning Menggunakan MobileNetV2 (Ensemble K-Fold)")
st.write("""
Aplikasi ini berfungsi untuk mendeteksi **tingkat kesegaran ikan ekor kuning** berdasarkan area mata ikan.  
Silakan unggah gambar mata ikan untuk mendapatkan hasil klasifikasi.
""")

uploaded_file = st.file_uploader("📤 Upload Gambar Ikan", type=["jpg", "jpeg", "png"])

# === 4. Jika ada gambar diunggah ===
if uploaded_file is not None:
    # Membaca gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is not None:
        # Tampilkan citra asli
        st.image(img, channels="BGR", caption="Citra yang diunggah", use_container_width=True)

        # === Preprocessing ===
        img_resized = cv2.resize(img, (224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # === Prediksi Ensemble ===
        with st.spinner("🔍 Sedang memproses citra..."):
            predictions = [model.predict(img_array) for model in models]
            final_prediction = np.mean(predictions, axis=0)  # rata-rata probabilitas
            confidence = np.max(final_prediction)
            label_index = np.argmax(final_prediction)

        # === Hasil Prediksi ===
        if confidence < threshold_not_fish:
            result = "🚫 Gambar tidak dikenali sebagai ikan ekor kuning"
            color = "red"
        else:
            result = f"✅ {class_labels[label_index]} (Confidence: {confidence*100:.2f}%)"
            color = "green"

        # === Tampilkan Hasil ===
        st.subheader("📊 Hasil Klasifikasi")
        st.markdown(f"<h3 style='color:{color}'>{result}</h3>", unsafe_allow_html=True)

    else:
        st.error("❌ Gagal membaca citra. Pastikan file berformat .jpg, .jpeg, atau .png yang valid.")

else:
    st.info("Silakan unggah citra terlebih dahulu untuk melakukan prediksi.")
