# ================================================================
# STREAMLIT WEB APP - DETEKSI PENYAKIT DAUN PADI
# ================================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tempfile

# === Load Model ===
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_padi_mobilenetv2.h5")
    return model

model = load_model()

# === Judul Halaman ===
st.title("🌾 Deteksi Dini Penyakit Daun Padi Menggunakan CNN (MobileNetV2)")
st.write("Upload gambar daun padi untuk mendeteksi apakah **Sehat** atau **Tidak Sehat**.")

# === Upload Gambar ===
uploaded_file = st.file_uploader("Pilih gambar daun padi...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Simpan file sementara di server Streamlit
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Baca gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # === Preprocessing ===
    img = image.resize((128, 128))
    img = np.array(img.convert("RGB"))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # === Prediksi ===
    prediction = model.predict(img, verbose=0)[0][0]
    label = "Tidak Sehat" if prediction > 0.5 else "Sehat"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # === Tampilkan hasil ===
    st.subheader("🔍 Hasil Deteksi:")
    st.write(f"**Status:** {label}")
    st.write(f"**Tingkat Keyakinan:** {confidence*100:.2f}%")

    # Tambah warna hasil
    if label == "Sehat":
        st.success("🌿 Daun padi terdeteksi **Sehat**")
    else:
        st.error("⚠️ Daun padi terdeteksi **Tidak Sehat**")
