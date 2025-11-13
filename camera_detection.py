# ============================================================
# DETEKSI DINI PENYAKIT DAUN PADI BERDASARKAN CITRA
# MENGGUNAKAN CNN (MobileNetV2)
# DENGAN FITUR CAPTURE & LOG HASIL PREDIKSI
# ============================================================

import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from datetime import datetime

# === 1. Load model ===
model = tf.keras.models.load_model("model_padi_mobilenetv2.h5")

# === 2. Kelas output ===
classes = ['Sehat', 'Tidak Sehat']

# === 3. Siapkan folder untuk hasil ===
save_dir = "captures"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# === 4. Siapkan file log prediksi ===
log_path = os.path.join(save_dir, "prediksi_log.txt")
if not os.path.exists(log_path):
    with open(log_path, "w") as f:
        f.write("Nama File\tLabel\tConfidence\tWaktu\n")

# === 5. Inisialisasi kamera ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera tidak bisa diakses.")
    exit()

print("=== SISTEM DETEKSI PENYAKIT DAUN PADI ===")
print("Tekan 'c' untuk capture gambar, atau 'q' untuk keluar.\n")

capture_count = 0

# === 6. Loop kamera ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Preprocessing untuk model
    img = cv2.resize(frame, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Prediksi model
    prediction = model.predict(img, verbose=0)[0][0]
    label = "Tidak Sehat" if prediction > 0.5 else "Sehat"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Tambahkan teks hasil prediksi di frame
    text = f"{label} ({confidence*100:.2f}%)"
    color = (0, 255, 0) if label == "Sehat" else (0, 0, 255)
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Tampilkan frame
    cv2.imshow("Deteksi Daun Padi", frame)

    # Kontrol keyboard
    key = cv2.waitKey(1) & 0xFF

    # === Jika tekan 'c' maka capture ===
    if key == ord('c'):
        capture_count += 1
        filename = os.path.join(save_dir, f"capture_{capture_count}.jpg")
        cv2.imwrite(filename, frame)

        waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Tulis ke log file
        with open(log_path, "a") as f:
            f.write(f"{os.path.basename(filename)}\t{label}\t{confidence*100:.2f}%\t{waktu}\n")

        print(f"[TERSIMPAN] {filename} | {label} ({confidence*100:.2f}%) | {waktu}")

    # === Jika tekan 'q' maka keluar ===
    elif key == ord('q'):
        print("\nSesi kamera selesai. Terima kasih telah menggunakan sistem deteksi daun padi.")
        break

# === 7. Tutup kamera dan window ===
cap.release()
cv2.destroyAllWindows()
