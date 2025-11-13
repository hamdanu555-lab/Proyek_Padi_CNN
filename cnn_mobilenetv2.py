# =============================================================
# DETEKSI PENYAKIT DAUN PADI MENGGUNAKAN TRANSFER LEARNING (MobileNetV2)
# =============================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# === 1. Parameter dasar ===
img_size = (128, 128)
batch_size = 32
epochs = 10

# === 2. Data Generator (dengan augmentasi sederhana) ===
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2
)

val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "data_split/train",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)
val_data = val_gen.flow_from_directory(
    "data_split/val",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)
test_data = test_gen.flow_from_directory(
    "data_split/test",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# === 3. Inisialisasi Model MobileNetV2 ===
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,       # buang layer klasifikasi bawaan
    input_shape=(*img_size, 3)
)

# Bekukan layer dasar (tidak dilatih ulang)
for layer in base_model.layers:
    layer.trainable = False

# === 4. Tambahkan layer klasifikasi kita ===
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# === 5. Kompilasi model ===
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === 6. Training model ===
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)

# === 7. Simpan model ===
model.save("model_padi_mobilenetv2.h5")
print("✅ Model berhasil disimpan sebagai model_padi_mobilenetv2.h5")

# === 8. Evaluasi model ===
loss, acc = model.evaluate(test_data)
print(f"🎯 Akurasi data uji: {acc*100:.2f}%")

# === 9. Visualisasi hasil ===
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Akurasi Training')
plt.plot(history.history['val_accuracy'], label='Akurasi Validasi')
plt.legend()
plt.title("Akurasi Model MobileNetV2")
plt.xlabel("Epoch")
plt.ylabel("Akurasi")
plt.show()
