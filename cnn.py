import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import matplotlib.pyplot as plt
import os

# Veri setinin yolu
base_dir = r'C:\Users\1must\Desktop\MeyveVeriseti\dataset'  

# Eğitim ve test veri yolları
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Eğitim ve test veri setlerini yükle
train_ds = keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(256, 256),  # Resim boyutunu değiştirin
    batch_size=32
)

test_ds = keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(256, 256),
    batch_size=32
)

# Veri artırma uygulamak için bir ön işleme katmanı oluşturun
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

# Modeli oluştur
model = keras.Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(256, 256, 3)),  # Resimleri normalleştir
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_ds.class_names), activation='softmax')  # Sınıf sayısını değiştirin
])

# Modeli derle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Modeli eğit
history = model.fit(
    train_ds,
    validation_data=test_ds,  # Test verisi doğrulama için kullanılıyor
    epochs=10  # Eğitim döngüsü sayısını değiştirin
)


# Eğitim sonuçlarını görselleştir
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

model.save('fruit_vegetable_model.keras')





