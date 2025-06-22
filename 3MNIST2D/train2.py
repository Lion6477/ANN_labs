from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import keras
import numpy as np
import cv2 as cv

# Налаштування параметрів навчання
learning_rate = 0.001
epochs = 3
batch_size = 64

# Завантаження даних
(features_train, labels_train), (features_test, labels_test) = mnist.load_data()

# Стандартизація вхідних даних. Значення кольору 0...255 -> 0..1
features_train = features_train / 255
features_test = features_test / 255

# Бінарне подання, 10 класів
labels_train_cat = keras.utils.to_categorical(labels_train, 10)
labels_test_cat = keras.utils.to_categorical(labels_test, 10)

# При роботі з зображенням остання розмірність -- кількість каналів
features_train = np.expand_dims(features_train, axis=3)  # додаємо кількість каналів, 1 = grayscale
features_test = np.expand_dims(features_test, axis=3)

# Побудова моделі
model = Sequential([
    Conv2D(20, (3, 3), padding='same', strides=(1, 1), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(50, (3, 3), padding='same', strides=(1, 1), activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(500, activation='relu'),
    Dense(10, activation='softmax')
])

# Створюємо оптимізатор з learning_rate
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# Компіляція моделі
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Навчання
history = model.fit(features_train, labels_train_cat, epochs=epochs, batch_size=batch_size, validation_data=(features_test, labels_test_cat))

# Отримуємо фінальні втрати і точність
final_loss, final_accuracy = model.evaluate(features_test, labels_test_cat)

# Формуємо ім'я файлу моделі
name = f"T2_trained_model_epochs-{epochs}_rate-{learning_rate}_loss-{round(final_loss, 3)}_acc-{round(final_accuracy, 3)}.keras"

print(f"Save model into file \"{name}\"? y/n")

if input().lower() == "n":
    print("Model not saved")
else:
    print(f"Saving as {name}")
    model.save(name)
    print("Model saved")
