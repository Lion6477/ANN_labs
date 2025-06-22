import keras
import cv2 as cv
import numpy as np

# Завантаження моделі
model = keras.models.load_model("T2_trained_model_epochs-3_rate-0.001_loss-0.034_acc-0.989.keras")

# Обробка зображення для передбачення
image_name = "image.png"
im1 = cv.imread(image_name, cv.IMREAD_GRAYSCALE)

im_arr = cv.resize(im1, [28, 28])
im_arr = im_arr.reshape((1, 28, 28, 1))  # Додаємо вимірності
im_arr = im_arr / 255.0  # Стандартизація

# Передбачення
prediction = model.predict(im_arr)
predicted_class = np.argmax(prediction)
print(f"Передбачений клас: {predicted_class}")

# # Обробка зображення для передбачення
# image_name = "image.png"
# im1 = cv.imread(image_name, cv.IMREAD_GRAYSCALE)  # як чорно-біле
# im_arr = cv.resize(im1, [28, 28])  # масштабування
# im_arr = im_arr.reshape([1, 28, 28, 1])  # додаємо розмірності
# im_arr = im_arr / 255  # стандартизація
