import keras
import cv2 as cv
import numpy as np
from keras.applications.vgg16 import decode_predictions

# Завантаження моделі
model = keras.models.load_model("Trained_model_epochs-15_rate-0.001_loss-0.868_acc-0.571.keras")

# Обробка зображення для передбачення
file_name = "orange_cat.png"
# file_name = "gray_cat.jpg"
img = cv.imread(file_name)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img1 = cv.resize(img, [224, 224])
# приводимо до вхідного формату VGG-мережі
img1 = keras.applications.vgg16.preprocess_input(img1)
img1 = np.expand_dims(img1, axis=0) # можна використати reshape

# Передбачення
prediction = model.predict(img1)
predicted_class = np.argmax(prediction)
print(f"Передбачений клас: {predicted_class}")

# # Обробка зображення для передбачення
# image_name = "orange_cat.png"
# im1 = cv.imread(image_name, cv.IMREAD_GRAYSCALE)  # як чорно-біле
# im_arr = cv.resize(im1, [28, 28])  # масштабування
# im_arr = im_arr.reshape([1, 28, 28, 1])  # додаємо розмірності
# im_arr = im_arr / 255  # стандартизація
