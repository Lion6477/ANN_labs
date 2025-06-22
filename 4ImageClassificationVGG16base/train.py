from keras.layers import Dense
from tensorflow.keras.preprocessing.image import (ImageDataGenerator)
import keras
import numpy as np
import d_create

learning_rate = 0.001
epochs = 15
batch_size = 4
n_iny = 224
n_inx = 224

traingen, testgen = d_create.run()

# 1) Завантажуємо вихідну навчену мережу як model_base.
model_base = keras.applications.VGG16(weights='imagenet', input_shape=(n_iny, n_inx, 3),
                                      include_top=True)
# 2) Заморожуємо всі шари model_base.
model_base.trainable = False
# 3) виводимо структуру моделі у консоль.
model_base.summary()
# 4) Створюємо нову модель model_top. Її вихід співпадає з виходом 2 шару з кінця model_base
model_top = model_base.layers[-2].output
# 5) Додаємо вихідний шар до model_top.
n_out = 3
model_top = Dense(n_out, activation='softmax')(model_top)
# 6) “Склеюємо” модель.
model = keras.models.Model(inputs=model_base.input, outputs=model_top)
# 7) Вказуємо функцію втрат, метод навчання
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.95,
                                             nesterov=True), metrics=['accuracy'])
# d_create.run()
history = model.fit(
    traingen,  # генератор навчальних даних
    epochs=epochs,
    validation_data=testgen  # генератор тестових (валідаційних) даних
)

# Отримуємо фінальні втрати і точність
final_loss, final_accuracy = model.evaluate(testgen)

# Формуємо ім'я файлу моделі
name = f"Trained_model_epochs-{epochs}_rate-{learning_rate}_loss-{round(final_loss, 3)}_acc-{round(final_accuracy, 3)}.keras"

print(f"Save model into file \"{name}\"? y/n")

if input().lower() == "n":
    print("Model not saved")
else:
    print(f"Saving as {name}")
    model.save(name)
    print("Model saved")

