import keras
import numpy as np
from keras.layers import Dense
import matplotlib.pyplot as plt

learning_rate = 0.1 # швидкість навчання
nEpochs = 20 # кількість епох -- скільки разів будемо пропускати навчальний набір для навчання мережі

# Створення набору даних. Набір ознак:
features = [[0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]]
# Набір міток, бінарне подання: вектори, які містять лише 0 на 1
labels = [[0, 1],  # перший нейрон - true, другий нейрон - false
          [1, 0],  # one-hot-encoding подяння, бінарний вектор для кожного значення
          [1, 0],
          [1, 0],
          [1, 0],
          [1, 0],
          [1, 0],
          [1, 0]]

features = np.array(features)
labels = np.array(labels)


#Архітектура моделі
initializer = keras.initializers.GlorotNormal(seed=12) #=Xavier
model = keras.Sequential([
    Dense(units=2, input_shape=(3,), kernel_initializer=initializer, activation='sigmoid'),
    Dense(units=2, kernel_initializer=initializer, activation='softmax')
])
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.95, nesterov=True))

#Навчання моделі
print('learning start')
history = model.fit(features, labels, epochs=nEpochs, batch_size=1, verbose=1)
print('learning finish')

#Оцінювання моделі. Набір даних невеликий, всього 4 елемента. Тому оцінюємо і навчаємо на одному наборі
print('evaluation:')
model.evaluate(features, labels) #вибірка замала, вибірки для валідації немає зовсім. Тому проводимо валідацію на тренувальній вибірці
# Вочевидь, що тут loss = loss при fit, тому що набор для навчання і оцінювання однакові

#Використання моделі. Викликаємо метод predict, подаючи йому, наприклад, вектор (0, 0), який відповідає False XOR False
audit = np.array([[0, 0, 0]])
print('prediction:')
audit_output = model.predict(audit)
print(audit_output)
if np.argmax(audit_output):
    print('False XOR False')
else:
    print('True XOR True')



#Побудова графіків
plt.plot(history.history['loss'])
plt.grid(True)
plt.show()