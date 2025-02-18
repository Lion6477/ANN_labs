import keras
import numpy as np
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import fill_csv as fl


if __name__ == '__main__':

    fl.run()

    # створення набору даних, зчитаних з csv файлу
    def csv_reader(file_name):
        csv_dataset = pd.read_csv(
            file_name,
            names=["Label", "x", "y"])
        # csv_dataset.head()
        dataset = csv_dataset.copy()
        dataset_labels = keras.utils.to_categorical(
            dataset.pop('Label'))  # перетворює у one-hot-encoding подання, у бінарний вектор для кожного значення
        dataset_features = np.array(dataset)
        return dataset_features, dataset_labels


    learning_rate = 0.01
    nEpochs = 200
    batch_size = 600

    # Завантаження даних для навчання і оцінювання
    features_train, labels_train = csv_reader(
        "saturn_data_train_second_ts.csv")
    features_val, labels_val = csv_reader(
        "saturn_data_eval_second_ts.csv")

    keras.utils.set_random_seed(123) #для відтворюваності результатів
    initializer = keras.initializers.GlorotNormal(seed=12)  # =Xavier
    model = keras.Sequential([
        Dense(input_shape=(2,), units=16, kernel_initializer=initializer, activation='relu'),
        Dense(units=3, kernel_initializer=initializer, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.95, nesterov=True))
    print('learning start')
    history = model.fit(features_train, labels_train, epochs=nEpochs, batch_size=batch_size,
                        verbose=1)  # , use_multiprocessing = True, workers = 16)
    print('learning finish')
    print('evaluation:')
    model.evaluate(features_val, labels_val)
    model.save("saturn_model.keras")
    print('predict=')
    # За допомогою моделі визначити, якому класу належить точка
    audit = np.array([[2,3]])
    audit_output = model.predict(audit)
    print(audit_output)
    print(np.argmax(audit_output))

    plt.plot(history.history['loss'])
    plt.grid(True)
    plt.show()

    plt.scatter(features_train[:, 0], features_train[:, 1], c=np.argmax(labels_train, axis=1), cmap='viridis')
    plt.colorbar()
    plt.show()

