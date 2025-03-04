# from tensorflow.keras.preprocessing.image import (ImageDataGenerator)
# from tensorflow import keras
#
# #der run
# def run():
#     train_dir_name = "C:/Users/sekre/PycharmProjects/ANN_labs/4lab/train_d"
#     test_dir_name = "C:/Users/sekre/PycharmProjects/ANN_labs/4lab/test_d"
#     batch_size = 4
#
#     generator = ImageDataGenerator(rescale=1 / 255.,
#                                    preprocessing_function=keras.applications.vgg16.preprocess_input)
#     traingen = generator.flow_from_directory(train_dir_name,
#                                              target_size=(224, 224),
#                                              class_mode='categorical', shuffle=False, batch_size=batch_size)
#     testgen = generator.flow_from_directory(test_dir_name,
#                                             target_size=(224, 224),
#                                             class_mode='categorical', shuffle=False, batch_size=batch_size)
#
# run()