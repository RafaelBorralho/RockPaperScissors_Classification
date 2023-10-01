import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import time
import os
import cv2
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import random

directory = 'rps_images'
categories = ['rock', 'paper', 'scissors']

IMG_SIZE = 50
train_data = []
X = []
y = []

def create_data():
    for category in categories:
        label = categories.index(category)
        path = os.path.join(directory, category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            train_data.append([resized_img_array, label])


create_data()

random.shuffle(train_data)

for feature, label in train_data:
    X.append(feature)
    y.append(label)

X = np.array(X).reshape((-1, IMG_SIZE, IMG_SIZE, 1))
y = np.array(y)

X = X/255.0

data_split = int(len(X)*0.9)

X_train = X[:data_split]
y_train = y[:data_split]

X_test = X[data_split:]
y_test = y[data_split:]

conv_layers = [1, 2, 3]
dense_layers = [0, 1, 2]
layer_size = [32, 64, 128]
best_error = 100
'''
for conv in conv_layers:
    for dense in dense_layers:
        for size in layer_size:
            name = f'rps-{conv}-conv-{dense}-dense-{size}-layer-size-time-{int(time.time())}'
            tensorboard = TensorBoard(log_dir=f'logs/{name}')

            model = keras.Sequential()

            model.add(Conv2D(size, kernel_size=(3, 3), activation='relu', input_shape=X.shape[1:]))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for c in range(conv-1):
                model.add(Conv2D(size, kernel_size=(3, 3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            for d in range(dense):
                model.add(Dense(size, activation='relu'))

            model.add(Dense(3, activation='softmax'))
            model.summary()

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=32, callbacks=[tensorboard])

            error = model.evaluate(X_test, y_test)[0]
            if error < best_error:
                model.save('rps_classification.keras')

'''

model = keras.models.load_model('rps_classification.keras')
results = model.evaluate(X_test, y_test)
print(f'error is {results[0]} and accuracy is {results[1]}')
