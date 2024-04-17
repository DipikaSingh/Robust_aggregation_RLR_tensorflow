
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
import random

class Model:
    def __init__(self, x_train_shape=(28, 28, 1), output_labels=10):
        self.x_train_shape=x_train_shape
        self.output_labels=output_labels
    def cnn(self):
        model = Sequential()
        model.add(Conv2D(32, (2, 2), activation='relu', kernel_initializer='he_uniform', input_shape=self.x_train_shape))
        # model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (2, 2), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        #model.add(Dropout(0.5))
        model.add(Dense(self.output_labels, activation='softmax'))
        return model