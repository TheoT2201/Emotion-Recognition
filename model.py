from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization,
    Input, GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0

def build_emotion_model(input_shape=(48,48,1), num_classes=7):
    model = Sequential()
    # bloc 1
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    #model.add(Dropout(0.1))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    #model.add(Dropout(0.1))

    # bloc 2
    model.add(Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.15))

    # bloc 3
    model.add(Conv2D(192, (3,3), activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Conv2D(192, (3,3), activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.15))

    # clasificator
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    return model