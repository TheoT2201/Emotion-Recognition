import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix

# Pentru CNN "de la zero"
from model import build_emotion_model

# Callback-uri și Sequence-uri
from mixed_sequence import MixedImageSequence
from confusion_matrix_callback import ConfusionMatrixCallback

BATCH_SIZE = 64

# setări
img_size = (48,48)
train_dir = './train_balanced'
validation_dir = './test'


# data augment & split
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,         # mai multă rotație
    width_shift_range=0.05,     # translație pe orizontală
    height_shift_range=0.05,    # translație pe verticală
    shear_range=0.1,          # shear mai mare
    zoom_range=0.15,            # zoom
    brightness_range=(0.9, 1.1), # îmbunătățire contraste
    horizontal_flip=True,
    validation_split=0.2       # 20% pentru validare
)

val_datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2
)

# generator de bază (aug. moderată)
base_gen = train_datagen.flow_from_directory(
    train_dir, target_size=img_size,
    color_mode='grayscale', batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training',
    shuffle=True
)

# definiți parametrii de augment agresiv pentru rare
rare_gen_params = {
    "rotation_range": 20,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "zoom_range": 0.2,
    "brightness_range": (0.8, 1.2),
    "horizontal_flip": True,
    "fill_mode": "nearest"
}

# clasele rare
classes_rare = ['disgust', 'fear']

# construim sequence-ul mixt
train_seq = MixedImageSequence(
    base_gen=base_gen,
    rare_dir=train_dir,
    rare_gen_params=rare_gen_params,
    classes_rare=classes_rare,
    class_indices=base_gen.class_indices,
    batch_size=BATCH_SIZE,
    target_size=img_size
)

val_gen = val_datagen.flow_from_directory(
    validation_dir, target_size=img_size,
    color_mode='grayscale', batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation',
    shuffle=False
)

print("Mapping clase→indice:", base_gen.class_indices)

def focal_loss(gamma=2.0, alpha=None):
    def focal_loss_fixed(y_true, y_pred):
        # evită log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        # loss-ul standard (cross‐entropy)
        cross_entropy = -y_true * tf.math.log(y_pred)
        # dacă ai alpha, aplici ponderile de clasă
        if alpha is not None:
            # y_true are one-hot, deci y_true * alpha va extrage alpha[i] pentru clasa i
            alpha_factor = y_true * alpha
        else:
            alpha_factor = 1.0
        # pe fiecare componentă, aplici (1 − p_i)^γ
        weight = alpha_factor * tf.pow(1.0 - y_pred, gamma)
        # loss final = sum peste clase de (weight * cross_entropy)
        loss = weight * cross_entropy
        # reduce pe axa claselor (sau axis=1), pentru a obține un vector (batch_size,)
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed

labels = base_gen.classes
classes = np.unique(labels)
raw_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)

# Aplică un scaling lin
min_w, max_w = raw_weights.min(), raw_weights.max()
scaled_weights = 0.7 + (raw_weights - min_w) * (1.3 - 0.7) / (max_w - min_w)
class_weights = dict(enumerate(scaled_weights))
print("class_weights:", class_weights)

alpha = np.zeros((len(classes),), dtype=np.float32)

for i in classes:
    alpha[i] = class_weights[i]
alpha = alpha / np.sum(alpha)


# model
model = build_emotion_model(input_shape=(*img_size,1), num_classes=len(base_gen.class_indices))
model.compile(
    optimizer=Adam(1e-3),
    loss=focal_loss(gamma=2.0, alpha=alpha),
    metrics=['accuracy']
)

# callback pentru cel mai bun val_accuracy
chkpt = ModelCheckpoint(
    'model.weights.h5', monitor='val_accuracy',
    save_best_only=True, save_weights_only=True,
    mode='max', verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',      # sau 'val_accuracy'
    factor=0.5,              # LR nou = LR vechi * 0.5
    patience=8,              # așteaptă 8 epoci fără îmbunătățire
    min_lr=1e-6,             # LR minim 
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    mode='max',
    restore_best_weights=True,
    verbose=1
)

# Instanțiem callback-ul
cm_callback = ConfusionMatrixCallback(val_gen, base_gen.class_indices, interval=15, save_dir="confusion_matrices", display_time=5)

# antrenare
history = model.fit(
    train_seq,
    steps_per_epoch=len(train_seq),
    epochs=60,
    validation_data=val_gen,
    validation_steps=len(val_gen),
    callbacks=[chkpt, reduce_lr, early_stop, cm_callback],
    class_weight=class_weights
)


# Plot the train and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the train and validation accuracy
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, train_acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Afișăm și matricea finală de confuzie (după ultima epocă)
val_probs = model.predict(val_gen, verbose=0)
val_preds = np.argmax(val_probs, axis=1)
val_true = val_gen.classes
cm_final = confusion_matrix(val_true, val_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_final, annot=True, fmt='d', cmap='YlGnBu',
    xticklabels=list(base_gen.class_indices.keys()),
    yticklabels=list(base_gen.class_indices.keys())
)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Final Confusion Matrix')
plt.show()