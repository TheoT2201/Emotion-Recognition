import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
import os

class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_generator, class_indices, interval=15, 
                 save_dir="confusion_matrices", display_time=5):
        """
        validation_generator: generatorul de validare (ex. val_gen)
        class_indices: dicționarul train_gen.class_indices
        interval: la câte epoci să genereze matricea (ex. 15)
        save_dir: directorul unde se salvează fișierele PNG
        display_time: câte secunde rămâne afișată imaginea
        """
        super().__init__()
        self.validation_generator = validation_generator
        self.classes = list(class_indices.keys())
        self.interval = interval
        self.save_dir = save_dir
        self.display_time = display_time

        # Creăm folderul dacă nu există
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1
        if current_epoch % self.interval == 0:
            # 1) Predict pe întreg setul de validare
            val_probs = self.model.predict(self.validation_generator, verbose=0)
            val_preds = np.argmax(val_probs, axis=1)
            val_true = self.validation_generator.classes

            # 2) Calculăm matricea de confuzie
            cm = confusion_matrix(val_true, val_preds)

            # 3) Construim figura
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=self.classes, yticklabels=self.classes
            )
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix at Epoch {current_epoch}')

            # 4) Salvăm figura pe disc
            save_path = os.path.join(self.save_dir, f"cm_epoch_{current_epoch}.png")
            plt.savefig(save_path)
            print(f"[INFO] Confusion matrix for epoch {current_epoch} saved to: {save_path}")

            # 5) Afișăm non-blocking, așteptăm display_time secunde, apoi închidem
            plt.show(block=False)
            plt.pause(self.display_time)
            plt.close()