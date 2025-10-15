import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import Sequence

class MixedImageSequence(Sequence):
    """
    Produce la fiecare batch:
      - majoritatea imaginilor (80%) din generatorul de bază (augmentări moderate pentru toate clasele)
      - restul (20%) provin din clasele rare (disgust, fear), încărcate manual și augmentate.
    """
    def __init__(
        self,
        base_gen,           # generatorul de bază (augmentări moderate pe întreg setul)
        rare_dir,           # directorul unde stocăm imaginile oversamplate pentru clasele rare
        rare_gen_params,    # parametri pentru ImageDataGenerator(...) agresiv
        classes_rare,       # lista claselor rare, ex ['disgust', 'fear']
        class_indices,      # dict-ul train_gen.class_indices (mapping clasă → index)
        batch_size=64,
        target_size=(48,48)
    ):
        self.base_gen = base_gen
        self.batch_size = batch_size
        self.classes_rare = classes_rare
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.target_size = target_size

        # ImageDataGenerator „agresiv” pentru imaginile rare
        self.rare_datagen = ImageDataGenerator(**rare_gen_params)

        # Pregătim lista de fișiere pentru clasele rare
        self.rare_filepaths = []
        for cls in classes_rare:
            cls_folder = os.path.join(rare_dir, cls)
            if not os.path.isdir(cls_folder):
                raise ValueError(f"Folderul pentru clasa rară '{cls}' nu există: {cls_folder}")
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith((".jpg","jpeg","png","bmp")):
                    self.rare_filepaths.append(os.path.join(cls_folder, fname))
        if not self.rare_filepaths:
            raise ValueError("Nu am găsit niciun fișier în clasele rare!")

        # Câte exemple rare vom include în fiecare batch (20% din batch)
        self.num_rare_per_batch = int(0.2 * batch_size)

    def __len__(self):
        """
        Returnează numărul de pași (batch-uri) pe epocă, _fără_ ultimul batch parțial.
        Astfel, ne asigurăm că __getitem__ primește întotdeauna batch-uri complete de dimensiune batch_size.
        """
        total_samples = self.base_gen.samples
        full_batches = total_samples // self.batch_size
        return full_batches  # ignorăm ultimul batch parțial

    def __getitem__(self, idx):
        # 1) Obținem batch-ul complet de bază (batch_size imagini)
        batch_x_base, batch_y_base = self.base_gen[idx]
        # Ne asigurăm că are exact batch_size; nu intră aici ultimul parțial
        if batch_x_base.shape[0] != self.batch_size:
            raise ValueError(f"Așteptam batch complet de {self.batch_size}, dar am {batch_x_base.shape[0]}")

        keep_normal = self.batch_size - self.num_rare_per_batch

        # Preluăm primele keep_normal exemple
        batch_x = batch_x_base[:keep_normal]
        batch_y = batch_y_base[:keep_normal]

        # 2) Generăm imaginile rare manual
        rare_x = []
        rare_y = []
        for _ in range(self.num_rare_per_batch):
            fp = random.choice(self.rare_filepaths)
            cls_name = os.path.basename(os.path.dirname(fp))
            cls_index = self.class_indices[cls_name]

            # Încarcă imaginea ca array grayscale (48x48x1)
            img = load_img(fp, color_mode="grayscale", target_size=self.target_size)
            arr = img_to_array(img)  # shape = (48,48,1)
            arr = np.expand_dims(arr, 0)  # shape = (1,48,48,1)

            # Aplică o transformare aleatorie prin datagen
            aug_iter = self.rare_datagen.flow(arr, batch_size=1)
            aug_img = next(aug_iter)[0]  # shape = (48,48,1)

            # Rescale (dacă base_gen folosește rescale=1./255)
            aug_img = aug_img / 255.0

            rare_x.append(aug_img)

            # Construim vectorul one-hot de dimensiune num_classes
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[cls_index] = 1.0
            rare_y.append(one_hot)

        rare_x = np.array(rare_x)  # shape = (num_rare_per_batch, 48,48,1)
        rare_y = np.array(rare_y)  # shape = (num_rare_per_batch, num_classes)

        # 3) Concatenăm batch-urile
        batch_x = np.concatenate([batch_x, rare_x], axis=0)
        batch_y = np.concatenate([batch_y, rare_y], axis=0)

        # 4) Amestecăm batch-ul
        perm = np.random.permutation(self.batch_size)
        batch_x = batch_x[perm]
        batch_y = batch_y[perm]

        return batch_x, batch_y