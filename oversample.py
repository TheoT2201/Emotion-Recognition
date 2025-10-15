# oversample.py

import os
import shutil
import random
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img

# === Configurație ===

# 1. Unde se află directorul cu imaginile originale
SOURCE_DIR = "train"

# 2. Unde vrem să scriem directorul peste care vom face oversampling
TARGET_DIR = "train_balanced"

# 3. Pentru care clase facem oversampling, și la câte imagini totale vrem să ajungem
TARGET_COUNTS = {
    "disgust": 2000
}

# 4. Parametrii de augmentare:
AUG_PARAMS = {
    "rotation_range":  10,
    "width_shift_range":  0.05,
    "height_shift_range": 0.05,
    "zoom_range": 0.1,
    "horizontal_flip": True,
    "brightness_range": (0.9, 1.1),
    "fill_mode": "nearest"
}

# 5. Prefix pentru imaginile noi generate
AUG_PREFIX = "aug"

# ====================================

def ensure_dir_exists(path):
    """Creează folderul dacă nu există."""
    if not os.path.isdir(path):
        os.makedirs(path)

def copy_all_classes(src_dir, tgt_dir):
    """
    Copiază recursiv toate subfolderele (clasele) și imaginile lor
    din src_dir în tgt_dir, păstrând structura.
    """
    for cls_name in os.listdir(src_dir):
        cls_src_path = os.path.join(src_dir, cls_name)
        cls_tgt_path = os.path.join(tgt_dir, cls_name)
        if os.path.isdir(cls_src_path):
            ensure_dir_exists(cls_tgt_path)
            # Copiem fiecare fișier din clasa respectivă
            for filename in os.listdir(cls_src_path):
                src_file = os.path.join(cls_src_path, filename)
                tgt_file = os.path.join(cls_tgt_path, filename)
                if os.path.isfile(src_file):
                    shutil.copy(src_file, tgt_file)

def get_image_filepaths(folder):
    """Returnează lista de căi absolute către fișierele de imagine dintr-un folder."""
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    all_files = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(exts):
            all_files.append(os.path.join(folder, fname))
    return all_files

def generate_augmentations_for_class(class_name, needed, gen_params, target_folder):
    """
    Generează <needed> imagini noi, pornind de la imaginile deja existente
    în target_folder/class_name (care conține deja imaginile deja copiate),
    și salvează-le tot în target_folder/class_name sub nume noi.

    gen_params: dicționar cu parametri pentru ImageDataGenerator
    """
    cls_folder = os.path.join(target_folder, class_name)
    filepaths = get_image_filepaths(cls_folder)
    if not filepaths:
        print(f"[WARN] Clasa '{class_name}' nu are imagini în '{cls_folder}'! Sar peste augmentare.")
        return

    datagen = ImageDataGenerator(**gen_params)

    # Vom itera ciclând prin imaginile existente și generând câte unul/două augmentate la fiecare pas
    idx = 0
    created = 0
    # Pentru a nu genera mai multe decât avem nevoie, iterăm până la needed
    while created < needed:
        # Alegem aleator un fișier din filepaths
        base_fp = random.choice(filepaths)
        # Încărcăm imaginea în array
        img = load_img(base_fp, color_mode="rgb", target_size=None)
        arr = img_to_array(img)  # shape = (h, w, 3)
        arr = np.expand_dims(arr, 0)  # shape = (1, h, w, 3)

        # Cream un generator mini‐batch, dar extragem doar o imagine augmentată
        # flow(…, batch_size=1) → yield un array cu 1 imagine
        aug_iter = datagen.flow(arr, batch_size=1)

        # Extragem imaginile generate până la needed-rămase
        # (putem genera și mai multe într-o singură iterație, dar aici doar una)
        batch = next(aug_iter)[0].astype("uint8")

        # Construim un nou nume de fișier: original_basename + prefix + numar_cu_zerouri + .jpg
        base_name = os.path.splitext(os.path.basename(base_fp))[0]
        new_name = f"{base_name}_{AUG_PREFIX}_{idx:04d}.jpg"
        new_fp = os.path.join(cls_folder, new_name)

        # Salvăm imaginea augmentată
        save_img(new_fp, batch)
        created += 1
        idx += 1

        if created % 100 == 0:
            print(f"[INFO] Generat {created}/{needed} imagini augmentate pentru clasa '{class_name}'...")

    print(f"[OK ] Am generat {needed} imagini noi pentru clasa '{class_name}'.")

def main():
    # 1. Creează (dacă nu există) folderul TARGET_DIR și copiază tot în el
    print(f"★ Creez/copiez imaginile din '{SOURCE_DIR}' în '{TARGET_DIR}' ...")
    copy_all_classes(SOURCE_DIR, TARGET_DIR)
    print("★ Copiere completă.\n")

    # 2. Pentru fiecare clasă din TARGET_COUNTS, calculăm câte imagini lipsesc
    for cls_name, target_count in TARGET_COUNTS.items():
        cls_src_folder = os.path.join(SOURCE_DIR, cls_name)
        cls_tgt_folder = os.path.join(TARGET_DIR, cls_name)

        if not os.path.isdir(cls_src_folder):
            print(f"[ERROR] Clasa '{cls_name}' nu există în source ({cls_src_folder}). Sar peste.")
            continue

        ensure_dir_exists(cls_tgt_folder)
        existing = len(get_image_filepaths(cls_tgt_folder))
        needed = target_count - existing
        if needed <= 0:
            print(f"[INFO] Clasa '{cls_name}' are deja {existing} imagini (≥ {target_count}). Nu fac oversampling.")
            continue

        print(f"★ Clasa '{cls_name}': există {existing}, target {target_count}, deci creez {needed} imagini noi.")
        # 3. Generează exact `needed` imagini augmentate
        generate_augmentations_for_class(cls_name, needed, AUG_PARAMS, TARGET_DIR)

    print("\n★ Oversampling finalizat. Găsești datele în:", TARGET_DIR)

if __name__ == "__main__":
    main()
