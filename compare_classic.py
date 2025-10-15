# compare_classical.py

import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from skimage.feature import local_binary_pattern

# ----------------------------
# 1. Funcții auxiliare
# ----------------------------

def load_images_from_folder(folder, img_size=(48,48)):
    """
    Parcurge toate imaginile dintr-un folder care conține subfoldere (nume de clasă),
    returnează:
      - X: lista de imagini grayscale redimensionate la img_size
      - y: lista de label-uri (index corespunzător subfolderului)
      - class_names: lista de nume de clasă, ordonate alfabetic
    """
    class_names = sorted(os.listdir(folder))
    # Filtrăm doar directoarele
    class_names = [c for c in class_names if os.path.isdir(os.path.join(folder, c))]
    class_to_idx = {c:i for i,c in enumerate(class_names)}

    X = []
    y = []

    for cls in class_names:
        cls_folder = os.path.join(folder, cls)
        image_files = glob(os.path.join(cls_folder, "*.*"))
        for img_path in image_files:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(class_to_idx[cls])

    X = np.array(X)    # shape = (N, H, W)
    y = np.array(y)    # shape = (N,)
    return X, y, class_names

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """
    Afișează o heatmap pentru matricea de confuzie `cm`.
    """
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def extract_lbp_features(image, P=8, R=1, grid_x=8, grid_y=8):
    """
    Pentru un patch grayscale (48×48), calculează LBP (uniform) și apoi
    împarte imaginea în `grid_x × grid_y` celule și returnează histogramele LBP
    concatenându-le într-un singur vector.
    - P = 8 (număr de pixeli în cerc)
    - R = 1 (raza)
    - n_bins (coșuri) = P + 2 = 10 (metoda 'uniform' produce exact P+2 stări)
    """
    # 1) calculăm LBP uniform:
    lbp = local_binary_pattern(image, P, R, method='uniform')  
    # Metoda 'uniform' generează în mod determinist valori în [0..P+1] => num_buckets = P+2
    n_bins = P + 2  # ex: P=8 => 10 coșuri 

    # 2) împărțim în grid
    h, w = image.shape
    cell_h = h // grid_y  # 48//8 = 6
    cell_w = w // grid_x  # 48//8 = 6

    hist = []
    for i in range(grid_y):
        for j in range(grid_x):
            cell = lbp[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            # histogramă pe valorile LBP din această celulă
            (hist_cell, _) = np.histogram(cell.ravel(),
                                          bins=n_bins,
                                          range=(0, n_bins),
                                          density=True)
            hist.append(hist_cell)
    hist = np.concatenate(hist)  # vector lung: 64 × 10 = 640
    return hist


# ----------------------------
# 2. Încărcăm datele (train + val)
# ----------------------------
train_folder = "./train"
val_folder   = "./test"
IMG_SIZE = (48, 48)

print("Loading training data …")
X_train, y_train, class_names = load_images_from_folder(train_folder, img_size=IMG_SIZE)
print(f"  → Found {X_train.shape[0]} train images in {len(class_names)} classes.")

print("Loading validation data …")
X_val, y_val, _ = load_images_from_folder(val_folder, img_size=IMG_SIZE)
print(f"  → Found {X_val.shape[0]} test images.")

num_classes = len(class_names)

# ----------------------------
# 3. PCA + SVM
# ----------------------------
print("\n[1] PCA + SVM")

# 3.1 Flatten (N,48,48) → (N,2304)
n_train = X_train.shape[0]
n_val   = X_val.shape[0]
X_train_flat = X_train.reshape(n_train, -1).astype(np.float32) / 255.0
X_val_flat   = X_val.reshape(n_val,   -1).astype(np.float32) / 255.0

# 3.2 Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_val_scaled   = scaler.transform(X_val_flat)

# 3.3 Alegem câte componente PCA:
pca = PCA(0.95)  # păstrăm 95% din varianță
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca   = pca.transform(X_val_scaled)
n_components = X_train_pca.shape[1]
print(f"PCA → num_components = {n_components} pentru 95% varianță")

# 3.4 Antrenăm un SVM (RBF Kernel)
svm_pca = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', verbose=False)
svm_pca.fit(X_train_pca, y_train)

# 3.5 Prezicem pe validare
y_pred_pca = svm_pca.predict(X_val_pca)

# 3.6 Matrice de confuzie + raport
cm_pca = confusion_matrix(y_val, y_pred_pca)
print("\nMatrice confuzie (PCA+SVM):")
plot_confusion_matrix(cm_pca, class_names, title="PCA + SVM")

print("\nClassification Report (PCA+SVM):")
print(classification_report(y_val, y_pred_pca, target_names=class_names))



# ----------------------------
# 4. LBP + SVM
# ----------------------------
print("\n[2] LBP + SVM")

# 4.1 Extragem featuri LBP pentru fiecare imagine (train + val)
print("  → Calculăm histograme LBP pentru train …")
lbp_histograms_train = []
for img in tqdm(X_train, total=n_train):
    hist = extract_lbp_features(img, P=8, R=1, grid_x=8, grid_y=8)
    lbp_histograms_train.append(hist)
X_train_lbp = np.array(lbp_histograms_train)  # shape = (N_train, 640)

print("  → Calculăm histograme LBP pentru val …")
lbp_histograms_val = []
for img in tqdm(X_val, total=n_val):
    hist = extract_lbp_features(img, P=8, R=1, grid_x=8, grid_y=8)
    lbp_histograms_val.append(hist)
X_val_lbp = np.array(lbp_histograms_val)  # shape = (N_val, 640)

print("  → Formă LBP features:", X_train_lbp.shape)

# 4.2 Standardizăm
scaler_lbp = StandardScaler()
X_train_lbp_scaled = scaler_lbp.fit_transform(X_train_lbp)
X_val_lbp_scaled   = scaler_lbp.transform(X_val_lbp)

# 4.3 Antrenăm SVM pe LBP
svm_lbp = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', verbose=False)
svm_lbp.fit(X_train_lbp_scaled, y_train)

y_pred_lbp = svm_lbp.predict(X_val_lbp_scaled)

cm_lbp = confusion_matrix(y_val, y_pred_lbp)
print("\nMatrice confuzie (LBP+SVM):")
plot_confusion_matrix(cm_lbp, class_names, title="LBP + SVM")

print("\nClassification Report (LBP+SVM):")
print(classification_report(y_val, y_pred_lbp, target_names=class_names))



# ----------------------------
# 5. Figură comparativă a acuratețelor
# ----------------------------
acc_pca = np.diag(cm_pca).sum() / cm_pca.sum()
acc_lbp = np.diag(cm_lbp).sum() / cm_lbp.sum()

plt.figure(figsize=(5,4))
methods = ["PCA + SVM", "LBP + SVM"]
scores  = [acc_pca, acc_lbp]
sns.barplot(x=methods, y=scores, palette="viridis")
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Comparativ Acuratețe: PCA+SVM vs LBP+SVM")
for i,v in enumerate(scores):
    plt.text(i, v+0.01, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.show()
