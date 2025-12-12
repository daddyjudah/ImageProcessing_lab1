import os
import sys
import numpy as np
import cv2
from pathlib import Path
from glob import glob
from tqdm import tqdm
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import joblib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMG_SIZE = 224
LABELS = {"WithMask": 0, "WithoutMask": 1}

class SimpleFaceDetector:
    def __init__(self):
        self.face_cascade = None
        self._initialize()

    def _initialize(self):
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            else:
                logger.warning("Haar cascade not found, using full images")
        except:
            pass

    def detect_and_crop(self, img_bgr, min_size=20):
        if img_bgr is None:
            return None

        try:
            if self.face_cascade is not None:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

                if len(faces) > 0:
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    x, y, w, h = faces[0]

                    if w >= min_size and h >= min_size:
                        face = img_bgr[y:y+h, x:x+w]
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                        return face

            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return None

class FaceMaskDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(10) if augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])
        image = self.transform(image)
        return image, label

class SimpleCNN(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        ds = IMG_SIZE // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * ds * ds, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def extract_hog_features(images):
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hog_feat = hog(gray, orientations=9, pixels_per_cell=(16, 16), 
                      cells_per_block=(2, 2), feature_vector=True)
        features.append(hog_feat)
    return np.array(features)

def train_svm_knn(X_train, y_train, X_val, y_val):
    print("\n ОБУЧЕНИЕ SVM И KNN МОДЕЛЕЙ")

    X_train_hog = extract_hog_features(X_train)
    X_val_hog = extract_hog_features(X_val)

    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train_hog, y_train)
    svm_acc = svm_model.score(X_val_hog, y_val)
    joblib.dump(svm_model, "model_svm_hog.pkl")
    print(f"SVM обучена | Точность: {svm_acc:.4f}")

    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train_hog, y_train)
    knn_acc = knn_model.score(X_val_hog, y_val)
    joblib.dump(knn_model, "model_knn_orb.pkl")
    print(f"KNN обучена | Точность: {knn_acc:.4f}")

    return svm_acc, knn_acc

def train_cnn_model(train_folder="Face_Mask_Dataset/Train", 
                    epochs=10, 
                    batch_size=16, 
                    model_path="model_cnn.pth"):

    print("ОБУЧЕНИЕ CNN, SVM И KNN МОДЕЛЕЙ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    detector = SimpleFaceDetector()
    X_train_full, y_train_full = [], []

    for class_name, label in LABELS.items():
        class_dir = os.path.join(train_folder, class_name)
        if not os.path.exists(class_dir):
            print(f"Папка {class_dir} не найдена")
            continue

        files = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            files.extend(glob(os.path.join(class_dir, f"*{ext}")))

        print(f"{class_name}: {len(files)} файлов")

        for fpath in tqdm(files[:200], desc=f"Загрузка {class_name}"):
            img = cv2.imread(fpath)
            if img is not None:
                face = detector.detect_and_crop(img)
                if face is not None:
                    X_train_full.append(face)
                    y_train_full.append(label)

    X_train_full = np.array(X_train_full)
    y_train_full = np.array(y_train_full)

    if len(X_train_full) == 0:
        print("Нет данных для обучения")
        return None

    print(f"\nЗагружено {len(X_train_full)} изображений")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    print(f"Train: {len(X_train)} изображений")
    print(f"Validation: {len(X_val)} изображений")

    svm_acc, knn_acc = train_svm_knn(X_train, y_train, X_val, y_val)

    train_dataset = FaceMaskDataset(X_train, y_train, augment=True)
    val_dataset = FaceMaskDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("ОБУЧЕНИЕ CNN")

    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"Сохранена лучшая CNN модель")

    cnn_acc = best_val_acc / 100.0

    print("РЕЗУЛЬТАТЫ ВСЕХ МОДЕЛЕЙ")
    print(f"CNN Точность: {cnn_acc:.4f}")
    print(f"SVM Точность: {svm_acc:.4f}")
    print(f"KNN Точность: {knn_acc:.4f}")

    best_model_name = "cnn"
    if svm_acc > cnn_acc and svm_acc > knn_acc:
        best_model_name = "svm"
    elif knn_acc > cnn_acc:
        best_model_name = "knn"

    metrics = {
        "accuracy_cnn": float(cnn_acc),
        "accuracy_svm": float(svm_acc),
        "accuracy_knn": float(knn_acc),
        "best_model_name": best_model_name
    }

    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nМетрики сохранены в model_metrics.json")
    print(f"Лучшая модель: {best_model_name}")

    return model, cnn_acc

if __name__ == "__main__":
    model, val_acc = train_cnn_model()

    if model is not None:
        print(f"\nВсе модели успешно обучены")
