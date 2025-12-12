import os
import sys
import argparse
import json
from glob import glob
from typing import Tuple, List
from mtcnn import MTCNN
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import telebot
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
import joblib


DATA_ROOT = "Face_Mask_Dataset"
IMG_SIZE = 224
DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
LABELS = {"WithMask": 0, "WithoutMask": 1}

SVM_PATH = "model_svm_hog.pkl"
KNN_PATH = "model_knn_orb.pkl"
CNN_PATH = "model_cnn.pth"

_FACE_DETECTOR = None


def ensure_mtcnn():
    global _FACE_DETECTOR
    if MTCNN is None:
        raise ImportError("mtcnn package is required. Install via `pip install mtcnn`.")
    if _FACE_DETECTOR is None:
        _FACE_DETECTOR = MTCNN()
    return _FACE_DETECTOR

def detect_and_crop_face_bgr(img_bgr: np.ndarray, min_size=20) -> np.ndarray:
    if img_bgr is None:
        return None
    detector = ensure_mtcnn()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb)
    if not detections:
        return None
    detections.sort(key=lambda d: d['box'][2] * d['box'][3], reverse=True)
    x, y, w, h = detections[0]['box']
    x, y = max(0, x), max(0, y)
    if w < min_size or h < min_size:
        return None
    face = rgb[y:y+h, x:x+w]
    if face.size == 0:
        return None
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    return face

def load_faces_from_folder(folder: str, extensions=(".jpg",".jpeg",".png",".bmp")) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    for class_name, label in LABELS.items():
        class_dir = os.path.join(folder, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: class directory not found: {class_dir}", file=sys.stderr)
            continue
        files = []
        for ext in extensions:
            files.extend(glob(os.path.join(class_dir, f"*{ext}")))
        for fpath in tqdm(files, desc=f"Loading {os.path.basename(folder)} / {class_name}"):
            img_bgr = cv2.imread(fpath)
            if img_bgr is None:
                continue
            face = detect_and_crop_face_bgr(img_bgr)
            if face is None:
                continue
            X.append(face)
            y.append(label)
    if len(X) == 0:
        return np.array([]), np.array([])
    return np.array(X), np.array(y)


def extract_hog_features(images_rgb: np.ndarray) -> np.ndarray:
    feats = []
    for img in images_rgb:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h = hog(gray, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2), feature_vector=True)
        feats.append(h)
    return np.array(feats)

def train_svm_hog(X_train_rgb: np.ndarray, y_train: np.ndarray):
    Xf = extract_hog_features(X_train_rgb)
    clf = SVC(kernel="linear", probability=True)
    clf.fit(Xf, y_train)
    joblib.dump(clf, SVM_PATH)
    return clf

def eval_svm(clf, X_rgb: np.ndarray, y_true: np.ndarray):
    Xf = extract_hog_features(X_rgb)
    preds = clf.predict(Xf)
    acc = accuracy_score(y_true, preds)
    return acc, preds


def orb_descriptor_vector(img_rgb: np.ndarray, max_kp=200, fixed_len=50):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=max_kp)
    kp, des = orb.detectAndCompute(gray, None)
    if des is None:
        return np.zeros(max_kp * 32, dtype=np.float32)[:fixed_len*32]
    des = des.astype(np.float32)
    if des.shape[0] >= fixed_len:
        des = des[:fixed_len].flatten()
    else:
        pad_len = fixed_len*32 - des.size
        des = np.pad(des.flatten(), (0, pad_len))
    return des

def extract_orb_features(images_rgb: np.ndarray, fixed_len=50) -> np.ndarray:
    feats = []
    orb = cv2.ORB_create(nfeatures=500)
    for img in images_rgb:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if des is None:
            dvec = np.zeros(fixed_len * 32, dtype=np.float32)
        else:
            des = des.astype(np.float32)
            if des.shape[0] >= fixed_len:
                dvec = des[:fixed_len].flatten()
            else:
                dvec = np.pad(des.flatten(), (0, fixed_len*32 - des.size))
        feats.append(dvec)
    return np.array(feats)

def train_knn_orb(X_train_rgb: np.ndarray, y_train: np.ndarray):
    Xf = extract_orb_features(X_train_rgb, fixed_len=50)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(Xf, y_train)
    joblib.dump(knn, KNN_PATH)
    return knn

def eval_knn(knn, X_rgb: np.ndarray, y_true: np.ndarray):
    Xf = extract_orb_features(X_rgb, fixed_len=50)
    preds = knn.predict(Xf)
    acc = accuracy_score(y_true, preds)
    return acc, preds


class IdentityTransform:
    def __call__(self, x):
        return x

if torch is not None:
    class FaceMaskDataset(Dataset):
        def __init__(self, imgs: np.ndarray, labels: np.ndarray, augment: bool = False):
            self.imgs = imgs
            self.labels = labels
            self.augment = augment
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip() if augment else IdentityTransform(),
                transforms.RandomRotation(10) if augment else IdentityTransform(),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
            ])
        def __len__(self):
            return len(self.imgs)
        def __getitem__(self, idx):
            img = self.imgs[idx]
            img = self.tf(img)
            label = int(self.labels[idx])
            return img, label

    class SimpleCNN(nn.Module):
        def __init__(self, n_classes=2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
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

    def train_cnn(X_train_rgb, y_train, X_val_rgb=None, y_val=None, epochs=7, batch_size=16, lr=1e-3):
        train_ds = FaceMaskDataset(X_train_rgb, y_train, augment=True)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        if X_val_rgb is not None:
            val_ds = FaceMaskDataset(X_val_rgb, y_val, augment=False)
            val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        else:
            val_dl = None

        model = SimpleCNN().to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            running = 0.0
            total = 0
            correct = 0
            loop = tqdm(train_dl, desc=f"Train Epoch {epoch+1}/{epochs}")
            for xb, yb in loop:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
                running += loss.item()
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
                loop.set_postfix(loss=running/total, acc=correct/total)
            print(f"Epoch {epoch+1} train loss: {running/len(train_dl):.4f} acc: {100*correct/total:.2f}%")
            if val_dl is not None:
                model.eval()
                vtotal = 0
                vcorrect = 0
                with torch.no_grad():
                    for xb, yb in val_dl:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        out = model(xb)
                        preds = out.argmax(dim=1)
                        vcorrect += (preds == yb).sum().item()
                        vtotal += yb.size(0)
                print(f"Validation acc: {100*vcorrect/vtotal:.2f}%")
        torch.save(model.state_dict(), CNN_PATH)
        return model

    def eval_cnn(model, X_rgb, y_true, batch_size=16):
        ds = FaceMaskDataset(X_rgb, y_true, augment=False)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        model.eval()
        preds = []
        true = []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(DEVICE)
                out = model(xb)
                p = out.argmax(dim=1).cpu().numpy()
                preds.extend(p.tolist())
                true.extend(yb.numpy().tolist())
        acc = accuracy_score(true, preds)
        return acc, np.array(preds), np.array(true)

else:
    def train_cnn(*args, **kwargs):
        raise ImportError("PyTorch is required for CNN training. Install torch & torchvision.")
    def eval_cnn(*args, **kwargs):
        raise ImportError("PyTorch is required for CNN evaluation. Install torch & torchvision.")


def train_all(train_folder: str, val_folder: str = None, test_folder: str = None):
    print("=== Loading datasets ===")
    X_train, y_train = load_faces_from_folder(train_folder)
    X_val, y_val = (np.array([]), np.array([]))
    if val_folder:
        X_val, y_val = load_faces_from_folder(val_folder)
    X_test, y_test = (np.array([]), np.array([]))
    if test_folder:
        X_test, y_test = load_faces_from_folder(test_folder)

    if len(X_train) == 0:
        raise RuntimeError("No training images loaded. Check dataset path and face detector.")

    print("\n--- Training HOG + SVM ---")
    svm = train_svm_hog(X_train, y_train)
    if len(X_test):
        acc_svm, preds_svm = eval_svm(svm, X_test, y_test)
        print("HOG+SVM test accuracy:", acc_svm)
        print(classification_report(y_test, preds_svm, target_names=list(LABELS.keys())))
    else:
        acc_svm = None

    print("\n--- Training ORB + KNN ---")
    knn = train_knn_orb(X_train, y_train)
    if len(X_test):
        acc_knn, preds_knn = eval_knn(knn, X_test, y_test)
        print("ORB+KNN test accuracy:", acc_knn)
        print(classification_report(y_test, preds_knn, target_names=list(LABELS.keys())))
    else:
        acc_knn = None

    if torch is not None:
        print("\n--- Training CNN (PyTorch) ---")
        model = train_cnn(X_train, y_train, X_val, y_val, epochs=7)
        if len(X_test):
            acc_cnn, preds_cnn, true_cnn = eval_cnn(model, X_test, y_test)
            print("CNN test accuracy:", acc_cnn)
            print(classification_report(true_cnn, preds_cnn, target_names=list(LABELS.keys())))
        else:
            acc_cnn = None
    else:
        acc_cnn = None

    accs = []
    if acc_svm is not None:
        accs.append((acc_svm, "hog_svm"))
    if acc_knn is not None:
        accs.append((acc_knn, "orb_knn"))
    if acc_cnn is not None:
        accs.append((acc_cnn, "cnn"))
    if not accs:
        print("No test accuracies available to compare. Save completed models anyway.")
        best = None
    else:
        best = max(accs, key=lambda x: x[0])
        print(f"\nBest model: {best[1]} with accuracy {best[0]:.4f}")

    metrics = {
        "accuracy_svm": acc_svm,
        "accuracy_knn": acc_knn,
        "accuracy_cnn": acc_cnn,
        "best_model_name": best[1]
    }

    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nSaved metrics to model_metrics.json")
    return metrics


def load_models_for_inference():
    svm = None
    knn = None
    cnn_model = None
    if os.path.exists(SVM_PATH):
        svm = joblib.load(SVM_PATH)
    if os.path.exists(KNN_PATH):
        knn = joblib.load(KNN_PATH)
    if os.path.exists(CNN_PATH) and torch is not None:
        cnn_model = SimpleCNN()
        cnn_model.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE))
        cnn_model.to(DEVICE)
        cnn_model.eval()
    return svm, knn, cnn_model

def predict_all_on_face(face_rgb: np.ndarray, svm, knn, cnn_model):
    res = {}
    if svm is not None:
        hf = hog(cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY), orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2), feature_vector=True)
        try:
            res['svm'] = int(svm.predict([hf])[0])
        except Exception:
            res['svm'] = None
    if knn is not None:
        d = orb_descriptor_vector(face_rgb, max_kp=500, fixed_len=50)
        try:
            res['knn'] = int(knn.predict([d])[0])
        except Exception:
            res['knn'] = None
    if cnn_model is not None and torch is not None:
        tf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                 transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        inp = tf(face_rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = cnn_model(inp)
            pred = int(out.argmax(dim=1).cpu().numpy()[0])
        res['cnn'] = pred
    return res

STATE_MENU = "menu"
STATE_MULTI = "multi"
STATE_BEST = "best"

user_states = {}

def run_telegram_bot(token: str, allowed_chat_ids: List[int] = None):
    global accuracy_svm, accuracy_knn, accuracy_cnn, best_model_name

    if os.path.exists("model_metrics.json"):
        with open("model_metrics.json", "r") as f:
            metrics = json.load(f)
            accuracy_svm = metrics.get("accuracy_svm")
            accuracy_knn = metrics.get("accuracy_knn")
            accuracy_cnn = metrics.get("accuracy_cnn")
            best_model_name = metrics.get("best_model_name")
        print("Loaded metrics from model_metrics.json")
    else:
        print("WARNING: model_metrics.json not found -> metrics unavailable.")
        accuracy_svm = None
        accuracy_knn = None
        accuracy_cnn = None
        best_model_name = None

    if telebot is None:
        raise ImportError("telebot (pyTelegramBotAPI) is required to run the bot. Install via `pip install pyTelegramBotAPI`.")
    bot = telebot.TeleBot(token)
    svm, knn, cnn_model = load_models_for_inference()
    if svm is None and knn is None and cnn_model is None:
        print("No models found. Train models first.")
        return

    @bot.message_handler(commands=['start'])
    def start(message):
        send_main_menu(bot, message.chat.id)

    def send_main_menu(bot, chat_id):
        bot.send_message(
            chat_id,
            "Что вы хотите сделать?\n"
            "1 — Показать accuracy всех моделей\n"
            "2 — Проверить фото всеми тремя моделями\n"
            "3 — Проверить фото лучшим методом\n\n"
            "Введите номер пункта:"
        )
        user_states[chat_id] = STATE_MENU

    @bot.message_handler(content_types=['text'])
    def handle_text(message):
        chat_id = message.chat.id
        text = message.text.strip().lower()

        if text == "назад":
            send_main_menu(bot, chat_id)
            return

        if user_states.get(chat_id) == STATE_MENU:
            if text == "1":
                acc_svm = globals().get("accuracy_svm", "N/A")
                acc_knn = globals().get("accuracy_knn", "N/A")
                acc_cnn = globals().get("accuracy_cnn", "N/A")
                best = globals().get("best_model_name", "N/A")
                msg = (
                    f"SVM (HOG): {acc_svm}\n"
                    f"KNN (ORB): {acc_knn}\n"
                    f"CNN: {acc_cnn}\n\n"
                    f"Лучшая модель: {best}"
                )
                bot.send_message(chat_id, msg)

            elif text == "2":
                bot.send_message(chat_id, "Отправьте фото. Напишите 'назад' чтобы вернуться.")
                user_states[chat_id] = STATE_MULTI

            elif text == "3":
                best = globals().get("best_model_name", None)
                bot.send_message(chat_id, f"Отправьте фото. Напишите 'назад' чтобы вернуться.\n"
                                          f"Лучший метод: {best}")
                user_states[chat_id] = STATE_BEST

            else:
                bot.send_message(chat_id, "Введите один из пунктов: 1, 2 или 3.")

    @bot.message_handler(content_types=['photo'])
    def handle_photo(message):
        chat_id = message.chat.id

        state = user_states.get(chat_id)
        if state not in [STATE_MULTI, STATE_BEST]:
            bot.send_message(chat_id, "Сначала выберите пункт в меню через /start")
            return

        file_info = bot.get_file(message.photo[-1].file_id)
        data = bot.download_file(file_info.file_path)
        arr = np.frombuffer(data, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            bot.send_message(chat_id, "Ошибка при загрузке изображения.")
            return

        face = detect_and_crop_face_bgr(img_bgr)
        if face is None:
            bot.send_message(chat_id, "Лицо не найдено.")
            return

        if state == STATE_MULTI:
            preds = predict_all_on_face(face, svm, knn, cnn_model)
            inv_labels = {v: k for k, v in LABELS.items()}

            msg = []
            for method in ("svm", "knn", "cnn"):
                if preds.get(method) is not None:
                    msg.append(f"{method.upper()}: {inv_labels[preds[method]]}")
                else:
                    msg.append(f"{method.upper()}: N/A")
            bot.send_message(chat_id, "\n".join(msg))

        elif state == STATE_BEST:
            best_model = globals().get("best_model_name")
            inv_labels = {v: k for k, v in LABELS.items()}

            if best_model.lower() == "svm":
                pred = predict_all_on_face(face, svm, None, None)["svm"]
            elif best_model.lower() == "knn":
                pred = predict_all_on_face(face, None, knn, None)["knn"]
            else:
                pred = predict_all_on_face(face, None, None, cnn_model)["cnn"]

            bot.send_message(chat_id, f"Результат ({best_model}): {inv_labels[pred]}")

    print("Bot polling started. Press Ctrl+C to exit.")
    bot.polling(none_stop=True)


def main():
    parser = argparse.ArgumentParser(description="FaceMask pipeline: train/eval/serve")
    parser.add_argument("--train-all", action="store_true", help="Train all three models (requires dataset)")
    parser.add_argument("--dataset-root", type=str, default=DATA_ROOT, help="Root dataset folder (default: dataset/FaceMask)")
    parser.add_argument("--run-bot", action="store_true", help="Start Telegram bot (requires token)")
    parser.add_argument("--telegram-token", type=str, default="8304860291:AAC6qdt304ladMUl_ta6FjbvxwEdgjaAqdA", help="Telegram Bot token")
    parser.add_argument("--classify-all", action="store_true", help="Classify Test set using trained models and print report")
    parser.add_argument("--train-cnn-only", action="store_true", help="Train only CNN")
    args = parser.parse_args()

    data_root = args.dataset_root
    train_folder = os.path.join(data_root, "Train")
    val_folder = os.path.join(data_root, "Validation")
    test_folder = os.path.join(data_root, "Test")

    if args.train_all:
        print("Training all models using dataset root:", data_root)
        res = train_all(train_folder, val_folder if os.path.isdir(val_folder) else None, test_folder if os.path.isdir(test_folder) else None)
        print("Train summary:", res)
        return

    if args.train_cnn_only:
        X_train, y_train = load_faces_from_folder(train_folder)
        X_val, y_val = (np.array([]), np.array([]))
        if os.path.isdir(val_folder):
            X_val, y_val = load_faces_from_folder(val_folder)
        if len(X_train) == 0:
            print("No training images found.")
            return
        model = train_cnn(X_train, y_train, X_val, y_val, epochs=7)
        print("CNN training done.")
        return

    if args.classify_all:
        X_test, y_test = load_faces_from_folder(test_folder)
        if len(X_test) == 0:
            print("No test images found.")
            return
        svm, knn, cnn_model = load_models_for_inference()
        if svm is not None:
            acc, preds = eval_svm(svm, X_test, y_test)
            print("SVM HOG acc:", acc)
            print(classification_report(y_test, preds, target_names=list(LABELS.keys())))
        if knn is not None:
            acc, preds = eval_knn(knn, X_test, y_test)
            print("KNN ORB acc:", acc)
            print(classification_report(y_test, preds, target_names=list(LABELS.keys())))
        if cnn_model is not None and torch is not None:
            acc, preds, true = eval_cnn(cnn_model, X_test, y_test)
            print("CNN acc:", acc)
            print(classification_report(true, preds, target_names=list(LABELS.keys())))
        return

    if args.run_bot:
        if args.telegram_token is None:
            print("Telegram token required: --telegram-token <TOKEN>")
            return
        run_telegram_bot(args.telegram_token)
        return

    parser.print_help()
    print("\nExamples:")
    print("  python face_mask_pipeline.py --train-all")
    print("  python face_mask_pipeline.py --classify-all")
    print("  python face_mask_pipeline.py --run-bot --telegram-token YOUR_TOKEN")

if __name__ == "__main__":
    main()