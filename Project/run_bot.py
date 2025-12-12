import os
import json
import cv2
import numpy as np
import torch
import joblib
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from PIL import Image
import io
from skimage.feature import hog
import torch.nn as nn
from torchvision import transforms

print("ПРОВЕРКА МОДЕЛЕЙ ДЛЯ ТЕЛЕГРАМ БОТА")

MODEL_FILES = {
    "cnn": "model_cnn.pth",
    "svm": "model_svm_hog.pkl", 
    "knn": "model_knn_orb.pkl",
    "metrics": "model_metrics.json"
}

for name, file in MODEL_FILES.items():
    if os.path.exists(file):
        print(f"{name.upper()}: {file} - найден")
    else:
        print(f"{name.upper()}: {file} - не найден")

if os.path.exists("model_metrics.json"):
    with open("model_metrics.json", "r") as f:
        metrics = json.load(f)
    print("МЕТРИКИ МОДЕЛЕЙ:")
    for key, value in metrics.items():
        if 'accuracy' in key:
            print(f"  {key}: {value:.4f} ({value*100:.1f}%)")
    print(f"Лучшая модель: {metrics.get('best_model_name', 'unknown')}")
else:
    print("Файл метрик не найден")
    metrics = {}

IMG_SIZE = 224
LABELS = {0: "С МАСКОЙ", 1: "БЕЗ МАСКИ"}

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

def load_models():
    models = {}

    if os.path.exists("model_cnn.pth"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load("model_cnn.pth", map_location=device))
        model.eval()
        models["cnn"] = model
        print("CNN модель загружена")

    if os.path.exists("model_svm_hog.pkl"):
        models["svm"] = joblib.load("model_svm_hog.pkl")
        print("SVM модель загружена")

    if os.path.exists("model_knn_orb.pkl"):
        models["knn"] = joblib.load("model_knn_orb.pkl")
        print("KNN модель загружена")

    return models

MODELS = load_models()
print(f"Всего загружено моделей: {len(MODELS)}")

def process_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    return img_rgb

def get_hog_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                  cells_per_block=(2, 2), feature_vector=True)
    return hog_feat.reshape(1, -1)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📊 Метрики моделей", callback_data="metrics")],
        [InlineKeyboardButton("🤖 CNN модель", callback_data="cnn")],
        [InlineKeyboardButton("📊 SVM модель", callback_data="svm")],
        [InlineKeyboardButton("📈 KNN модель", callback_data="knn")],
        [InlineKeyboardButton("🏆 Лучшая модель", callback_data="best")],
        [InlineKeyboardButton("📸 Все модели", callback_data="all")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "👋 Я бот для определения масок!\n"
        "📸 Отправьте мне фото лица, и я скажу есть ли на нём маска.\n\n"
        "🤖 Используемые модели:\n"
        "• CNN (нейросеть)\n"
        "• SVM (метод опорных векторов)\n"
        "• KNN (метод ближайших соседей)\n\n"
        "Выберите действие:",
        reply_markup=reply_markup
    )

async def info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if metrics:
        msg = "📊 МЕТРИКИ МОДЕЛЕЙ:\n\n"
        for key, value in metrics.items():
            if 'accuracy' in key:
                model_name = key.replace('accuracy_', '').upper()
                msg += f"{model_name}: {value:.4f} ({value*100:.1f}%)\n"
        msg += f"\n🏆 Лучшая модель: {metrics.get('best_model_name', 'unknown').upper()}"
    else:
        msg = "❌ Метрики не найдены"

    await update.message.reply_text(msg)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = await photo_file.download_as_bytearray()

    # Сохраняем фото в контексте
    context.user_data["last_photo"] = photo_bytes

    keyboard = [
        [InlineKeyboardButton("🤖 CNN", callback_data="predict_cnn"),
         InlineKeyboardButton("📊 SVM", callback_data="predict_svm")],
        [InlineKeyboardButton("📈 KNN", callback_data="predict_knn"),
         InlineKeyboardButton("🏆 Лучшая", callback_data="predict_best")],
        [InlineKeyboardButton("📸 Все модели", callback_data="predict_all")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "🖼 Фото получено! Выберите модель для анализа:",
        reply_markup=reply_markup
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data

    if data == "metrics":
        if metrics:
            msg = "📊 МЕТРИКИ МОДЕЛЕЙ:\n\n"
            for key, value in metrics.items():
                if 'accuracy' in key:
                    model_name = key.replace('accuracy_', '').upper()
                    msg += f"{model_name}: {value:.4f} ({value*100:.1f}%)\n"
            msg += f"\n🏆 Лучшая модель: {metrics.get('best_model_name', 'unknown').upper()}"
            await query.edit_message_text(msg)

    elif data in ["cnn", "svm", "knn", "best", "all"]:
        model_name = data.upper()
        if data == "best":
            model_name = metrics.get("best_model_name", "unknown").upper()

        await query.edit_message_text(
            f"📸 Отправьте фото для анализа моделью {model_name}"
        )

    elif data.startswith("predict_"):
        model_type = data.replace("predict_", "")

        if "last_photo" not in context.user_data:
            await query.edit_message_text("❌ Фото не найдено")
            return

        photo_bytes = context.user_data["last_photo"]
        img = process_image(photo_bytes)

        if model_type == "all":
            results = []

            if "cnn" in MODELS:
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                device = next(MODELS["cnn"].parameters()).device
                with torch.no_grad():
                    input_tensor = transform(img).unsqueeze(0).to(device)
                    output = MODELS["cnn"](input_tensor)
                    probs = torch.softmax(output, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                results.append(f"🤖 CNN: {LABELS[pred]} (маска: {probs[0][0]:.3f}, без: {probs[0][1]:.3f})")

            if "svm" in MODELS:
                features = get_hog_features(img)
                pred = MODELS["svm"].predict(features)[0]
                probs = MODELS["svm"].predict_proba(features)[0]
                results.append(f"📊 SVM: {LABELS[pred]} (маска: {probs[0]:.3f}, без: {probs[1]:.3f})")

            if "knn" in MODELS:
                features = get_hog_features(img)
                pred = MODELS["knn"].predict(features)[0]
                probs = MODELS["knn"].predict_proba(features)[0]
                results.append(f"📈 KNN: {LABELS[pred]} (маска: {probs[0]:.3f}, без: {probs[1]:.3f})")

            if results:
                msg = "📊 РЕЗУЛЬТАТЫ ВСЕХ МОДЕЛЕЙ:\n\n" + "\n".join(results)
                best_model = metrics.get("best_model_name", "unknown").upper()
                msg += f"\n\n🏆 Лучшая модель: {best_model}"
            else:
                msg = "❌ Модели не загружены"

            await query.edit_message_text(msg)

        else:
            if model_type not in MODELS:
                await query.edit_message_text(f"❌ Модель {model_type.upper()} не загружена")
                return

            if model_type == "cnn":
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                device = next(MODELS["cnn"].parameters()).device
                with torch.no_grad():
                    input_tensor = transform(img).unsqueeze(0).to(device)
                    output = MODELS["cnn"](input_tensor)
                    probs = torch.softmax(output, dim=1)
                    pred = torch.argmax(probs, dim=1).item()

                msg = f"🤖 CNN МОДЕЛЬ:\n\n"
                msg += f"Результат: {LABELS[pred]}\n"
                msg += f"Вероятность маски: {probs[0][0]:.3f}\n"
                msg += f"Вероятность без маски: {probs[0][1]:.3f}\n"
                msg += f"\nТочность модели: {metrics.get('accuracy_cnn', 'N/A'):.4f}"

            elif model_type in ["svm", "knn"]:
                features = get_hog_features(img)
                model = MODELS[model_type]
                pred = model.predict(features)[0]
                probs = model.predict_proba(features)[0]

                model_name = model_type.upper()
                msg = f"{'📊 SVM' if model_type == 'svm' else '📈 KNN'} МОДЕЛЬ:\n\n"
                msg += f"Результат: {LABELS[pred]}\n"
                msg += f"Вероятность маски: {probs[0]:.3f}\n"
                msg += f"Вероятность без маски: {probs[1]:.3f}\n"
                msg += f"\nТочность модели: {metrics.get(f'accuracy_{model_type}', 'N/A'):.4f}"

            elif model_type == "best":
                best_model = metrics.get("best_model_name", "unknown")
                if best_model in MODELS:
                    if best_model == "cnn":
                        transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        device = next(MODELS["cnn"].parameters()).device
                        with torch.no_grad():
                            input_tensor = transform(img).unsqueeze(0).to(device)
                            output = MODELS["cnn"](input_tensor)
                            probs = torch.softmax(output, dim=1)
                            pred = torch.argmax(probs, dim=1).item()
                    else:
                        features = get_hog_features(img)
                        model = MODELS[best_model]
                        pred = model.predict(features)[0]
                        probs = model.predict_proba(features)[0]

                    msg = f"🏆 ЛУЧШАЯ МОДЕЛЬ ({best_model.upper()}):\n\n"
                    msg += f"Результат: {LABELS[pred]}\n"
                    msg += f"Вероятность маски: {probs[0] if isinstance(probs, np.ndarray) else float(probs[0]):.3f}\n"
                    msg += f"Вероятность без маски: {probs[1] if isinstance(probs, np.ndarray) else float(probs[1]):.3f}\n"
                    msg += f"\nТочность модели: {metrics.get(f'accuracy_{best_model}', 'N/A'):.4f}"
                else:
                    msg = f"❌ Лучшая модель {best_model.upper()} не загружена"

            await query.edit_message_text(msg)

def main():
    TOKEN = "8531592850:AAEJmozYY2YgeKObWTT6HcinZXXct7YaV8s"
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("info", info))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(CallbackQueryHandler(button_handler))

    print("🤖 БОТ ЗАПУЩЕН")
    print("1. Откройте Telegram")
    print("2. Найдите своего бота")
    print("3. Отправьте команду /start")

    app.run_polling()

if __name__ == "__main__":
    main()
