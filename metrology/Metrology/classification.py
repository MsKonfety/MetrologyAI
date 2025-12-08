import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import os


class ClassificationModel:
    """Класс для работы с классификационной моделью"""

    def __init__(self, model_path, class_names):
        self.model_path = model_path
        self.class_names = class_names
        self.model = None
        self.device = None
        self.transform = None
        self.is_loaded = False

    def load_model(self):
        """Загрузка классификационной модели"""
        try:
            if not os.path.exists(self.model_path):
                print(f"Файл модели не найден: {self.model_path}")
                return False

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Упрощенная загрузка модели
            model = models.resnet18(weights=None)
            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, len(self.class_names))

            # Пробуем загрузить веса
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
            except:
                print(
                    "Не удалось загрузить веса модели, используем случайные инициализации"
                )

            model.to(self.device)
            model.eval()

            self.model = model

            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            self.is_loaded = True
            print("Модель загружена успешно")
            return True

        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            self.is_loaded = False
            return False

    def predict(self, image):
        """Предсказание класса для изображения"""
        if not self.is_loaded or self.model is None:
            return None, 0.0

        try:
            # BGR в RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

            # Предсказание
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_prob, predicted_class = torch.max(probabilities, 1)

            predicted_class_idx = predicted_class.item()
            confidence = predicted_prob.item()

            # Безопасное получение имени класса
            if predicted_class_idx < len(self.class_names):
                predicted_class_name = self.class_names[predicted_class_idx]
            else:
                predicted_class_name = f"class_{predicted_class_idx}"

            return predicted_class_name, confidence

        except Exception as e:
            print(f"Ошибка при предсказании: {e}")
            return None, 0.0
