import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import os
import numpy as np
from PIL import Image


class ResNetWithBBox(nn.Module):
    """ResNet модель с добавлением параметров bounding box к эмбеддингам"""

    def __init__(self, num_classes, use_pretrained=False, bbox_dim=4):
        super(ResNetWithBBox, self).__init__()

        # Загружаем ResNet
        if use_pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            self.resnet = models.resnet18(weights=weights)
        else:
            self.resnet = models.resnet18(weights=None)

        # Сохраняем размерность признаков
        self.original_fc_in_features = self.resnet.fc.in_features

        # Удаляем последний fc слой
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Новый fc слой с учетом bbox параметров
        self.bbox_dim = bbox_dim
        self.classifier = nn.Linear(
            self.original_fc_in_features + bbox_dim, num_classes
        )

        # Инициализация весов классификатора
        nn.init.normal_(self.classifier.weight, 0, 0.01)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x, bbox_params=None):
        features = self.resnet(x)
        features = features.view(features.size(0), -1)

        if bbox_params is None:
            batch_size = features.size(0)
            bbox_params = torch.zeros(batch_size, self.bbox_dim).to(features.device)

        combined_features = torch.cat([features, bbox_params], dim=1)
        output = self.classifier(combined_features)

        return output


def prepare_bbox_params(bbox_tensor, img_size=(224, 224)):
    """Подготавливает bbox параметры для модели"""
    if bbox_tensor.dim() == 1:
        bbox_tensor = bbox_tensor.unsqueeze(0)

    img_h, img_w = img_size
    bbox_normalized = bbox_tensor.clone().float()

    x1, y1, w, h = (
        bbox_normalized[:, 0],
        bbox_normalized[:, 1],
        bbox_normalized[:, 2],
        bbox_normalized[:, 3],
    )
    x_center = (x1 + w / 2) / img_w
    y_center = (y1 + h / 2) / img_h
    width_norm = w / img_w
    height_norm = h / img_h

    bbox_params = torch.stack([x_center, y_center, width_norm, height_norm], dim=1)
    bbox_params = torch.clamp(bbox_params, 0.0, 1.0)

    return bbox_params


class ClassificationModel:
    """Класс для работы с multi-label классификационной моделью с bbox параметрами"""

    def __init__(self, model_path, class_names, multi_label=True, threshold=0.5):
        self.model_path = model_path
        self.class_names = class_names
        self.multi_label = multi_label
        self.threshold = threshold
        self.model = None
        self.device = None
        self.transform = None
        self.is_loaded = False

    def load_model(self):
        """Загрузка multi-label классификационной модели"""
        try:
            if not os.path.exists(self.model_path):
                print(f"Файл модели не найден: {self.model_path}")
                return False

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(self.model_path, map_location=self.device)

            num_classes = len(self.class_names)
            self.model = ResNetWithBBox(num_classes, use_pretrained=False)

            # Загружаем веса
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            elif "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"])
            else:
                try:
                    self.model.load_state_dict(checkpoint)
                except:
                    print("Не удалось загрузить веса модели")
                    return False

            self.model.to(self.device)
            self.model.eval()

            if "multi_label" in checkpoint:
                self.multi_label = checkpoint["multi_label"]

            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            self.is_loaded = True
            return True

        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            self.is_loaded = False
            return False

    def predict(self, image, bbox=None):
        """Предсказание класса для изображения"""
        if not self.is_loaded or self.model is None:
            return None, 0.0

        try:
            # Конвертируем изображение
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                else:
                    pil_image = Image.fromarray(image)
            else:
                pil_image = image

            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            # Подготавливаем bbox параметры
            bbox_params = None
            if bbox is not None:
                try:
                    bbox_tensor = torch.tensor(bbox).float()
                    bbox_params = prepare_bbox_params(bbox_tensor).to(self.device)
                except:
                    bbox_params = None

            with torch.no_grad():
                if bbox_params is not None:
                    outputs = self.model(image_tensor, bbox_params)
                else:
                    outputs = self.model(image_tensor)

                if self.multi_label:
                    probabilities = torch.sigmoid(outputs)
                    predicted_probs = probabilities.cpu().numpy()[0]

                    predicted_labels = (predicted_probs > self.threshold).astype(int)
                    predicted_classes = [
                        self.class_names[j]
                        for j, pred in enumerate(predicted_labels)
                        if pred == 1
                    ]

                    if predicted_classes:
                        confidence = np.mean(
                            [
                                predicted_probs[j]
                                for j, pred in enumerate(predicted_labels)
                                if pred == 1
                            ]
                        )
                        return predicted_classes[0], float(confidence)
                    else:
                        max_prob_idx = np.argmax(predicted_probs)
                        max_prob = predicted_probs[max_prob_idx]
                        if max_prob > 0.1:
                            return self.class_names[max_prob_idx], float(max_prob)
                        else:
                            return "неизвестно", 0.0
                else:
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    predicted_prob, predicted_class = torch.max(probabilities, 1)

                    predicted_class_idx = predicted_class.item()
                    confidence = predicted_prob.item()

                    if predicted_class_idx < len(self.class_names):
                        return self.class_names[predicted_class_idx], confidence
                    else:
                        return f"class_{predicted_class_idx}", confidence

        except Exception as e:
            print(f"Ошибка при предсказании: {e}")
            return None, 0.0
