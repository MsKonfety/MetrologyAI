from ultralytics import YOLO
import numpy as np


class DetectionProcessor:
    """Класс для детектирования"""

    def __init__(self, model_path, classification_model):
        self.model_path = model_path
        self.classification_model = classification_model
        self.model = None
        self.last_bbox_size = None

    def load_model(self):
        """Загрузка модели детектирования"""
        try:
            self.model = YOLO(self.model_path, verbose=False)
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели детектирования: {str(e)}")
            return False

    def process_detection(self, frame):
        """Обработка детектирования на кадре"""
        try:
            if self.model is None:
                if not self.load_model():
                    return frame, None, None

            results = self.model(frame, verbose=False, conf=0.5)[0]

            if len(results.boxes) > 0:
                box = results.boxes.xywh.cpu().numpy().astype(np.int32)[0]
                x_center, y_center, w, h = box
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)

                self.last_bbox_size = w * h

                distance = self.predict_distance(frame)

                bbox = (x1, y1, w, h)
                return frame, bbox, distance
            else:
                return frame, None, None

        except Exception as e:
            print(f"Ошибка детектирования: {str(e)}")
            return frame, None, None

    def predict_distance(self, image):
        """Предсказание расстояния с помощью классификационной модели"""
        try:
            predicted_class, confidence = self.classification_model.predict(image)

            if predicted_class:
                distance_str = "".join(filter(str.isdigit, predicted_class))
                distance = int(distance_str) if distance_str else 0
                print(
                    f"Предсказано расстояние: {distance}см (уверенность: {confidence:.2%})"
                )
                return distance
            else:
                return 0

        except Exception as e:
            print(f"Ошибка при предсказании расстояния: {e}")
            return 0
