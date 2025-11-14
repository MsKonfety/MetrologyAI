from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QFileDialog, QMessageBox
)
from PyQt5.QtCore import QTimer
from datetime import datetime
import json
import cv2
from ultralytics import YOLO
import numpy as np

from visualization_tab import VisualizationTab
from statistic_tab import StatisticTab
from calculations_tab import CalculationsTab

import time
import os

import torch
import torchvision.transforms as transforms
import torchvision.models as models


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

            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )

            # Загружаем чекпоинт для получения информации о количестве классов
            checkpoint = torch.load(self.model_path, map_location=self.device)

            num_classes = None

            # Ищем ключи с весами выходного слоя
            fc_keys = ['fc.weight', 'classifier.weight', 'head.weight']
            for key in fc_keys:
                if key in checkpoint.get('model_state_dict', checkpoint):
                    if 'model_state_dict' in checkpoint:
                        num_classes = checkpoint['model_state_dict'][key].shape[0]
                    else:
                        num_classes = checkpoint[key].shape[0]
                    break

            # Если не нашли в state_dict, ищем в корне checkpoint
            if num_classes is None:
                for key in fc_keys:
                    if key in checkpoint:
                        num_classes = checkpoint[key].shape[0]
                        break

            # Если все еще не определили, используем количество class_names
            if num_classes is None:
                num_classes = len(self.class_names)

            # Создаем архитектуру ResNet18
            model = models.resnet18(weights=None)
            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, num_classes)

            # Загружаем веса с обработкой несоответствий
            state_dict = None
            possible_keys = ['model_state_dict', 'state_dict', 'model', 'net']

            for key in possible_keys:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break

            if state_dict is None:
                print("Пробуем загрузить checkpoint как state_dict")
                state_dict = checkpoint

            # Удаляем проблемные ключи, если размеры не совпадают
            model_state_dict = model.state_dict()

            # Фильтруем state_dict для загрузки только подходящих параметров
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_state_dict:
                    if model_state_dict[k].shape == v.shape:
                        filtered_state_dict[k] = v

            # Загружаем отфильтрованные веса
            model_state_dict.update(filtered_state_dict)
            model.load_state_dict(model_state_dict, strict=False)

            model.to(self.device)
            model.eval()

            self.model = model

            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            self.is_loaded = True
            print("Модель загружена успешно")
            print(f"Количество классов: {num_classes}")

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
                    return None, None, None

            results = self.model(frame, verbose=False)[0]

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
                # Извлекаем число из названия класса (например, "150cm" -> 150)
                distance_str = ''.join(filter(str.isdigit, predicted_class))
                distance = int(distance_str) if distance_str else 100
                print(
                    f"Предсказанное расстояние: {distance}см "
                    f"(уверенность: {confidence:.2%})"
                )
                return distance
            else:
                return 0

        except Exception as e:
            print(f"Ошибка при предсказании расстояния: {e}")


class MetrologyAI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Отключение вывода в терминал
        self.suppress_output()

        # Инициализация классификационной модели
        self.distance_class_names = [
            '110', '130', '150', '170', '190', '210', '230', '250',
            '50', '70', '90', 'MY_data'
        ]
        self.classification_model_path = (
            "C:\\Users\\astaf\\Desktop\\metrology\\"
            "phone_fast_created_merged_data_resnet18_multilabel.pth"
        )
        self.classification_model = ClassificationModel(
            self.classification_model_path,
            self.distance_class_names
        )

        # Загрузка классификационной модели
        self.classification_model.load_model()

        # Инициализация процессора детектирования
        self.model_path = "C:\\Users\\astaf\\Desktop\\metrology\\best.pt"
        self.detection_processor = DetectionProcessor(
            self.model_path, self.classification_model
        )

        self.initUI()

        # видео
        self.cap = None
        self.timer = QTimer()
        self.is_running = False

        # Подключение сигналов
        self.timer.timeout.connect(self.update_frame)

        self.last_graph_update = datetime.now()
        self.graph_update_interval = 1000  # 1 секунда
        self.frame_counter = 0

        self.detection_interval = 20
        self.frame_counter = 0

        self.last_bbox_change = time.time()
        self.bbox_change_interval = 1.0  # 1 секунда
        self.current_bbox = None
        self.current_distance = 100  # значение по умолчанию

        # переменные для выбора источника
        self.video_source = "camera"  # По умолчанию камера
        self.video_file_path = None

        # Проверка существования файлов моделей
        self.check_model_files()

    def suppress_output(self):
        """Отключение вывода в терминал для различных библиотек"""
        # Отключение вывода OpenCV
        cv2.setLogLevel(0)

        # Отключение вывода YOLO
        os.environ['ULTRALYTICS_VERBOSE'] = 'False'

    def check_model_files(self):
        """Проверка существования файлов моделей"""
        if not os.path.exists(self.model_path):
            print("Файл модели детектирования не найден")
        else:
            print(f"Файл модели детектирования: {self.model_path}")

        if not os.path.exists(self.classification_model_path):
            print("Файл классификационной модели не найден")
        else:
            print(
                f"Файл классификационной модели: "
                f"{self.classification_model_path}"
            )

    def initUI(self):
        self.setWindowTitle("Metrology AI")
        self.setGeometry(100, 100, 1200, 800)

        # виджет вкладок
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Создание вкладок
        self.visualization_tab = VisualizationTab()
        self.statistic_tab = StatisticTab()
        self.calculations_tab = CalculationsTab()

        # Подключение сигналов
        self.visualization_tab.start_button.clicked.connect(self.start_detection)
        self.visualization_tab.stop_button.clicked.connect(self.stop_detection)
        self.visualization_tab.select_camera_button.clicked.connect(
            self.select_camera
        )
        self.visualization_tab.select_file_button.clicked.connect(
            self.select_video_file
        )

        # Добавление вкладок
        self.tabs.addTab(self.visualization_tab, "Визуализация")
        self.tabs.addTab(self.statistic_tab, "Относительная статистика")
        self.tabs.addTab(self.calculations_tab, "Результаты расчетов")

        # Отображение статуса классификационной модели
        self.add_to_log("Система инициализирована.")
        self.add_to_log("Выберите источник видео и нажмите 'Старт'.")

    def get_frame_size(self):
        """Получение размера кадра видео"""
        if self.cap and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return 600, 400  # размеры по умолчанию

    def select_camera(self):
        """Выбор камеры в качестве источника"""
        self.video_source = "camera"
        self.video_file_path = None
        self.visualization_tab.update_source_info("Камера", "Готов к запуску")
        self.add_to_log("Выбран источник: Камера")

    def select_video_file(self):
        """Выбор видеофайла в качестве источника"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите видеофайл",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)"
        )

        if file_path:
            self.video_source = "file"
            self.video_file_path = file_path
            file_name = os.path.basename(file_path)
            self.visualization_tab.update_source_info("Файл", file_name)
            self.add_to_log(f"Выбран источник: Видеофайл ({file_name})")

    def start_detection(self):
        """Запуск захвата видео и детектирования"""
        try:
            # Инициализация видеозахвата в зависимости от выбранного источника
            if self.video_source == "camera":
                self.cap = cv2.VideoCapture(0)
                source_name = "камера"
            else:  # video file
                if not self.video_file_path:
                    QMessageBox.warning(
                        self, "Ошибка", "Сначала выберите видеофайл"
                    )
                    return
                self.cap = cv2.VideoCapture(self.video_file_path)
                source_name = os.path.basename(self.video_file_path)

            if not self.cap.isOpened():
                self.add_to_log(f"Не удалось открыть видео: {source_name}")
                return

            self.is_running = True
            self.visualization_tab.start_button.setEnabled(False)
            self.visualization_tab.stop_button.setEnabled(True)
            self.visualization_tab.select_camera_button.setEnabled(False)
            self.visualization_tab.select_file_button.setEnabled(False)

            # Запуск таймера для обновления кадров
            self.timer.start(30)

            self.add_to_log(f"Видеопоток запущен ({source_name})")

        except Exception as e:
            self.add_to_log(f"Ошибка при запуске: {str(e)}")

    def stop_detection(self):
        """Остановка детектирования"""
        self.is_running = False
        self.timer.stop()

        if self.cap:
            self.cap.release()
            self.cap = None

        self.visualization_tab.start_button.setEnabled(True)
        self.visualization_tab.stop_button.setEnabled(False)
        self.visualization_tab.select_camera_button.setEnabled(True)
        self.visualization_tab.select_file_button.setEnabled(True)

        # Очистка текущих данных детектирования
        self.current_bbox = None
        self.current_distance = 100

        # Обновление интерфейса
        self.visualization_tab.video_label.clear()
        self.visualization_tab.video_label.setText("Видеопоток остановлен")
        self.visualization_tab.video_label.setStyleSheet(
            "border: 2px solid gray; background-color: #1e1e1e; color: white;"
        )

        # Очистка информации о детектировании
        self.visualization_tab.info_text.clear()

        self.add_to_log("Видеопоток остановлен")
        self.save_to_json(
            {"status": "stopped", "timestamp": datetime.now().isoformat()}
        )

    def update_frame(self):
        """Обновление кадра видео"""
        if not self.is_running or not self.cap:
            return

        ret, frame = self.cap.read()

        # перезапуск видеофайла
        if not ret and self.video_source == "file":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Перемотка в начало
            ret, frame = self.cap.read()
            if ret:
                self.add_to_log("Видеофайл перезапущен")

        if ret:
            self.frame_counter += 1

            # Получаем размеры кадра для графика
            frame_width, frame_height = self.get_frame_size()

            # Детектировать только каждый N-й кадр
            if self.frame_counter % self.detection_interval == 0:
                # Выполняем детектирование в основном потоке
                processed_frame, bbox, distance = (
                    self.detection_processor.process_detection(frame)
                )

                # Обновляем текущий bbox и расстояние
                if bbox is not None:
                    self.current_bbox = bbox
                    self.current_distance = distance

                # Обрабатываем кадр с результатами детектирования
                processed_frame, detection_info = self.process_frame(
                    frame, bbox, distance
                )

                # Обновление информации
                self.visualization_tab.update_detection_info(detection_info)
                self.save_detection_result(detection_info)

                # Обновление графика
                if detection_info["bbox"] is not None:
                    current_time = datetime.now()
                    x, y, w, h = detection_info["bbox"]
                    area = w * h
                    center_x = x + w // 2
                    center_y = y + h // 2

                    frame_width, frame_height = self.get_frame_size()
                    self.calculations_tab.update_graph(
                        area, center_x, center_y, current_time,
                        frame_width, frame_height
                    )

                # Отображаем обработанный кадр
                self.visualization_tab.display_frame(processed_frame)
            else:
                # Используем последний известный bbox для отрисовки
                processed_frame, _ = self.process_frame(frame)
                self.visualization_tab.display_frame(processed_frame)

    def process_frame(self, frame, bbox=None, distance=None):
        """Обработка кадра и отрисовка результатов"""
        # копия кадра для рисования
        processed_frame = frame.copy()

        # Используем переданные bbox и distance или последние известные
        if bbox is None:
            bbox = self.current_bbox
        if distance is None:
            distance = self.current_distance

        # рисуем прямоугольник
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)

            # Отображение центра
            center = x + w // 2, y + h // 2
            cv2.circle(processed_frame, center, 3, (0, 255, 0), -1)

            # Отображение информации на кадре
            if distance is not None:
                cv2.putText(
                    processed_frame, f"distance: {distance}cm",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )
            else:
                cv2.putText(
                    processed_frame, "distance: вычисляется...",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )

        detection_info = {
            "bbox": bbox,
            "distance": distance,
            "timestamp": datetime.now().isoformat(),
            "frame_size": frame.shape
        }

        return processed_frame, detection_info

    def save_detection_result(self, detection_info):
        """Сохранение результата детектирования"""
        if detection_info["bbox"] is not None:
            x, y, w, h = detection_info["bbox"]
            area = w * h
            if detection_info['distance'] is not None:
                distance_text = f"{detection_info['distance']}cm"
            else:
                distance_text = "неизвестно"
            log_entry = (
                f"[{detection_info['timestamp']}] "
                f"Объект: pos=({x},{y}) size={w}x{h} "
                f"area={area} px "
                f"distance={distance_text}\n"
            )
        else:
            log_entry = f"[{detection_info['timestamp']}] Объект не найден\n"

        # Добавление в текстовое поле лога
        self.add_to_log(log_entry.strip())

        # сохранение в JSON
        self.save_to_json(detection_info)

    def save_to_json(self, data):
        """Сохранение данных в JSON файл"""
        try:
            json_file = "detection_log.json"

            # Преобразование NumPy типов в стандартные Python типы
            data = self.convert_numpy_types(data)

            # Чтение существующих данных
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = {"detections": []}

            # Добавление новых данных
            if "detections" in existing_data:
                existing_data["detections"].append(data)
            else:
                existing_data["detections"] = [data]

            # Сохранение
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.add_to_log(f"Ошибка сохранения в JSON: {str(e)}")

    def convert_numpy_types(self, obj):
        """преобразование NumPy типов в стандартные Python типы"""
        if isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key,
                    value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_numpy_types(item) for item in obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def add_to_log(self, message):
        """Добавление сообщения в лог-панель"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.visualization_tab.add_to_log(f"[{timestamp}] {message}")

    def closeEvent(self, event):
        """Обработка закрытия приложения"""
        self.stop_detection()
        event.accept()


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = MetrologyAI()
    window.show()
    sys.exit(app.exec_())
