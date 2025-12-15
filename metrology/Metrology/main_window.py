from PyQt5.QtWidgets import QMainWindow, QTabWidget, QFileDialog
from PyQt5.QtCore import QTimer, pyqtSlot
import cv2
import json
import os
import time

from visualization_tab import VisualizationTab
from calculations_tab import CalculationsTab
from video_stream import VideoStream
from classification import ClassificationModel
from processing import ProcessingWorker
from detection import DetectionProcessor


class MetrologyAI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Отключение вывода в терминал
        self.suppress_output()

        # Путь к лог-файлу
        self.log_file_path = "detection_log.json"

        # Инициализация классификационной модели
        self.distance_class_names = [
            "110",
            "130",
            "150",
            "170",
            "190",
            "210",
            "230",
            "250",
            "50",
            "70",
            "90",
        ]

        self.classification_model_path = "best_resnet18_with_bbox_multilabel.pth"
        self.classification_model = ClassificationModel(
            self.classification_model_path, self.distance_class_names
        )

        # Загрузка классификационной модели
        self.classification_model.load_model()

        # Инициализация процессора детектирования
        self.model_path = "best.pt"
        self.detection_processor = DetectionProcessor(
            self.model_path, self.classification_model
        )

        # Инициализация видео потока и потока обработки
        self.video_stream = None
        self.processing_worker = None

        self.initUI()

        # Таймер для проверки состояния
        self.health_timer = QTimer()
        self.health_timer.timeout.connect(self.check_health)
        self.health_timer.start(5000)

        # переменные для выбора источника
        self.video_source = "camera"
        self.video_file_path = None

        # Проверка существования файлов моделей
        self.check_model_files()

        # Проверка существования лог-файла
        self.check_log_file()

    def suppress_output(self):
        """Отключение вывода в терминал"""
        cv2.setLogLevel(0)
        os.environ["ULTRALYTICS_VERBOSE"] = "False"

    def check_model_files(self):
        """Проверка существования файлов моделей"""
        if not os.path.exists(self.model_path):
            print("Файл модели детектирования не найден")
        if not os.path.exists(self.classification_model_path):
            print("Файл классификационной модели не найден")

    def check_log_file(self):
        """Проверка существования лог-файла, создание если отсутствует"""
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=4)

    def initUI(self):
        self.setWindowTitle("Metrology AI")
        self.setGeometry(100, 100, 1200, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Создание вкладок
        self.visualization_tab = VisualizationTab()
        self.calculations_tab = CalculationsTab()

        # Подключение сигналов
        self.visualization_tab.start_button.clicked.connect(self.start_detection)
        self.visualization_tab.stop_button.clicked.connect(self.stop_detection)
        self.visualization_tab.select_camera_button.clicked.connect(self.select_camera)
        self.visualization_tab.select_file_button.clicked.connect(
            self.select_video_file
        )

        # Добавление вкладок
        self.tabs.addTab(self.visualization_tab, "Визуализация")
        self.tabs.addTab(self.calculations_tab, "Результаты расчетов")

        self.add_to_log("Система инициализирована.")
        self.add_to_log("Выберите источник видео и нажмите 'Старт'.")

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
            "Video Files (*.mp4);;All Files (*)",
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
            # Останавливаем предыдущие потоки если они есть
            self.stop_detection()

            # Инициализация видеопотока
            source = 0 if self.video_source == "camera" else self.video_file_path
            self.video_stream = VideoStream(source=source)
            self.video_stream.start()

            # Ждем инициализации буфера
            for i in range(50):
                if self.video_stream.more():
                    break
                time.sleep(0.1)
            else:
                raise Exception("Не удалось инициализировать видеопоток")

            # Создаем и запускаем поток обработки
            self.processing_worker = ProcessingWorker(
                self.video_stream, self.detection_processor
            )
            self.processing_worker.frame_processed.connect(self.on_frame_processed)
            self.processing_worker.detection_info_ready.connect(
                self.on_detection_info_ready
            )
            self.processing_worker.start()

            self.visualization_tab.start_button.setEnabled(False)
            self.visualization_tab.stop_button.setEnabled(True)
            self.visualization_tab.select_camera_button.setEnabled(False)
            self.visualization_tab.select_file_button.setEnabled(False)

            # Обновляем размеры кадра для графика
            width, height = self.video_stream.get_frame_size()
            self.calculations_tab.update_frame_size(width, height)

            source_name = (
                "камера"
                if self.video_source == "camera"
                else os.path.basename(self.video_file_path)
            )
            self.add_to_log(f"Видеопоток запущен ({source_name})")

        except Exception as e:
            self.add_to_log(f"Ошибка при запуске: {str(e)}")
            self.stop_detection()

    def stop_detection(self):
        """Остановка детектирования"""
        if self.processing_worker:
            self.processing_worker.stop_processing()
            self.processing_worker = None

        if self.video_stream:
            self.video_stream.stop()
            self.video_stream = None

        self.visualization_tab.start_button.setEnabled(True)
        self.visualization_tab.stop_button.setEnabled(False)
        self.visualization_tab.select_camera_button.setEnabled(True)
        self.visualization_tab.select_file_button.setEnabled(True)

        # Обновление интерфейса
        self.visualization_tab.video_label.clear()
        self.visualization_tab.video_label.setText("Видеопоток остановлен")
        self.visualization_tab.video_label.setStyleSheet(
            "border: 2px solid gray; background-color: #1e1e1e; color: white;"
        )

        self.add_to_log("Видеопоток остановлен")

    @pyqtSlot(object, object, object)
    def on_frame_processed(self, frame, bbox, distance):
        """Обработка обработанного кадра из потока"""
        try:
            self.visualization_tab.display_frame(frame)

            # Обновление информации о детектировании
            detection_info = {"bbox": bbox, "distance": distance}
        except Exception as e:
            print(f"Ошибка отображения кадра: {e}")

    @pyqtSlot(object)
    def on_detection_info_ready(self, detection_info):
        """Обработка информации о детектировании из потока"""
        try:
            # Обновление графика только при обнаружении объекта
            if (
                detection_info is not None
                and detection_info["bbox"] is not None
                and detection_info["is_detection_frame"]
            ):
                self.save_detection_result(detection_info)

                current_time = detection_info["timestamp"]
                x, y, w, h = detection_info["bbox"]
                area = w * h
                center_x = x + w // 2
                center_y = y + h // 2
                self.visualization_tab.update_detection_info(
                    {
                        "bbox": detection_info["bbox"],
                        "distance": detection_info["distance"],
                    }
                )
                if self.video_stream:
                    frame_width, frame_height = self.video_stream.get_frame_size()
                else:
                    frame_width, frame_height = 640, 480

                self.calculations_tab.update_graph(
                    detection_info["distance"],
                    center_x,
                    center_y,
                    current_time,
                    frame_width,
                    frame_height,
                )
        except Exception as e:
            print(f"Ошибка обработки информации детектирования: {e}")

    def save_detection_result(self, detection_info):
        """Сохранение результата детектирования"""
        try:
            if (
                detection_info["is_detection_frame"]
                and detection_info["bbox"] is not None
            ):
                x, y, w, h = detection_info["bbox"]

                distance = detection_info.get("distance", "неизвестно")
                timestamp = detection_info["timestamp"].strftime("%H:%M:%S")

                log_entry = (
                    f"[{timestamp}] Объект: "
                    f"pos=({x},{y}) size={w}x{h} distance={distance}cm"
                )
                self.add_to_log(log_entry)

                # Сохранение в JSON файл
                self.save_to_json(detection_info)

            elif (
                detection_info["is_detection_frame"] and detection_info["bbox"] is None
            ):
                timestamp = detection_info["timestamp"].strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] Объект не обнаружен"
                self.add_to_log(log_entry)

        except Exception as e:
            print(f"Ошибка сохранения результата: {e}")

    def save_to_json(self, detection_info):
        """Сохранение данных детектирования в JSON файл"""
        try:
            # Загрузка существующих данных
            if os.path.exists(self.log_file_path):
                with open(self.log_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = []

            # Подготовка данных для сохранения
            x, y, w, h = detection_info["bbox"]
            log_entry = {
                "timestamp": detection_info["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "frame_counter": detection_info.get("frame_counter", 0),
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "distance": detection_info.get("distance", "неизвестно"),
                "center_x": int(x + w // 2),
                "center_y": int(y + h // 2),
                "video_source": self.video_source,
            }

            # Добавление записи
            data.append(log_entry)

            # Сохранение в файл
            with open(self.log_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Ошибка сохранения в JSON: {e}")

    def add_to_log(self, message):
        """Добавление сообщения в лог-панель"""
        try:
            self.visualization_tab.add_to_log(message)
        except:
            pass

    def check_health(self):
        """Проверка состояния системы"""
        pass

    def closeEvent(self, event):
        """Обработка закрытия приложения"""
        self.stop_detection()
        if self.health_timer.isActive():
            self.health_timer.stop()
        event.accept()


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = MetrologyAI()
    window.show()
    sys.exit(app.exec_())
