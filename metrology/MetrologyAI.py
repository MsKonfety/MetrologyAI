import sys
import json
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QWidget, QTextEdit)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap


class MetrologyAI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        # видео
        self.cap = None
        self.timer = QTimer()
        self.is_running = False
        
        # Подключение сигналов
        self.timer.timeout.connect(self.update_frame)
        
    def initUI(self):
        self.setWindowTitle("Metrology AI")
        self.setGeometry(100, 100, 1200, 800)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Левая часть - видео и кнопки
        left_layout = QVBoxLayout()
        
        # Метка для видео
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Видеопоток не запущен")
        self.video_label.setStyleSheet("border: 2px solid gray; background-color: #f0f0f0;")
        left_layout.addWidget(self.video_label)
        
        # кнопки
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Старт")
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 14px; }")
        self.start_button.clicked.connect(self.start_detection)
        
        self.stop_button = QPushButton("Стоп")
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-size: 14px; }")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()
        
        left_layout.addLayout(button_layout)
        
        # лог и информация
        right_layout = QVBoxLayout()
        
        # Информация о детектировании
        info_label = QLabel("Информация о детектировании:")
        info_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_layout.addWidget(info_label)
        
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(200)
        self.info_text.setReadOnly(True)
        right_layout.addWidget(self.info_text)
        
        # Лог операцийй
        log_label = QLabel("Лог операций:")
        log_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.log_text)
        
        # Добавление layout'ов в основной
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)
        
        self.add_to_log("Система инициализирована. Нажмите 'Старт' для начала работы.")
        
    def start_detection(self):
        """Запуск захвата видео и детектирования"""
        try:
            # Инициализация видеозахвата
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                self.add_to_log("Не удалось открыть камеру")
                return
            
            self.is_running = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            
            # Запуск таймера для обновления кадров
            self.timer.start(30) 
            
            self.add_to_log("Видеопоток запущен")
            
        except Exception as e:
            self.add_to_log(f"Ошибка при запуске: {str(e)}")
    
    def stop_detection(self):
        """Остановка детектирования"""
        self.is_running = False
        self.timer.stop()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        self.video_label.setText("Видеопоток остановлен")
        self.video_label.setStyleSheet("border: 2px solid gray; background-color: #f0f0f0;")
        
        self.add_to_log("Видеопоток остановлен")
        self.save_to_json({"status": "stopped", "timestamp": datetime.now().isoformat()})
    
    def update_frame(self):
        """Обновление кадра видео"""
        if not self.is_running or not self.cap:
            return
        
        ret, frame = self.cap.read()
        if ret:
            processed_frame, detection_info = self.process_frame(frame)
    
            # Отображение обработанного кадра
            self.display_frame(processed_frame)
            
            # Обновление информации
            self.update_detection_info(detection_info)
            
            # Сохранение в лог
            self.save_detection_result(detection_info)
    
    def process_frame(self, frame):
        """Обработка кадра и детектирование объектов"""
        # копия кадра для рисования
        processed_frame = frame.copy()
        
        # Заглушка
        bbox, distance = self.detector(frame)
        
        # Отрисовка прямоугольника
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            
            # Отображение центра
            center_x, center_y = x + w // 2, y + h // 2
            cv2.circle(processed_frame, (center_x, center_y), 3, (0, 255, 0), -1)
        
        detection_info = {
            "bbox": bbox,
            "distance": distance,
            "timestamp": datetime.now().isoformat(),
            "frame_size": frame.shape
        }
        
        return processed_frame, detection_info
    
    def detector(self, frame):
        """Заглушка детектора объектов."""
        # прямоугольник в центре кадра
        h, w = frame.shape[:2]
        rect_w, rect_h = 200, 150
        x = (w - rect_w) // 2
        y = (h - rect_h) // 2
        
        # расстояние
        distance = 2.5  
        
        return (x, y, rect_w, rect_h), distance
    
    def display_frame(self, frame):
        """Отображение кадра в QLabel"""
        # Конвертация BGR (OpenCV) в RGB (Qt)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Создание QImage
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Масштабирование изображения под размер метки
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.width(), 
                                    self.video_label.height(), 
                                    Qt.KeepAspectRatio,
                                    Qt.SmoothTransformation)
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def update_detection_info(self, detection_info):
        """Обновление информации о детектировании"""
        bbox = detection_info["bbox"]
        distance = detection_info["distance"]
        
        if bbox is not None:
            x, y, w, h = bbox
            info_text = f"""Объект обнаружен:
- Координаты: ({x}, {y})
- Размер: {w}x{h}
- Центр: ({x + w//2}, {y + h//2})
- Расстояние: {distance:.2f} м
- Время: {datetime.now().strftime('%H:%M:%S')}"""
        else:
            info_text = "Объекты не обнаружены"
        
        self.info_text.setText(info_text)
    
    def save_detection_result(self, detection_info):
        """Сохранение результата детектирования"""
        if detection_info["bbox"] is not None:
            x, y, w, h = detection_info["bbox"]
            log_entry = (f"[{detection_info['timestamp']}] "
                        f"Объект: pos=({x},{y}) size={w}x{h} "
                        f"distance={detection_info['distance']:.2f}m\n")
        else:
            log_entry = f"[{detection_info['timestamp']}] Объекты не обнаружены\n"
            
        
        # Добавление в текстовое поле лога
        self.add_to_log(log_entry.strip())
        
        #сохранение в JSON
        self.save_to_json(detection_info)
    
    def save_to_json(self, data):
        """Сохранение данных в JSON файл"""
        try:
            json_file = "detection_log.json"
            
            # Чтениесуществующих данных
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
    
    def add_to_log(self, message):
        """Добавление сообщения в лог-панель"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # Автопрокрутка к последнему
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        """Обработка закрытия приложения"""
        self.stop_detection()
        event.accept()


app = QApplication(sys.argv)
    
    
app.setStyleSheet("""
    QMainWindow {
        background-color: #2b2b2b;
        color: white;
    }
    QPushButton {
        padding: 8px 16px;
        border: none;
        border-radius: 4px;
        font-size: 12px;
    }
    QTextEdit {
        background-color: #1e1e1e;
        color: #d4d4d4;
        border: 1px solid #404040;
        border-radius: 4px;
        font-family: 'Courier New';
    }
    QLabel {
        color: white;
    }
    """)
    
window = MetrologyAI()
window.show()
    
sys.exit(app.exec_())

