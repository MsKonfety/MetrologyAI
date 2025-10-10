from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QWidget, QTextEdit)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2


class VisualizationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Настройка главной вкладки"""
        # основной layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Левая часть - видео и кнопки
        left_layout = QVBoxLayout()
        
        # Метка для видео
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Видеопоток не запущен")
        self.video_label.setStyleSheet("border: 2px solid gray; background-color: #1e1e1e;")
        left_layout.addWidget(self.video_label)
        
        # кнопки
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Старт")
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 14px; }")
        
        self.stop_button = QPushButton("Стоп")
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-size: 14px; }")
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
    
    def display_frame(self, frame):
        """Отображение кадра в QLabel"""
        # Конвертация BGR (OpenCV) в RGB (Qt)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Создание QImage
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Масштабирование изображения
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
            from datetime import datetime
            info_text = f"""Объект обнаружен:
- Координаты: ({x}, {y})
- Размер: {w}x{h}
- Центр: ({x + w//2}, {y + h//2})
- Расстояние: {distance:.2f} м
- Время: {datetime.now().strftime('%H:%M:%S')}"""
        else:
            info_text = "Объекты не обнаружены"
        
        self.info_text.setText(info_text)
    
    def add_to_log(self, message):
        """Добавление сообщения в лог-панель"""
        self.log_text.append(message)
        
        # Автопрокрутка к последнему
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())