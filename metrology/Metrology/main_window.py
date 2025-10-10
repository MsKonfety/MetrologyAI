from PyQt5.QtWidgets import QMainWindow, QTabWidget
from PyQt5.QtCore import QTimer
from datetime import datetime
import json
import cv2

from visualization_tab import VisualizationTab
from statistic_tab import StatisticTab
from calculations_tab import CalculationsTab

import random
import time


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

        self.last_graph_update = datetime.now()
        self.graph_update_interval = 1000  #  1 секунда
        self.frame_counter = 0

        self.last_bbox_change = time.time()
        self.bbox_change_interval = 1.0  # 1 секунда
        self.current_bbox = None
        self.current_distance = None
        
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
        
        # Добавление вкладок
        self.tabs.addTab(self.visualization_tab, "Визуализация")
        self.tabs.addTab(self.statistic_tab, "Относительная статистика")
        self.tabs.addTab(self.calculations_tab, "Результаты расчетов")
        
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
            self.visualization_tab.start_button.setEnabled(False)
            self.visualization_tab.stop_button.setEnabled(True)
            
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
        
        self.visualization_tab.start_button.setEnabled(True)
        self.visualization_tab.stop_button.setEnabled(False)
        
        self.visualization_tab.video_label.setText("Видеопоток остановлен")
        self.visualization_tab.video_label.setStyleSheet("border: 2px solid gray; background-color: #f0f0f0;")
        
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
            self.visualization_tab.display_frame(processed_frame)
    
        # Обновление информации
            self.visualization_tab.update_detection_info(detection_info)
    
        # Сохранение в лог
            self.save_detection_result(detection_info)
    
            self.frame_counter += 1
    
        # обновление графика
            if detection_info["bbox"] is not None:
                current_time = datetime.now()
                x, y, w, h = detection_info["bbox"]
                area = w * h

                center_x = x + w // 2
                center_y = y + h // 2
            
                # Передаем все данные в график
                self.calculations_tab.update_graph(area, center_x, center_y, current_time)
            
            

    
    def process_frame(self, frame):
        """Обработка кадра и детектирование объектов"""
        # копия кадра для рисования
        processed_frame = frame.copy()
        
        # Заглушка
        bbox, distance = self.detector(frame)
        
        # Отрисовка прямоугольника
        if bbox is not None:
            x, y, w, h = bbox
            import cv2
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
        """Заглушка для детектора"""
        current_time = time.time()
        
        if (self.current_bbox is None or 
            current_time - self.last_bbox_change >= self.bbox_change_interval):
            
            h, w = frame.shape[:2]
            
            # Случайный размер прямоугольника
            rect_w = random.randint(100, 300)
            rect_h = random.randint(100, 300)
            
            # Случайное положение
            max_x = w - rect_w
            max_y = h - rect_h
            
            if max_x > 0 and max_y > 0:
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)

            else:
                # размещаем по центру, если прямоугольник выходит за границы
                x = max(0, (w - rect_w) // 2)
                y = max(0, (h - rect_h) // 2)
            
            # Случайное расстояние
            distance = random.uniform(1.0, 5.0)
            
            self.current_bbox = (x, y, rect_w, rect_h)
            self.current_distance = distance
            self.last_bbox_change = current_time
        
        return self.current_bbox, self.current_distance
    
    def save_detection_result(self, detection_info):
        """Сохранение результата детектирования"""
        if detection_info["bbox"] is not None:
            x, y, w, h = detection_info["bbox"]
            area = w * h
            log_entry = (f"[{detection_info['timestamp']}] "
                        f"Объект: pos=({x},{y}) size={w}x{h} "
                        f"area={area} px² "
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
    
    def add_to_log(self, message):
        """Добавление сообщения в лог-панель"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.visualization_tab.add_to_log(f"[{timestamp}] {message}")
    
    def closeEvent(self, event):
        """Обработка закрытия приложения"""
        self.stop_detection()
        event.accept()