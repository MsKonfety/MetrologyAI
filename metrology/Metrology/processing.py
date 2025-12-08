from PyQt5.QtCore import QThread, pyqtSignal, QMutex, pyqtSlot
from datetime import datetime
import cv2


from tracker import MyTracker

import time


class ProcessingWorker(QThread):
    frame_processed = pyqtSignal(object, object, object)  # frame, bbox, distance
    detection_info_ready = pyqtSignal(object)

    def __init__(self, video_stream, detection_processor):
        super().__init__()
        self.video_stream = video_stream
        self.detection_processor = detection_processor
        self.tracker = None
        self.current_bbox = None
        self.current_distance = None
        self.stopped = False
        self.detection_interval = 200
        self.frame_counter = 0
        self.mutex = QMutex()

    def run(self):
        while not self.stopped:
            try:
                if not self.video_stream.more():
                    time.sleep(0.01)
                    continue

                frame = self.video_stream.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                self.frame_counter += 1
                is_detection_frame = self.frame_counter % self.detection_interval == 0

                # Обрабатываем кадр (детектирование + трекерирование в одном потоке)
                processed_frame, bbox, distance, detection_info = (
                    self.process_single_frame(frame, is_detection_frame)
                )

                # Отправляем результаты в основной поток
                if processed_frame is not None:
                    self.frame_processed.emit(processed_frame, bbox, distance)
                    self.detection_info_ready.emit(detection_info)

                # Небольшая пауза для снижения нагрузки на CPU
                time.sleep(0.001)

            except Exception as e:
                print(f"Ошибка в потоке обработки: {e}")
                time.sleep(0.01)

    def process_single_frame(self, frame, is_detection_frame):
        """Обработка одного кадра: детектирование и трекерирование в одном потоке"""
        try:
            bbox = None
            distance = None

            if is_detection_frame:
                _, new_bbox, new_distance = self.detection_processor.process_detection(
                    frame
                )

                if new_bbox is not None:
                    self.current_bbox = new_bbox
                    self.current_distance = new_distance

                    # Инициализируем или переинициализируем трекер
                    try:
                        self.tracker = MyTracker(frame, new_bbox)
                        bbox = new_bbox
                        distance = new_distance
                    except Exception as e:
                        print(f"Ошибка инициализации трекера: {e}")
                        self.tracker = None
                        bbox = None
                        distance = None
                else:
                    print("Объект не обнаружен при детектировании")
                    self.current_bbox = None
                    self.current_distance = None
                    self.tracker = None
                    bbox = None
                    distance = None

            else:
                # Трекерирование на каждом кадре (кроме детекционных)
                if self.tracker is not None and self.current_bbox is not None:
                    try:
                        # Используем трекер для отслеживания
                        tracked_bbox = self.tracker.update(frame)
                        self.current_bbox = tracked_bbox
                        bbox = tracked_bbox
                        distance = self.current_distance
                        # print(f"Трекинг: {bbox}")  # Можно раскомментировать для отладки
                    except Exception as e:
                        print(f"Ошибка трекера: {e}")
                        self.tracker = None
                        bbox = None
                        distance = None
                else:
                    # Если трекер не инициализирован, используем последние известные значения
                    bbox = self.current_bbox
                    distance = self.current_distance

            # Отрисовка результатов на кадре
            processed_frame = self.draw_detection_results(
                frame, bbox, distance, is_detection_frame
            )

            detection_info = {
                "bbox": bbox,
                "distance": distance,
                "timestamp": datetime.now(),
                "frame_size": frame.shape,
                "is_detection_frame": is_detection_frame,
                "frame_counter": self.frame_counter,
            }
            if is_detection_frame and bbox is not None:
                return processed_frame, bbox, distance, detection_info
            else:
                return processed_frame, None, None, None

        except Exception as e:
            print(f"Ошибка обработки кадра: {e}")
            return (
                frame,
                None,
                None,
                {
                    "bbox": None,
                    "distance": None,
                    "timestamp": datetime.now(),
                    "frame_size": frame.shape,
                    "is_detection_frame": is_detection_frame,
                    "frame_counter": self.frame_counter,
                },
            )

    def draw_detection_results(self, frame, bbox, distance, is_detection_frame):
        """Отрисовка детекции и информации на кадре"""
        try:
            processed_frame = frame.copy()

            if bbox is not None:
                x, y, w, h = bbox

                # Рисуем прямоугольник
                color = (0, 255, 0)

                thickness = 3 if is_detection_frame else 2
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, thickness)

                # Рисуем центр
                center = x + w // 2, y + h // 2
                cv2.circle(processed_frame, center, 4, color, -1)

                # Добавляем информацию о расстоянии
                if distance is not None:
                    cv2.putText(
                        processed_frame,
                        f"Distance: {distance}cm",
                        (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
            return processed_frame

        except Exception as e:
            print(f"Ошибка отрисовки: {e}")
            return frame

    def stop_processing(self):
        self.stopped = True
        self.wait(1000)
