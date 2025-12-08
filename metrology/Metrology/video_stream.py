import cv2
import threading
import time
from collections import deque
import numpy as np


class VideoStream:
    def __init__(self, source=0, buffer_size=64):
        self.stream = cv2.VideoCapture(source)
        self.stopped = False
        self.buffer = deque(maxlen=1)  # Только последний кадр
        self.lock = threading.Lock()

        # Получаем размеры кадра
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.stream.get(cv2.CAP_PROP_FPS) or 50

    def start(self):
        self.stopped = False
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()

            if not grabbed:
                # Для видеофайлов - перезапускаем
                self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            with self.lock:
                if self.buffer:
                    self.buffer.pop()
                self.buffer.append(frame)

            # Контроль FPS
            time.sleep(1.0 / min(self.fps, 60))

    def read(self):
        with self.lock:
            if self.buffer:
                return self.buffer[-1].copy()  # Возвращаем копию самого свежего кадра
        return None

    def more(self):
        with self.lock:
            return len(self.buffer) > 0

    def stop(self):
        self.stopped = True
        time.sleep(0.1)  # Даем время потоку завершиться
        if self.stream.isOpened():
            self.stream.release()

    def get_frame_size(self):
        return self.width, self.height

    def get_fps(self):
        return self.fps