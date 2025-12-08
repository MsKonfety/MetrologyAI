import numpy as np
import cv2


class MyTracker:
    def __init__(self, frame, init_box):

        x, y, w, h = init_box

        shrink_factor = 1
        roi_w = int(w * shrink_factor)
        roi_h = int(h * shrink_factor)
        roi_x = x + (w - roi_w) // 2
        roi_y = y + (h - roi_h) // 2

        frame_h, frame_w = frame.shape[:2]
        new_x = max(0, roi_x)
        new_y = max(0, roi_y)
        new_w = min(roi_x + roi_w, frame_w) - new_x
        new_h = min(roi_y + roi_h, frame_h) - new_y

        self.track_window = (new_x, new_y, new_w, new_h)

        roi = frame[new_y : new_y + new_h, new_x : new_x + new_w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv_roi, np.array((0.0, 60.0, 32.0)), np.array((180.0, 255.0, 255.0))
        )
        self.roi_hist = cv2.calcHist(
            [hsv_roi], [0, 1], mask, [180, 256], [0, 180, 0, 256]
        )
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        cent_x = new_x + new_w / 2
        cent_y = new_y + new_h / 2
        self.kalman.statePre = np.array([[cent_x], [cent_y], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[cent_x], [cent_y], [0], [0]], np.float32)

    def update(self, frame):
        prediction = self.kalman.predict()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], self.roi_hist, [0, 180, 0, 256], 1)
        ret, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)

        x_t, y_t, w_t, h_t = self.track_window
        measurement = np.array([[x_t + w_t / 2], [y_t + h_t / 2]], np.float32)

        corrt = self.kalman.correct(measurement)

        kalman_x, kalman_y = int(corrt[0]), int(corrt[1])
        x_ = kalman_x - w_t // 2
        y_ = kalman_y - h_t // 2

        return (x_, y_, w_t, h_t)
