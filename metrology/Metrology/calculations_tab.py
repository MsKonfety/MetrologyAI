from PyQt5.QtWidgets import QVBoxLayout, QWidget, QHBoxLayout
import pyqtgraph as pg
from datetime import datetime
from collections import deque


def SlidingWindow(nums_list, win_size=5):
    sum = 0
    dq = deque()
    result = [0] * len(nums_list)
    for i in range(len(nums_list)):
        if len(dq) == win_size:
            sum -= dq.popleft()
        sum += nums_list[i]
        dq.append(nums_list[i])
        result[i] = sum / len(dq)
    return result


class CalculationsTab(QWidget):
    def __init__(self):
        super().__init__()

        self.time_data = []
        self.area_data = []
        self.center_x_data = []
        self.center_y_data = []

        # Добавляем сглаженные данные
        self.smoothed_area_data = []
        self.smoothed_center_x_data = []
        self.smoothed_center_y_data = []

        self.last_update_time = None
        self.frame_width = 600  # начальные размеры
        self.frame_height = 400

        self.setup_ui()
        self.start_time = datetime.now().timestamp()

    def setup_ui(self):
        """Настройка вкладки с графиками"""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Создаем layout для двух графиков
        graphs_layout = QHBoxLayout()
        main_layout.addLayout(graphs_layout)

        # График площади
        self.setup_area_graph()
        graphs_layout.addWidget(self.area_plot_widget)

        # График положения центра
        self.setup_center_graph()
        graphs_layout.addWidget(self.center_plot_widget)

    def setup_area_graph(self):
        """Настройка графика площади"""
        self.area_plot_widget = pg.PlotWidget()

        self.area_plot_widget.setBackground("#1e1e1e")

        self.area_plot_widget.setLabel("left", "Площадь", "кв.см")
        self.area_plot_widget.setLabel("bottom", "Время", "секунды")
        self.area_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.area_plot_widget.setTitle(
            "Изменение площади объекта", color="white", size="12pt"
        )

        self.area_plot_widget.setMouseEnabled(x=True, y=False)
        self.area_plot_widget.setLimits(xMin=0, yMin=0)

        # Линия для точек
        self.area_plot_curve = self.area_plot_widget.plot(
            pen=pg.mkPen(color="#4CAF50", width=2)
        )

    def setup_center_graph(self):
        """Настройка графика положения центра"""
        self.center_plot_widget = pg.PlotWidget()
        self.center_plot_widget.setBackground("#1e1e1e")
        self.center_plot_widget.setLabel("left", "Y координата", "пиксели")
        self.center_plot_widget.setLabel("bottom", "X координата", "пиксели")
        self.center_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.center_plot_widget.setTitle(
            "Положение центра объекта", color="white", size="12pt"
        )

        self.center_plot_widget.setMouseEnabled(x=False, y=False)
        self.center_plot_widget.setLimits(xMin=0, yMin=0)

        # Начальные настройки осей (будут обновляться при получении данных)
        self.update_center_axes()

        # График для положения центра
        self.center_line = self.center_plot_widget.plot(
            pen=pg.mkPen(color="#4CAF50", width=1, style=pg.QtCore.Qt.DashLine)
        )

        # Точечный график для положения центра
        self.center_scatter = pg.ScatterPlotItem(
            size=8,
            pen=pg.mkPen(color="#4CAF50", width=1),
            brush=pg.mkBrush(0, 255, 0, 180),
            symbol="o",
        )
        self.center_plot_widget.addItem(self.center_scatter)

    def update_center_axes(self):
        """Обновление осей графика положения центра"""
        if hasattr(self, "frame_width") and hasattr(self, "frame_height"):
            # Устанавливаем диапазоны осей с небольшим запасом
            x_margin = self.frame_width * 0.05
            y_margin = self.frame_height * 0.05

            self.center_plot_widget.setXRange(-x_margin, self.frame_width + x_margin)
            self.center_plot_widget.setYRange(-y_margin, self.frame_height + y_margin)
            self.center_plot_widget.invertY(True)  # ось Y инвертирована

    def update_frame_size(self, width, height):
        """Обновление размера кадра для адаптации графика"""
        self.frame_width = width
        self.frame_height = height
        self.update_center_axes()

    def update_graph(
        self, area, center_x, center_y, timestamp, frame_width=None, frame_height=None
    ):
        """Обновление графиков - упрощенная версия"""
        # Обновляем размеры кадра если они предоставлены
        if frame_width is not None and frame_height is not None:
            self.update_frame_size(frame_width, frame_height)

        # Добавляем данные только если все значения валидны
        if (
            area is not None
            and area > 0
            and center_x is not None
            and center_y is not None
        ):
            current_time = timestamp.timestamp()

            if self.time_data:
                time_offset = current_time - self.start_time
            else:
                time_offset = 0

            # Добавляем данные без ограничения по времени
            self.time_data.append(time_offset)
            self.area_data.append(area)
            self.center_x_data.append(center_x)
            self.center_y_data.append(center_y)

            # Применяем скользящее окно к данным
            if len(self.area_data) >= 3:  # Минимум 3 точки для сглаживания
                self.smoothed_area_data = SlidingWindow(self.area_data, 5)
                self.smoothed_center_x_data = SlidingWindow(self.center_x_data, 5)
                self.smoothed_center_y_data = SlidingWindow(self.center_y_data, 5)
            else:
                # Если точек недостаточно, используем исходные данные
                self.smoothed_area_data = self.area_data.copy()
                self.smoothed_center_x_data = self.center_x_data.copy()
                self.smoothed_center_y_data = self.center_y_data.copy()

            # Обновляем график площади (используем сглаженные данные)
            self.area_plot_curve.setData(self.time_data, self.smoothed_area_data)

            # Обновляем график положения центра (используем сглаженные данные)
            if (
                len(self.smoothed_center_x_data) > 0
                and len(self.smoothed_center_y_data) > 0
            ):
                self.center_line.setData(
                    self.smoothed_center_x_data, self.smoothed_center_y_data
                )
                self.center_scatter.setData(
                    self.smoothed_center_x_data, self.smoothed_center_y_data
                )

            # Автомасштабирование только если данных достаточно
            if len(self.time_data) > 1:
                max_time = max(self.time_data)
                max_area = max(self.area_data)
                if max_time > 0 and max_area > 0:
                    self.area_plot_widget.setXRange(0, max_time * 1.1)
                    self.area_plot_widget.setYRange(0, max_area * 1.1)
