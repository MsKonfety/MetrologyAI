from PyQt5.QtWidgets import QVBoxLayout, QWidget, QHBoxLayout
import pyqtgraph as pg
from datetime import datetime


class CalculationsTab(QWidget):
    def __init__(self):
        super().__init__()

        self.time_data = []
        self.area_data = []
        self.center_x_data = []
        self.center_y_data = []

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

        self.area_plot_widget.setBackground('#1e1e1e')

        self.area_plot_widget.setLabel('left', 'Площадь', 'кв.см')
        self.area_plot_widget.setLabel('bottom', 'Время', 'секунды')
        self.area_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.area_plot_widget.setTitle("Изменение площади объекта", color='white', size='12pt')
        
        self.area_plot_widget.setMouseEnabled(x=True, y=False)
        self.area_plot_widget.setLimits(xMin=0, yMin=0) 

        # Линия для точек
        self.area_plot_curve = self.area_plot_widget.plot(pen=pg.mkPen(color='#4CAF50', width=2))

    def setup_center_graph(self):
        """Настройка графика положения центра"""
        self.center_plot_widget = pg.PlotWidget()
        self.center_plot_widget.setBackground('#1e1e1e')
        self.center_plot_widget.setLabel('left', 'Y координата', 'пиксели')
        self.center_plot_widget.setLabel('bottom', 'X координата', 'пиксели')
        self.center_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.center_plot_widget.setTitle("Положение центра объекта", color='white', size='12pt')

        self.center_plot_widget.setMouseEnabled(x=False, y=False)
        self.center_plot_widget.setLimits(xMin=0, yMin=0)
        
        # Начальные настройки осей (будут обновляться при получении данных)
        self.update_center_axes()
        
        # График для положения центра
        self.center_line = self.center_plot_widget.plot(
            pen=pg.mkPen(color='#4CAF50', width=1, style=pg.QtCore.Qt.DashLine)
        )
        
        # Точечный график для положения центра
        self.center_scatter = pg.ScatterPlotItem(
            size=8, 
            pen=pg.mkPen(color="#4CAF50", width=1), 
            brush=pg.mkBrush(0, 255, 0, 180),
            symbol='o'
        )
        self.center_plot_widget.addItem(self.center_scatter)
    
    def update_center_axes(self):
        """Обновление осей графика положения центра в соответствии с размером кадра"""
        if hasattr(self, 'frame_width') and hasattr(self, 'frame_height'):
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
        
    def update_graph(self, area, center_x, center_y, timestamp, frame_width=None, frame_height=None):
        """Обновление графиков"""
        # Обновляем размеры кадра если они предоставлены
        if frame_width is not None and frame_height is not None:
            self.update_frame_size(frame_width, frame_height)
            
        if area is not None and area > 0 and center_x is not None and center_y is not None:
            current_time = timestamp.timestamp()
            
            # Обновляем данные только раз в секунду
            if self.last_update_time is None or (current_time - self.last_update_time) >= 1.0:
                if self.time_data:
                    time_offset = (current_time - self.start_time)
                else:
                    time_offset = 0
                
                # Добавляем данные
                self.time_data.append(time_offset)
                self.area_data.append(area)
                self.center_x_data.append(center_x)
                self.center_y_data.append(center_y)
                self.last_update_time = current_time

                # Обновляем график площади
                self.area_plot_curve.setData(self.time_data, self.area_data)
                
                # Обновляем график положения центра
                if len(self.center_x_data) > 0 and len(self.center_y_data) > 0:
                    # данные для линии
                    self.center_line.setData(self.center_x_data, self.center_y_data)
                    # данные для точек
                    self.center_scatter.setData(self.center_x_data, self.center_y_data)

                # Автомасштабирование графика площади (только если данных достаточно)
                if len(self.time_data) > 1:
                    self.area_plot_widget.setXRange(0, max(self.time_data) * 1.1)
                    self.area_plot_widget.setYRange(0, max(self.area_data) * 1.1)
