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
        
        # Настройка осей для отображения координат
        self.center_plot_widget.setXRange(0, 600)
        self.center_plot_widget.setYRange(0, 400)
        self.center_plot_widget.invertY(True)  # ось Y инвертирована

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
        
        
    def update_graph(self, area, center_x, center_y, timestamp):
        """Обновление графиков"""
        if area is not None and area > 0 and center_x is not None and center_y is not None:
            current_time = timestamp.timestamp()
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

                print(f"Добавлена точка: время={time_offset:.1f}с, площадь={area}, центр=({center_x},{center_y})")

                # Автомасштабирование графика площади
                if len(self.time_data) > 1:
                    self.area_plot_widget.setXRange(0, max(self.time_data) * 1.1)
                    self.area_plot_widget.setYRange(0, max(self.area_data) * 1.1)