from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel

class StatisticTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        """Настройка вкладки статистики"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        