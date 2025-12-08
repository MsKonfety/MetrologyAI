import sys
from PyQt5.QtWidgets import QApplication
from main_window import MetrologyAI


def main():
    app = QApplication(sys.argv)

    app.setStyleSheet(
        """
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
        QTabWidget::pane {
            border: 1px solid #404040;
            background-color: #2b2b2b;
        }
        QTabWidget::tab-bar {
            alignment: center;
        }
        QTabBar::tab {
            background-color: #404040;
            color: white;
            padding: 8px 16px;
            margin: 2px;
            border: none;
            border-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #4CAF50;
        }
        QTabBar::tab:hover {
            background-color: #555;
        }
        """
    )

    window = MetrologyAI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
