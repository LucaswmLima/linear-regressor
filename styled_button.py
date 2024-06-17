from PyQt5.QtWidgets import QPushButton

class StyledButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: #cc092f;
                color: white;
                font-size: 12pt;
                min-width: 100px;
                max-width: 150px;
                height: 40px;
                border-radius: 20px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #e60032;
            }
            QPushButton:pressed {
                background-color: #b30026;
            }
            QPushButton[disabled="true"] {
                background-color: #999999;
            }
        """)
