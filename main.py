import sys
from PyQt5.QtWidgets import QApplication
from regression_app import RegressionApp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = RegressionApp()
    ex.center_window()
    ex.show()
    sys.exit(app.exec_())
