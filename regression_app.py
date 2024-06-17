from PyQt5.QtWidgets import QWidget, QLabel, QComboBox, QPushButton, QVBoxLayout, QMessageBox, QFileDialog, QLineEdit, QHBoxLayout, QGridLayout, QDesktopWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from styled_button import StyledButton
from help_dialog import HelpDialog
from regression_logic import RegressionLogic

class RegressionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.regression_logic = RegressionLogic()

    def initUI(self):
        self.setWindowTitle('Regressão Linear Simples')
        self.setFixedSize(600, 400)
        self.setStyleSheet("background-color: white;")
        self.setWindowIcon(QIcon('assets/icon.png'))

        layout = QGridLayout()
        layout.setSpacing(10)

        self.separator_label = QLabel('Separador:', self)
        layout.addWidget(self.separator_label, 0, 0)
        self.separator_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.separator_combobox = QComboBox(self)
        self.separator_combobox.addItems([',', ';', ':', '|', '/'])  # Opções de separadores
        layout.addWidget(self.separator_combobox, 0, 1)

        self.load_button = StyledButton('Carregar CSV', self)
        self.load_button.clicked.connect(self.load_csv)
        layout.addWidget(self.load_button, 0, 2)

        self.column_x_label = QLabel('Coluna de X:', self)
        layout.addWidget(self.column_x_label, 1, 0)
        self.column_x_combobox = QComboBox(self)
        self.column_x_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.column_x_combobox, 1, 1)

        self.column_y_label = QLabel('Coluna de Y:', self)
        layout.addWidget(self.column_y_label, 2, 0)
        layout.setColumnStretch(1, 1)
        self.column_y_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.column_y_combobox = QComboBox(self)
        layout.addWidget(self.column_y_combobox, 2, 1)

        self.correlation_label = QLabel('', self)
        layout.addWidget(self.correlation_label, 1, 2, Qt.AlignCenter)

        self.r_squared_label = QLabel('', self)
        layout.addWidget(self.r_squared_label, 2, 2, Qt.AlignCenter)

        self.run_regression_button = StyledButton('Rodar Regressão', self)
        self.run_regression_button.setEnabled(False)
        self.run_regression_button.clicked.connect(self.run_regression)
        layout.addWidget(self.run_regression_button, 5, 0)

        self.plot_regression_button = StyledButton('Mostrar Regressão', self)
        self.plot_regression_button.setEnabled(False)
        self.plot_regression_button.clicked.connect(self.plot_regression)
        layout.addWidget(self.plot_regression_button, 5, 1)

        self.plot_residuals_button = StyledButton('Mostrar Resíduos', self)
        self.plot_residuals_button.setEnabled(False)
        self.plot_residuals_button.clicked.connect(self.plot_residuals)
        layout.addWidget(self.plot_residuals_button, 5, 2)

        hbox = QHBoxLayout()
        hbox.setSpacing(10)

        self.input_label = QLabel('Valor de X:', self)
        self.input_label.setStyleSheet('QLabel {font-size: 12pt;}')
        hbox.addWidget(self.input_label)

        self.input_value = QLineEdit(self)
        self.input_value.setStyleSheet('QLineEdit {font-size: 12pt;}')
        hbox.addWidget(self.input_value)

        self.predict_button = StyledButton('Prever', self)
        self.predict_button.setEnabled(False)
        self.predict_button.clicked.connect(self.predict)
        hbox.addWidget(self.predict_button)

        layout.addLayout(hbox, 6, 0, 1, 3)

        self.help_button = QPushButton('Ajuda', self)
        self.help_button.setStyleSheet("""
            QPushButton {
                background-color: #cc092f;
                color: white;
                font-size: 12pt;
                min-width: 20px;
                max-width: 50px;
                height: 20px;
                border-radius: 15px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #e60032;
            }
            QPushButton:pressed {
                background-color: #b30026;
            }
        """)
        self.help_button.clicked.connect(self.show_help_dialog)
        layout.addWidget(self.help_button, 7, 0, 1, 2)

        self.setLayout(layout)

        self.file_path = None
        self.data = None
        self.model = None
        self.X = None
        self.y = None

        # Conectar sinais de mudança nos comboboxes para atualizar as métricas
        self.column_x_combobox.currentIndexChanged.connect(self.update_metrics)
        self.column_y_combobox.currentIndexChanged.connect(self.update_metrics)

    def load_csv(self):
        separator = self.separator_combobox.currentText()
        file_path, _ = QFileDialog.getOpenFileName(self, 'Selecionar Arquivo CSV', '', 'CSV Files (*.csv)')
        if file_path:
            try:
                self.data = self.regression_logic.load_data(file_path, separator)
                self.column_x_combobox.clear()
                self.column_y_combobox.clear()
                self.column_x_combobox.addItems(self.data.columns)
                self.column_y_combobox.addItems(self.data.columns)
                QMessageBox.information(self, "Informação", "CSV carregado com sucesso!")
                self.run_regression_button.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Falha ao carregar CSV: {e}")

    def run_regression(self):
        if self.data is not None:
            try:
                x_column = self.column_x_combobox.currentText()
                y_column = self.column_y_combobox.currentText()
                self.regression_logic.run_regression(x_column, y_column)
                self.plot_regression_button.setEnabled(True)
                self.plot_residuals_button.setEnabled(True)
                self.predict_button.setEnabled(True)
                QMessageBox.information(self, "Informação", "Regressão rodada com sucesso!")
                self.update_metrics()  # Atualiza as métricas após a regressão
            except ValueError as ve:
                self.correlation_label.setText("Dados não numéricos")
                self.r_squared_label.setText("Dados não numéricos")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Falha ao rodar regressão: {e}")

    def update_metrics(self):
        # Atualiza a exibição da correlação e R² na interface
        if self.column_x_combobox.currentText() and self.column_y_combobox.currentText():
            x_column = self.column_x_combobox.currentText()
            y_column = self.column_y_combobox.currentText()
            try:
                self.regression_logic.run_regression(x_column, y_column)
                if self.regression_logic.correlation is not None:
                    self.correlation_label.setText(f"Correlação: {self.regression_logic.correlation:.2f}")
                else:
                    self.correlation_label.setText("Dados não numéricos")

                if self.regression_logic.r_squared is not None:
                    self.r_squared_label.setText(f"R²: {self.regression_logic.r_squared:.2f}")
                else:
                    self.r_squared_label.setText("Dados não numéricos")
            except ValueError as ve:
                self.correlation_label.setText("Dados não numéricos")
                self.r_squared_label.setText("Dados não numéricos")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao calcular métricas: {e}")

    def plot_regression(self):
        try:
            self.regression_logic.plot_regression()
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao plotar regressão: {e}")

    def plot_residuals(self):
        try:
            self.regression_logic.plot_residuals()
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao plotar resíduos: {e}")

    def predict(self):
        try:
            x_value = float(self.input_value.text())
            prediction = self.regression_logic.predict(x_value)
            QMessageBox.information(self, "Previsão", f"A previsão para X={x_value} é Y={prediction.round(2)}")
        except ValueError:
            QMessageBox.critical(self, "Erro", "Por favor, insira um valor numérico válido para X.")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao fazer previsão: {e}")

    def show_help_dialog(self):
        help_dialog = HelpDialog(self)
        help_dialog.exec_()

    def center_window(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
