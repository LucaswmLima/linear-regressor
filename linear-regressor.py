import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMessageBox, QFileDialog, QLabel, QLineEdit, QHBoxLayout, QGridLayout, QDesktopWidget, QDialog, QTextBrowser
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

class StyledButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        # Estilo dos botões
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

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Ajuda - Regressão Linear Simples')
        self.setWindowIcon(QIcon('assets/icon.png'))

        layout = QVBoxLayout()

        # Texto de ajuda
        text = """
        <h2>Como utilizar o programa:</h2>
        <p>1. Clique no botão 'Carregar CSV' para selecionar um arquivo CSV contendo seus dados.</p>
        <p>2. Após carregar o arquivo, clique no botão 'Rodar Regressão' para calcular a regressão linear.</p>
        <p>3. Use os botões 'Mostrar Regressão' e 'Mostrar Resíduos' para visualizar os gráficos.</p>
        <p>4. Insira um valor de X no campo correspondente e clique em 'Prever' para fazer uma previsão.</p>
        <p><b>Créditos:</b></p>
        <p>Desenvolvido por LUCAS WILLIAM MARTINS LIMA.</p>
        <p>Github: <a href="https://www.github.com/lucaswmlima">www.github.com/lucaswmlima</a></p>
        """

        help_text = QTextBrowser()
        help_text.setHtml(text)
        help_text.setOpenExternalLinks(True)
        help_text.setMinimumWidth(300)  # Largura mínima da janela de ajuda
        help_text.setMinimumHeight(300)  # Altura mínima da janela de ajuda
        layout.addWidget(help_text)

        self.setLayout(layout)

class RegressionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Configurações iniciais da janela principal
        self.setWindowTitle('Regressão Linear Simples')
        self.setFixedSize(500, 350)  # Tamanho fixo da janela
        self.setStyleSheet("background-color: white;")  # Cor de fundo branco

        # Ícone da janela
        self.setWindowIcon(QIcon('assets/icon.png'))

        layout = QGridLayout()
        layout.setSpacing(10)  # Espaçamento entre os widgets

        # Botão para carregar CSV
        self.load_button = StyledButton('Carregar CSV', self)
        self.load_button.clicked.connect(self.load_csv)
        layout.addWidget(self.load_button, 0, 0)

        # Botão para rodar a regressão
        self.run_regression_button = StyledButton('Rodar Regressão', self)
        self.run_regression_button.setEnabled(False)
        self.run_regression_button.clicked.connect(self.run_regression)
        layout.addWidget(self.run_regression_button, 0, 1)

        # Botão para mostrar o gráfico da regressão
        self.plot_regression_button = StyledButton('Mostrar Regressão', self)
        self.plot_regression_button.setEnabled(False)
        self.plot_regression_button.clicked.connect(self.plot_regression)
        layout.addWidget(self.plot_regression_button, 1, 0)

        # Botão para mostrar o gráfico dos resíduos
        self.plot_residuals_button = StyledButton('Mostrar Resíduos', self)
        self.plot_residuals_button.setEnabled(False)
        self.plot_residuals_button.clicked.connect(self.plot_residuals)
        layout.addWidget(self.plot_residuals_button, 1, 1)

        # Layout horizontal para entrada de X e botão de previsão
        hbox = QHBoxLayout()
        hbox.setSpacing(10)  # Espaçamento entre os widgets

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

        layout.addLayout(hbox, 2, 0, 1, 2)

        # Botão de ajuda
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
        layout.addWidget(self.help_button, 3, 0, 1, 2)

        self.setLayout(layout)

        # Variáveis de controle
        self.file_path = None
        self.data = None
        self.model = None
        self.X = None
        self.y = None

    def load_csv(self):
        # Abrir o explorador de arquivos padrão do sistema
        file_path, _ = QFileDialog.getOpenFileName(self, 'Selecionar Arquivo CSV', '', 'CSV Files (*.csv)')
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                QMessageBox.information(self, "Informação", "CSV carregado com sucesso!")
                self.run_regression_button.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Falha ao carregar CSV: {e}")

    def run_regression(self):
        if self.data is not None:
            try:
                self.data = self.data.drop(self.data.columns[0], axis=1)
                self.X = self.data.iloc[:, 1].values.reshape(-1, 1)
                self.y = self.data.iloc[:, 0].values

                self.model = LinearRegression()
                self.model.fit(self.X, self.y)

                self.plot_regression_button.setEnabled(True)
                self.plot_residuals_button.setEnabled(True)
                self.predict_button.setEnabled(True)

                QMessageBox.information(self, "Informação", "Regressão rodada com sucesso!")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Falha ao rodar regressão: {e}")

    def plot_regression(self):
        if self.model is not None:
            try:
                plt.figure()
                plt.scatter(self.X, self.y)
                plt.plot(self.X, self.model.predict(self.X), color='red')
                plt.title('Regressão Linear')
                plt.xlabel('X')
                plt.ylabel('y')
                plt.show()
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Falha ao mostrar regressão: {e}")

    def plot_residuals(self):
        if self.model is not None:
            try:
                plt.figure()
                visualizer = ResidualsPlot(self.model)
                visualizer.fit(self.X, self.y)
                visualizer.poof()
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Falha ao mostrar resíduos: {e}")

    def predict(self):
        if self.model is not None:
            try:
                x_input = float(self.input_value.text())
                prediction = self.model.predict([[x_input]])
                QMessageBox.information(self, "Previsão", f"Para X={x_input}, a previsão é Y={prediction[0]:.2f}")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Falha ao fazer previsão: {e}")

    def show_help_dialog(self):
        # Mostra o diálogo de ajuda
        help_dialog = HelpDialog(self)
        help_dialog.exec_()

    def center_window(self):
        # Centraliza a janela na tela
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = RegressionApp()
    ex.center_window()  # Centraliza a janela principal na tela
    ex.show()
    sys.exit(app.exec_())
