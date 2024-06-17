from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextBrowser
from PyQt5.QtGui import QIcon

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Ajuda - Regressão Linear Simples')
        self.setWindowIcon(QIcon('assets/icon.png'))

        layout = QVBoxLayout()
        text = """
        <h2>Como utilizar o programa:</h2>
        <p>1. Selecione o separador correto para seu arquivo CSV.</p>
        <p>2. Clique no botão 'Carregar CSV' para selecionar um arquivo CSV contendo seus dados.</p>
        <p>3. Escolha as colunas de X e Y que serão utilizadas para a regressão linear.</p>
        <p>5. Após a escolha, serão mostradas a correlação entre as colunas de X e Y, assim como o coeficiente de determinação (R²)</p>
        <p>4. Clique no botão 'Rodar Regressão' para calcular a regressão linear.</p>
        <p>6. Utilize os botões 'Mostrar Regressão' e 'Mostrar Resíduos' para visualizar os gráficos correspondentes.</p>
        <p>7. Insira um valor de X no campo correspondente e clique em 'Prever' para fazer uma previsão do valor de Y.</p>
        <p><b>Créditos:</b></p>
        <p>Desenvolvido por LUCAS WILLIAM MARTINS LIMA.</p>
        <p>Github: <a href="https://www.github.com/lucaswmlima">www.github.com/lucaswmlima</a></p>
        """
        help_text = QTextBrowser()
        help_text.setHtml(text)
        help_text.setOpenExternalLinks(True)
        help_text.setMinimumWidth(300)
        help_text.setMinimumHeight(380)
        layout.addWidget(help_text)
        self.setLayout(layout)
