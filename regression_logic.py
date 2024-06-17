import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class RegressionLogic:
    def __init__(self):
        self.model = None
        self.X = None
        self.y = None
        self.correlation = None
        self.r_squared = None

    def load_data(self, file_path, separator=','):
        self.data = pd.read_csv(file_path, sep=separator)
        return self.data

    def run_regression(self, x_column, y_column):
        self.X = self.data[x_column].values.reshape(-1, 1)
        self.y = self.data[y_column].values

        if not np.issubdtype(self.X.dtype, np.number) or not np.issubdtype(self.y.dtype, np.number):
            raise ValueError("As colunas selecionadas devem ser numéricas.")

        self.model = LinearRegression()
        self.model.fit(self.X, self.y)

        # Calcula a correlação entre as colunas selecionadas
        self.correlation = self.data[x_column].corr(self.data[y_column])

        # Calcula o coeficiente de determinação (R²)
        y_pred = self.model.predict(self.X)
        self.r_squared = r2_score(self.y, y_pred)

    def plot_regression(self):
        if self.model is not None:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.scatter(self.X, self.y)
            plt.plot(self.X, self.model.predict(self.X), color='red')
            plt.title('Regressão Linear')
            plt.xlabel('X')
            plt.ylabel('y')
            plt.show()

    def plot_residuals(self):
        if self.model is not None:
            from yellowbrick.regressor import ResidualsPlot
            visualizer = ResidualsPlot(self.model)
            visualizer.fit(self.X, self.y)
            visualizer.show()

    def predict(self, x_value):
        x_value = np.array([[x_value]])
        return self.model.predict(x_value)[0]
