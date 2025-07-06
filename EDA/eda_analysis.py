from EDA.visualizations import Visualizations

class EDAAnalysis:
    def __init__(self, path: str = "data/OULADX.csv"):
        self.path = path
        self.df = None
        self.viz = Visualizations()

    def run(self):
        # Cargar dataset
        self.df = self.viz.load_dataset(self.path)

        # Descriptivo
        self.viz.descriptive_stats(self.df)

        # Curtosis de columnas numéricas
        self.viz.compute_kurtosis(self.df)

        # Distribuciones univariadas
        self.viz.univariate_plots(self.df)

        # Gráfico de dispersión
        y_col = "sum_click_total" if "sum_click_total" in self.df.columns else "sum_click"
        self.viz.bivariate_scatter(self.df, y=y_col)

        # Boxplot
        self.viz.box_plot(self.df)

        # Heatmap de correlación
        self.viz.correlation_heatmap(self.df)

        # Matriz de confusión (categórica)
        self.viz.confusion_matrix_table(self.df)

        # Matriz de dispersión entre variables numéricas
        self.viz.pair_plot(self.df)