from EDA.visualizations import Visualizations

class EDAAnalysis:
    def __init__(self, path: str = "data/OULADX.csv"):
        self.path = path
        self.df = None
        self.viz = Visualizations()

    def run(self):
        # Cargar dataset
        self.df = self.viz.load_dataset(self.path)

        # Ejecutar análisis
        self.viz.descriptive_stats(self.df)
        self.viz.compute_kurtosis(self.df)
        self.viz.univariate_plots(self.df)

        y_col = "sum_click_total" if "sum_click_total" in self.df.columns else "sum_click"
        self.viz.bivariate_scatter(self.df, y=y_col)

        self.viz.box_plot(self.df)
        self.viz.correlation_heatmap(self.df)
        self.viz.confusion_matrix_table(self.df)
        self.viz.pair_plot(self.df)
