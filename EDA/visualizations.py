from __future__ import annotations
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import confusion_matrix

class Visualizations:
    @staticmethod
    def load_dataset(path="data/OULADX.csv"):
        """Load dataset from a CSV file."""
        return pd.read_csv(path)

    @staticmethod
    def descriptive_stats(df):
        """Return basic descriptive statistics for numeric columns."""
        desc = df.describe()
        print("\n--- Descriptive Statistics ---")
        print(desc)
        return desc


    @staticmethod
    def compute_kurtosis(df):
        """Compute kurtosis for numeric columns."""
        numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
        kurt = df[numeric_cols].kurtosis()
        print("\n--- Kurtosis ---")
        print(kurt)
        return kurt

    @staticmethod
    def univariate_plots(df):
        """Plot distributions for selected columns."""
        if "final_result" in df.columns:
            sns.countplot(data=df, x="final_result")
            plt.title("Distribuci\u00f3n de Resultados Finales")
            plt.show()

        numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
        for col in numeric_cols:
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribuci\u00f3n de {col}")
            plt.show()

    @staticmethod
    def bivariate_scatter(df, x="score", y="sum_click"):
        """Scatter plot for two numeric variables."""
        if x in df.columns and y in df.columns:
            sns.scatterplot(data=df, x=x, y=y)
            plt.title(f"Dispersi\u00f3n {x} vs {y}")
            plt.show()

    @staticmethod
    def box_plot(df, x="final_result", y="score"):
        """Box plot comparing a numeric variable across categories."""
        if x in df.columns and y in df.columns:
            sns.boxplot(data=df, x=x, y=y)
            plt.title(f"Boxplot de {y} por {x}")
            plt.show()

    @staticmethod
    def pair_plot(df):
        """Generate a scatter matrix for numeric features."""
        numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
        if len(numeric_cols) > 1:
            sns.pairplot(df[numeric_cols].dropna())
            plt.show()

    @staticmethod
    def correlation_heatmap(df):
        """Heatmap of correlations between numeric variables."""
        corr = df.select_dtypes(include="number").corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Mapa de Correlaci\u00f3n")
        plt.show()
        return corr

    @staticmethod
    def confusion_matrix_table(df, col1="final_result", col2="gender"):
        """Compute and print a confusion matrix (contingency table) between two categorical variables."""
        if col1 in df.columns and col2 in df.columns:
            table = pd.crosstab(df[col1], df[col2])
            print("\n--- Matriz de Confusi\u00f3n ---")
            print(table)
            return table
        else:
            print("Columns not found for confusion matrix.")
            return pd.DataFrame()