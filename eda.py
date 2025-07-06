# eda.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(df):
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="final_result")
    plt.title("Distribución de Resultados Finales")
    plt.show()

    sns.histplot(df["total_clicks"].dropna(), bins=30, kde=True)
    plt.title("Distribución de Clicks")
    plt.show()

    sns.boxplot(data=df, x="target", y="score_mean")
    plt.title("Promedio de Notas vs Target")
    plt.show()

def correlation_heatmap(df):
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Mapa de Correlación")
    plt.show()
