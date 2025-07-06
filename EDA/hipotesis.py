import pandas as pd
import numpy as np
from scipy import stats

class HipotesisOULAD:
    def __init__(self, df):
        self.df = df.copy()

    def chi_square_education_vs_result(self):
        print("\nEjecutando Chi-cuadrado: 'highest_education' vs 'final_result'")
        tabla = pd.crosstab(self.df['highest_education'], self.df['final_result'])
        print(tabla)
        chi2, p, _, _ = stats.chi2_contingency(tabla)
        print(f"Chi² = {chi2:.3f}, p-valor = {p:.4f}")

    def compare_studied_credits(self):
        print("\nComparando 'studied_credits' entre aprobados y no aprobados")
        aprob = self.df[self.df['final_result'] == 'Pass']['studied_credits']
        reprob = self.df[self.df['final_result'] == 'Fail']['studied_credits']
        _, p1 = stats.shapiro(aprob)
        _, p2 = stats.shapiro(reprob)

        if p1 > 0.05 and p2 > 0.05:
            t_stat, p_val = stats.ttest_ind(aprob, reprob)
            print(f"t = {t_stat:.3f}, p-valor = {p_val:.4f} (t-test)")
        else:
            u_stat, p_val = stats.mannwhitneyu(aprob, reprob)
            print(f"U = {u_stat:.3f}, p-valor = {p_val:.4f} (Mann-Whitney)")

    def correlacion_clicks_vs_score(self):
        print("\nCorrelación entre 'sum_click' y 'score'")
        df_corr = self.df.dropna(subset=['sum_click', 'score'])
        _, p_x = stats.shapiro(df_corr['sum_click'])
        _, p_y = stats.shapiro(df_corr['score'])

        if p_x > 0.05 and p_y > 0.05:
            corr, p_val = stats.pearsonr(df_corr['sum_click'], df_corr['score'])
            print(f"Pearson r = {corr:.3f}, p-valor = {p_val:.4f}")
        else:
            corr, p_val = stats.spearmanr(df_corr['sum_click'], df_corr['score'])
            print(f"Spearman ρ = {corr:.3f}, p-valor = {p_val:.4f}")

    def wilcoxon_simulado(self):
        print("\nWilcoxon simulado: clicks antes vs después")
        df_w = self.df[['id_student', 'sum_click']].dropna()
        df_w['clicks_mitad1'] = df_w['sum_click'] * np.random.uniform(0.4, 0.6, len(df_w))
        df_w['clicks_mitad2'] = df_w['sum_click'] - df_w['clicks_mitad1']
        stat, p_val = stats.wilcoxon(df_w['clicks_mitad1'], df_w['clicks_mitad2'])
        print(f"Wilcoxon stat = {stat:.3f}, p-valor = {p_val:.4f}")

    def logistic_regression_clicks_vs_pass_simple(self):
        print("\nRegresión logística simple aproximada: sum_click predice aprobación")
        # Aquí hacemos regresión logística simple con sklearn para evitar statsmodels
        from sklearn.linear_model import LogisticRegression
        df_model = self.df.dropna(subset=['sum_click', 'final_result']).copy()
        df_model['pass_bin'] = df_model['final_result'].apply(lambda x: 1 if x == 'Pass' else 0)
        X = df_model[['sum_click']]
        y = df_model['pass_bin']
        model = LogisticRegression()
        model.fit(X, y)
        score = model.score(X, y)
        print(f"Precisión del modelo: {score:.3f}")
        coefs = model.coef_[0][0]
        intercept = model.intercept_[0]
        print(f"Coeficiente (sum_click): {coefs:.4f}, Intercepto: {intercept:.4f}")

class ModelosOULAD:
    def __init__(self, df):
        self.df = df.copy()

    def regresion_lineal_simple(self, x_col, y_col):
        print(f"\nRegresión lineal simple: '{y_col}' vs '{x_col}'")
        df_model = self.df.dropna(subset=[x_col, y_col])
        x = df_model[x_col]
        y = df_model[y_col]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        print(f"Slope: {slope:.4f}, Intercept: {intercept:.4f}")
        print(f"R^2: {r_value**2:.4f}, p-valor: {p_value:.4f}")
        print(f"Error estándar: {std_err:.4f}")

if __name__ == "__main__":
    print("¡Ejecutando pruebas de hipótesis y regresiones simples sobre OULAD!")
    path = "data/OULADX.csv"
    df = pd.read_csv(path)
    print(f"Datos cargados: {df.shape}")

    print("\n=== PRUEBAS DE HIPÓTESIS ===")
    pruebas = HipotesisOULAD(df)
    pruebas.chi_square_education_vs_result()
    pruebas.compare_studied_credits()
    pruebas.correlacion_clicks_vs_score()
    pruebas.wilcoxon_simulado()
    pruebas.logistic_regression_clicks_vs_pass_simple()

    print("\n=== REGRESIONES SIMPLES ===")
    modelos = ModelosOULAD(df)
    modelos.regresion_lineal_simple('studied_credits', 'score')
    modelos.regresion_lineal_simple('sum_click', 'score')
