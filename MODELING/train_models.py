import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

class Modeling:
    def __init__(self, path="data/OULADX.csv"):
        self.path = path

    def load_dataset(self, path=None):
        """Carga el dataset combinado desde un archivo CSV."""
        if path is None:
            path = self.path
        return pd.read_csv(path, low_memory=False)

    def prepare_features(self, df):
        """Crea las columnas necesarias para el modelo a partir de datos brutos."""
        grouped = df.groupby("id_student").agg({
            "score": "mean",
            "id_assessment": "count",
            "weight": "sum",
            "sum_click_total": "max",
            "studied_credits": "first",
            "num_of_prev_attempts": "first",
            "final_result": "first"
        }).reset_index()

        grouped = grouped.rename(columns={
            "score": "score_mean",
            "id_assessment": "n_assessments",
            "weight": "total_weight",
            "sum_click_total": "total_clicks",
            "final_result": "target"
        })

        grouped["target"] = grouped["target"].map(lambda x: 1 if x == "Pass" else 0)
        return grouped

    def split_data(self, df):
        """Divide los datos en entrenamiento y prueba."""
        features = [
            "score_mean",
            "n_assessments",
            "total_weight",
            "total_clicks",
            "studied_credits",
            "num_of_prev_attempts",
        ]
        df = df.dropna(subset=features + ["target"])
        X = df[features]
        y = df["target"]
        return train_test_split(X, y, test_size=0.3, random_state=42)

    def train_logistic_regression(self, X_train, X_test, y_train, y_test, export=False):
        """Entrena regresión logística."""
        print("\n=== Logistic Regression ===")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

        if export:
            self.evaluate_and_export("logistic_regression", y_test, y_pred)

    def train_random_forest(self, X_train, X_test, y_train, y_test, export=False):
        """Entrena Random Forest."""
        print("\n=== Random Forest ===")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

        if export:
            self.evaluate_and_export("random_forest", y_test, y_pred)

    def train_gradient_boosting(self, X_train, X_test, y_train, y_test, export=False):
        """Entrena Gradient Boosting."""
        print("\n=== Gradient Boosting ===")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

        if export:
            self.evaluate_and_export("gradient_boosting", y_test, y_pred)

    def describe_clusters(self, clustered_df):
        """Muestra estadísticas descriptivas por grupo (clúster) para entender su perfil."""
        print("\n=== Análisis descriptivo por clúster ===")
        summary = clustered_df.groupby("cluster")[[
            "score_mean", "total_clicks", "studied_credits"
        ]].agg(["mean", "median", "std", "count"])

        print(summary)
        return summary

    def explore_clusters(self, df, n_clusters=3):
        """Agrupa estudiantes por características numéricas usando K-Means."""
        print("\n=== K-Means Clustering ===")
        cols = ["score_mean", "total_clicks", "studied_credits"]
        df = df.dropna(subset=cols).copy()
        scaler = StandardScaler()
        X = scaler.fit_transform(df[cols])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df["cluster"] = kmeans.fit_predict(X)

        # Mostrar conteos
        print("Cluster counts:\n", df["cluster"].value_counts())

        # Mostrar análisis descriptivo
        self.describe_clusters(df)

        return df

    def evaluate_and_export(self, model_name, y_test, y_pred, output_dir="outputs"):
        """Evalúa manualmente las métricas y exporta CSV con resultados por caso."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Construir DataFrame con predicciones
        df_result = pd.DataFrame({
            "y_test": y_test,
            "y_pred": y_pred
        })
        df_result.index.name = "case"

        # Guardar CSV de resultados por caso
        case_csv_path = f"{output_dir}/{model_name}_predictions.csv"
        df_result.to_csv(case_csv_path, index=True)

        # Calcular TP, FP, TN, FN
        TP = ((y_test == 1) & (y_pred == 1)).sum()
        TN = ((y_test == 0) & (y_pred == 0)).sum()
        FP = ((y_test == 0) & (y_pred == 1)).sum()
        FN = ((y_test == 1) & (y_pred == 0)).sum()

        # F1 Score Manual
        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        # MSE
        mse = mean_squared_error(y_test, y_pred)

        # Exportar métricas a CSV
        metrics = {
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "Precision": precision,
            "Recall": recall,
            "F1_score": f1,
            "MSE": mse,
        }
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f"{output_dir}/{model_name}_metrics.csv", index=False)

        # Imprimir
        print(f"\n--- {model_name.upper()} METRICS ---")
        print(metrics_df.to_string(index=False))

    def run(self, model_type="all", export=False):
        """Ejecuta el flujo completo según el modelo deseado."""
        data = self.load_dataset()
        prepared = self.prepare_features(data)
        print("\n=== Muestra del dataset preparado ===")
        print(prepared.head())

        if model_type in ["logistic", "random_forest", "gradient_boosting", "all"]:
            X_train, X_test, y_train, y_test = self.split_data(prepared)

        if model_type == "logistic":
            self.train_logistic_regression(X_train, X_test, y_train, y_test, export)
        elif model_type == "random_forest":
            self.train_random_forest(X_train, X_test, y_train, y_test, export)
        elif model_type == "gradient_boosting":
            self.train_gradient_boosting(X_train, X_test, y_train, y_test, export)
        elif model_type == "clustering":
            self.explore_clusters(prepared)
        elif model_type == "all":
            self.train_logistic_regression(X_train, X_test, y_train, y_test, export)
            self.train_random_forest(X_train, X_test, y_train, y_test, export)
            self.train_gradient_boosting(X_train, X_test, y_train, y_test, export)
            self.explore_clusters(prepared)