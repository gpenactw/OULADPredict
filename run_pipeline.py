# run_pipeline.py
from etl import load_raw_data, prepare_dataset
from eda import plot_distributions, correlation_heatmap
from modeling import split_data, train_model, evaluate_model

def main():
    print("\n" + "="*50)
    print(" Sistema OULAD - Aprendizaje Supervisado")
    print("="*50)

    # ETL
    print("\n=== ETL INICIADO   ===")
    print("\n")
    si, sa, a, sv, v, xlsx = load_raw_data()
    print("\n")
    df = prepare_dataset(si, sa, a, sv)
    print("\n=== DATASET FINAL  ===")
    print(df.head())
    print("\n=== LIMPIANDO DF   ===")
    # todo
    print("\n=== ETL COMPLETADO ===")

    # EDA
    print("\n=== EDA   ===")
    plot_distributions(df)
    correlation_heatmap(df)

    # Modeling
    print("\n=== MODELO   ===")
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()