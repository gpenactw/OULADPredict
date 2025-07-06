from ETL.etl_process import ETLProcess
from EDA.eda_analysis import EDAAnalysis
from MODELING.train_models import Modeling
from EDA.hipotesis import ModelosOULAD

def run_etl_process():
    print("\n¿Está seguro que desea ejecutar el ETL? Esto puede tomar varios minutos.")
    response = input("Continuar? (s/n): ")

    if response.lower() == 's':
        etl = ETLProcess()
        etl.run()
    else:
        print("ETL cancelado.")

def run_eda():
    print("\nEjecutando Análisis Exploratorio de Datos (EDA)...")
    eda = EDAAnalysis()
    eda.run()

def run_modeling():
    print("\nEjecutando Modelo (algoritmos supervisado)...")
    model = Modeling()
    model.run("all", True)

def run_hipotesis():
    print("¡Ejecutando pruebas de hipótesis y regresiones simples sobre OULAD!")
    h = ModelosOULAD()
    h.run()

def main():
    print("\n" + "="*50)
    print(" Sistema OULAD - Aprendizaje Supervisado")
    print("="*50)

    while True:
        print("\n=== MENÚ PRINCIPAL ===")
        print("1. Ejecutar ETL")
        print("2. Ejecutar EDA (Análisis Exploratorio)")
        print("3. Ejecutar Hipótesis")
        print("4. Ejecutar MODELING (Entrenamiento)")
        print("0. Salir")

        choice = input("\nSelecciona una opción: ")

        if choice == "1":
            run_etl_process()
        elif choice == "2":
            run_eda()
        elif choice == "3":
            run_hipotesis()
        elif choice == "4":
            run_modeling()
        elif choice == "0":
            print("\n¡Hasta luego!")
            break
        else:
            print("\n⚠️  Opción inválida. Por favor intenta de nuevo.")

if __name__ == "__main__":
    main()
