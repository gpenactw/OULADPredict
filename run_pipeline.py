from ETL.etl_process import ETLProcess
from EDA.eda_analysis import EDAAnalysis

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



def main():
    print("\n" + "="*50)
    print(" Sistema OULAD - Aprendizaje Supervisado")
    print("="*50)

    while True:
        print("\n=== MENÚ PRINCIPAL ===")
        print("1. Ejecutar ETL")
        print("2. Ejecutar EDA (Análisis Exploratorio)")
        print("0. Salir")

        choice = input("\nSelecciona una opción: ")

        if choice == "1":
            run_etl_process()
        elif choice == "2":
            run_eda()
        elif choice == "0":
            print("\n¡Hasta luego!")
            break
        else:
            print("\n⚠️  Opción inválida. Por favor intenta de nuevo.")

if __name__ == "__main__":
    main()
