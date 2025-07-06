from ETL.etl_process import ETLProcess

def run_etl_process():
    print("\n¿Está seguro que desea ejecutar el ETL? Esto puede tomar varios minutos.")
    response = input("Continuar? (s/n): ")

    if response.lower() == 's':
        etl = ETLProcess()
        etl.run()
    else:
        print("ETL cancelado.")

def main():
    print("\n" + "="*50)
    print(" Sistema OULAD - Aprendizaje Supervisado")
    print("="*50)

    while True:
        print("\n=== MENÚ PRINCIPAL ===")
        print("1. Ejecutar ETL")
        print("0. Salir")

        choice = input("\nSelecciona una opción: ")

        if choice == "1":
            run_etl_process()
        elif choice == "0":
            print("\n¡Hasta luego!")
            break
        else:
            print("\n⚠️  Opción inválida. Por favor intenta de nuevo.")

if __name__ == "__main__":
    main()
