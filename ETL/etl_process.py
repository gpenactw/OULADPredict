from pathlib import Path
from .concat import generate_combine_csv
from .cleaning import cleaning_data
class ETLProcess:
    def check_datasets(self):
        """Verifica si los datasets están descargados."""
        datasets_path = Path("./data/raw/")
        required_files = [
            "assessments.csv", "courses.csv", "studentAssessment.csv",
            "studentInfo.csv", "studentRegistration.csv", "studentVle.csv", "vle.csv"
        ]

        missing_files = []
        for file in required_files:
            if not (datasets_path / file).exists():
                missing_files.append(file)

        if missing_files:
            print("\nFaltan los siguientes archivos de datos:")
            for file in missing_files:
                print(f"   - {file}")
            print("\nPor favor ejecuta primero:")
            print("  python data/downloadDatasets.py data/raw")
            return False

        return True

    def run(self):
        if not self.check_datasets():
            return
        print("\n=== INICIANDO PROCESO ETL OULAD ===\n")
        print("\n")
        generate_combine_csv(data_dir="./data/raw", output_dir="./data/combined")
        cleaning_data()
        print("\n=== fin ===\n")

