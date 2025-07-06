from pathlib import Path
import pandas as pd
from tqdm import tqdm

def combine_source(df1, df2, source1='original', source2='experimento'):
    df1 = df1.assign(source=source1)
    df2 = df2.assign(source=source2)
    combined = pd.concat([df1, df2], ignore_index=True).drop_duplicates().reset_index(drop=True)
    return combined

def load_and_combine_data(data_dir, excel_filename):
    data_dir = Path(data_dir)
    excel_path = excel_filename

    # CSV (original OULAD)
    df_assessment = pd.read_csv(data_dir / "assessments.csv")
    df_student_vle = pd.read_csv(data_dir / "studentVle.csv", on_bad_lines="skip")
    df_courses = pd.read_csv(data_dir / "courses.csv")
    df_student_info = pd.read_csv(data_dir / "studentInfo.csv")
    df_vle = pd.read_csv(data_dir / "vle.csv")
    df_registration = pd.read_csv(data_dir / "studentRegistration.csv")
    df_student_assessment = pd.read_csv(data_dir / "studentAssessment.csv")

    # Normalizar claves
    df_assessment["id_assessment_general"] = df_assessment["id_assessment"].astype(str)
    df_student_vle["student_id_general"] = df_student_vle["id_student"].astype(str)
    df_student_vle["id_site_general"] = df_student_vle["id_site"].astype(str)
    df_student_info["student_id_general"] = df_student_info["id_student"].astype(str)
    df_vle["id_site_general"] = df_vle["id_site"].astype(str)
    df_registration["student_id_general"] = df_registration["id_student"].astype(str)
    df_student_assessment["student_id_general"] = df_student_assessment["id_student"].astype(str)
    df_student_assessment["id_assessment_general"] = df_student_assessment["id_assessment"].astype(str)

    # Cargar el archivo excel (experimento)
    df_assess_plan = pd.read_excel(excel_path, sheet_name="Assess Plan")
    df_assess_plan["id_assessment_general"] = df_assess_plan["guid_assess_id"].astype(str)

    df_vle_clickstream = pd.read_excel(excel_path, sheet_name="VLE_clickStream")
    df_vle_clickstream["id_site_general"] = df_vle_clickstream["guid_site_id"].astype(str)
    df_vle_clickstream["student_id_general"] = df_vle_clickstream["guid_student_id"].astype(str)

    df_cursos = pd.read_excel(excel_path, sheet_name="cursos")

    df_student_info_new = pd.read_excel(excel_path, sheet_name="StudentInfo")
    df_student_info_new["student_id_general"] = df_student_info_new["guid_student_id"].astype(str)

    df_vle_modules = pd.read_excel(excel_path, sheet_name="Vle_modules")
    df_vle_modules["id_site_general"] = df_vle_modules["guid_site_id"].astype(str)

    df_registration_new = pd.read_excel(excel_path, sheet_name="Registration")
    df_registration_new["student_id_general"] = df_registration_new["guid_studente_id"].astype(str)

    df_assess_detail = pd.read_excel(excel_path, sheet_name="Assesss_detail")
    df_assess_detail["student_id_general"] = df_assess_detail["guid_student_id"].astype(str)
    df_assess_detail["id_assessment_general"] = df_assess_detail["guid_assess_id"].astype(str)

    # Combinar
    df_assessment_combined = combine_source(df_assessment, df_assess_plan)
    df_student_vle_combined = combine_source(df_student_vle, df_vle_clickstream)
    df_courses_combined = combine_source(df_courses, df_cursos)
    df_student_info_combined = combine_source(df_student_info, df_student_info_new)
    df_vle_combined = combine_source(df_vle, df_vle_modules)
    df_registration_combined = combine_source(df_registration, df_registration_new)
    df_student_assessment_combined = combine_source(df_student_assessment, df_assess_detail)

    return {
        "Assessment_combinado.csv": df_assessment_combined,
        "StudentVle_combinado.csv": df_student_vle_combined,
        "Courses_combinado.csv": df_courses_combined,
        "StudentInfo_combinado.csv": df_student_info_combined,
        "Vle_combinado.csv": df_vle_combined,
        "Registration_combinado.csv": df_registration_combined,
        "StudentAssessment_combinado.csv": df_student_assessment_combined,
    }


def generate_combine_csv(data_dir="./data/raw", output_dir="./data/combined", show_progress=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_files = [
        "Assessment_combinado.csv",
        "StudentVle_combinado.csv",
        "Courses_combinado.csv",
        "StudentInfo_combinado.csv",
        "Vle_combinado.csv",
        "Registration_combinado.csv",
        "StudentAssessment_combinado.csv",
    ]

    # Verificar si ya existen todos los archivos
    if all((output_dir / file).exists() for file in expected_files):
        print(f"Archivos ya generados en '{output_dir}'. Saltando generación.")
        return

    # Si faltan, cargar y combinar los datos
    datasets = load_and_combine_data(data_dir=data_dir, excel_filename="./data/AnonymisezData_oulad_context-Kongo-2024.xlsx")
    iterator = tqdm(datasets.items(), desc="Guardando CSVs") if show_progress else datasets.items()

    for filename, df in iterator:
        df.to_csv(output_dir / filename, index=False)

    print(f"\nrchivos combinados guardados en '{output_dir}'.")
