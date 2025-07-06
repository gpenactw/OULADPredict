import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

combined_dir = Path("./data/combined/")

def cleaning_data():
    output_dir = Path("./data")
    expected_files = [output_dir / "OULADX.csv"]

    if all(file.exists() for file in expected_files):
        print(f"Archivos ya generados en '{output_dir}'. Saltando limpieza de datos.")
        return

    # Cargar el dataset combinado
    df_assessment_combined = pd.read_csv(combined_dir / 'Assessment_combinado.csv')
    Courses_combined = pd.read_csv(combined_dir / 'Courses_combinado.csv')
    Registration_combined = pd.read_csv(combined_dir / 'Registration_combinado.csv')
    StudentAssessment_combined = pd.read_csv(combined_dir / 'StudentAssessment_combinado.csv')
    StudentInfo_combined = pd.read_csv(combined_dir / 'StudentInfo_combinado.csv')
    StudentVle_combined = pd.read_csv(combined_dir / 'StudentVle_combinado.csv')
    vle_combined = pd.read_csv(combined_dir / 'Vle_combinado.csv')

    # Revisar estructura
    print(df_assessment_combined.head())
    print(Courses_combined.head())
    print(Registration_combined.head())
    print(StudentAssessment_combined.head())
    print(StudentInfo_combined.head())
    print(StudentVle_combined.head())
    print(vle_combined.head())

    # Porcentaje de missing values
    print("Assessment\n",(df_assessment_combined.isnull().mean() * 100).round(2))
    print("\n")
    print("Courses\n",(Courses_combined.isnull().mean() * 100).round(2))
    print("\n")
    print("Registration\n",(Registration_combined.isnull().mean() * 100).round(2))
    print("\n")
    print("Student Assessment\n",(StudentAssessment_combined.isnull().mean() * 100).round(2))
    print("\n")
    print("Student Info\n",(StudentInfo_combined.isnull().mean() * 100).round(2))
    print("\n")
    print("Student Vle\n",(StudentVle_combined.isnull().mean() * 100).round(2))
    print("\n")
    print("Vle\n",(vle_combined.isnull().mean() * 100).round(2))

    # Eliminar columnas con >80% missing que no aportan

    cols_to_drop_assessment = ['guid_assess_id', 'days']
    cols_to_drop_student_assessment = [
        'guid_student_id', 'guid_assess_id', 'assessment_type', 'date', 'weight', 'gender', 'region',
        'highest_education', 'imd_band', 'age_band', 'num_of_prev_attempts', 'studied_credits',
        'disability', 'final_result', 'status', 'module', 'presentation', 'date_real_days', 'id_assetcode'
    ]
    cols_to_drop_student_vle = [
        'guid_student_id', 'guid_site_id', 'type_assign', 'week_from', 'weel_to',
        'disability', 'modulo', 'week1', 'week2', 'days', 'presentation'
    ]
    cols_to_drop_vle = ['guid_site_id', 'week_from', 'week_to']

    # Drop
    df_assessment_combined.drop(columns=cols_to_drop_assessment, inplace=True, errors='ignore')
    StudentAssessment_combined.drop(columns=cols_to_drop_student_assessment, inplace=True, errors='ignore')
    StudentVle_combined.drop(columns=cols_to_drop_student_vle, inplace=True, errors='ignore')
    vle_combined.drop(columns=cols_to_drop_vle, inplace=True, errors='ignore')

    # “Se eliminaron columnas con >80% de valores nulos y sin valor predictivo directo o sin correspondencia con otras tablas, para reducir ruido en el dataset.”

    # Completar valores
    Registration_combined['date_unregistration'] = Registration_combined['date_unregistration'].fillna(9999)
    StudentInfo_combined['imd_band'] = StudentInfo_combined['imd_band'].fillna('Unknown')
    StudentInfo_combined['age_band'] = StudentInfo_combined['age_band'].fillna(StudentInfo_combined['age_band'].mode()[0])
    StudentInfo_combined['num_of_prev_attempts'] = StudentInfo_combined['num_of_prev_attempts'].fillna(StudentInfo_combined['num_of_prev_attempts'].median())
    StudentInfo_combined['studied_credits'] = StudentInfo_combined['studied_credits'].fillna(StudentInfo_combined['studied_credits'].median())
    df_assessment_combined['date'] = df_assessment_combined['date'].fillna(df_assessment_combined['date'].median())
    StudentAssessment_combined = StudentAssessment_combined.dropna(subset=['id_assessment', 'id_student'])

    # Mostrar categorías únicas para definir el orden real
    print("age_band:", StudentInfo_combined['age_band'].unique())
    print("highest_education:", StudentInfo_combined['highest_education'].unique())
    print("imd_band:", StudentInfo_combined['imd_band'].unique())

    StudentInfo_combined['age_band'] = pd.Categorical(
        StudentInfo_combined['age_band'],
        categories=['0-35', '35-55', '55<='],
        ordered=True
    )

    StudentInfo_combined['highest_education'] = pd.Categorical(
        StudentInfo_combined['highest_education'],
        categories=[
            'No Formal quals',
            'Lower Than A Level',
            'A Level or Equivalent',
            'HE Qualification',
            'Post Graduate Qualification'
        ],
        ordered=True
    )

    StudentInfo_combined['imd_band'] = pd.Categorical(
        StudentInfo_combined['imd_band'],
        categories=[
            '0-10%',
            '10-20%',
            '20-30%',
            '30-40%',
            '40-50%'
        ],
        ordered=True
    )

    # Verificar datos limpios
    print(StudentInfo_combined.isnull().mean() * 100)
    print(df_assessment_combined.isnull().mean() * 100)
    print(Registration_combined.isnull().mean() * 100)
    print(StudentAssessment_combined.isnull().mean() * 100)
    print(StudentVle_combined.isnull().mean() * 100)
    print(vle_combined.isnull().mean() * 100)

    # Verificar datos limpios
    print(StudentInfo_combined.isnull().mean() * 100)
    print(df_assessment_combined.isnull().mean() * 100)
    print(Registration_combined.isnull().mean() * 100)
    print(StudentAssessment_combined.isnull().mean() * 100)
    print(StudentVle_combined.isnull().mean() * 100)
    print(vle_combined.isnull().mean() * 100)

    StudentInfo_combined.drop(columns=['guid_student_id'], inplace=True)
    df_assessment_combined.drop(columns=['id_assessment'], inplace=True)
    Registration_combined.drop(columns=['guid_studente_id'], inplace=True)
    vle_combined.drop(columns=['id_site'], inplace=True)

    df_main = StudentInfo_combined.merge(Registration_combined, on=['id_student', 'code_module', 'code_presentation'],
                                         how='left', suffixes=('_info', '_reg'))
    df_main = df_main.merge(StudentAssessment_combined, on=['id_student'], how='left', suffixes=('', '_studassess'))
    df_main = df_main.merge(df_assessment_combined, on=['id_assessment_general'], how='left', suffixes=('', '_assess'))
    df_main = df_main.merge(StudentVle_combined, on=['id_student', 'code_module', 'code_presentation'], how='left',
                            suffixes=('', '_vle'))

    df_main.to_csv('./data/OULADX.csv', index=False)