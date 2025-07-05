from pathlib import Path
import pandas as pd
from tqdm import tqdm

def load_raw_data(data_dir="data"):
    data_dir = Path(data_dir)

    files_to_load = [
        ("studentInfo.csv", pd.read_csv),
        ("studentAssessment.csv", pd.read_csv),
        ("assessments.csv", pd.read_csv),
        ("studentVle.csv", pd.read_csv),
        ("vle.csv", pd.read_csv),
        ("AnonymisezData_oulad_context-Kongo-2024.xlsx", pd.read_excel),
    ]

    loaded_data = []

    tqdm_bar = tqdm(files_to_load, desc="Cargando CSV", unit="file")
    for filename, loader in tqdm_bar:
        tqdm_bar.set_postfix_str(f"Loading: {filename}")
        tqdm.write(f">>> Loading {filename}")
        data_path = data_dir / filename
        df = loader(data_path)
        loaded_data.append(df)

    return tuple(loaded_data)

def clean_student_info(df):
    df = df.copy()

    # Fill missing values
    df["age_band"] = df["age_band"].fillna("Unknown")
    df["gender"] = df["gender"].fillna("Unknown")
    df["num_of_prev_attempts"] = df["num_of_prev_attempts"].fillna(df["num_of_prev_attempts"].median())

    # Education level ordinal encoding
    edu_map = {
        "No Formal quals": 0,
        "Lower Than A Level": 1,
        "A Level or Equivalent": 2,
        "HE Qualification": 3,
        "Post Graduate Qualification": 4
    }
    df["highest_education_code"] = df["highest_education"].map(edu_map)

    # One-hot encode region
    if "region" in df.columns:
        df = pd.get_dummies(df, columns=["region"], drop_first=True)

    return df

def clean_registration(df):
    df = df.copy()
    df["date_registration"] = df["date_registration"].fillna(df["date_registration"].median())
    df["date_unregistration"] = df["date_unregistration"].fillna(-1)
    return df

def clean_vle_clicks(df):
    df = df.copy()
    df["sum_click"] = df["sum_click"].fillna(0)
    return df


def prepare_dataset(student_info, student_assess, assessments, student_vle):
    steps = [
        "Merging student assessments with assessment metadata",
        "Merging with student information",
        "Aggregating assessment features",
        "Aggregating click features from VLE",
        "Merging aggregated features",
        "Generating target labels",
    ]

    tqdm_bar = tqdm(steps, desc="Preparando dataset", unit="step")

    # Step 1: Merge assessments
    tqdm_bar.set_postfix_str(steps[0])
    df = student_assess.merge(assessments, on="id_assessment", how="left")
    tqdm_bar.update()

    # Step 2: Merge with student info
    tqdm_bar.set_postfix_str(steps[1])
    df = df.merge(student_info, on="id_student", how="left")
    tqdm_bar.update()

    # Step 3: Aggregate assessment features
    tqdm_bar.set_postfix_str(steps[2])
    df_agg = df.groupby("id_student").agg({
        "score": ["mean", "count"],
        "date_submitted": "max",
        "date": "max",
        "weight": "sum"
    }).reset_index()
    df_agg.columns = ["id_student", "score_mean", "n_assessments", "last_submit", "last_assess_date", "total_weight"]
    tqdm_bar.update()

    # Step 4: Aggregate click features
    tqdm_bar.set_postfix_str(steps[3])
    click_agg = student_vle.groupby("id_student")["sum_click"].sum().reset_index()
    click_agg.columns = ["id_student", "total_clicks"]
    tqdm_bar.update()

    # Step 5: Merge everything
    tqdm_bar.set_postfix_str(steps[4])
    final_df = df_agg.merge(student_info, on="id_student", how="left")
    final_df = final_df.merge(click_agg, on="id_student", how="left")
    tqdm_bar.update()

    # Step 6: Target label
    tqdm_bar.set_postfix_str(steps[5])
    final_df["target"] = final_df["final_result"].apply(lambda x: 1 if x in ["Fail", "Withdrawn"] else 0)
    tqdm_bar.update()

    tqdm_bar.close()
    return final_df
