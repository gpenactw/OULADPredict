from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import time


def split_data(df):
    tqdm.write("🔄 Splitting dataset into train/test...")

    features = [
        "score_mean", "n_assessments", "total_weight",
        "total_clicks", "studied_credits", "num_of_prev_attempts"
    ]
    df = df.dropna(subset=features + ["target"])

    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    tqdm.write(f"✅ Training set: {X_train.shape}, Testing set: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    tqdm.write("🧠 Training RandomForestClassifier...")
    start_time = time.time()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    elapsed = time.time() - start_time
    tqdm.write(f"✅ Model trained in {elapsed:.2f} seconds.")
    return model


def evaluate_model(model, X_test, y_test):
    tqdm.write("📊 Evaluating model...")
    y_pred = model.predict(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))