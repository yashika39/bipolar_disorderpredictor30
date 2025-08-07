from sklearn.model_selection import train_test_split
from preprocess import preprocess_data
from models.model_utils import get_models, tune_model
import joblib

def train_models():
    X, y, vectorizer = preprocess_data('data/sample_transcripts.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    models = get_models()
    tuned_models = {}

    param_grids = {
        "LogisticRegression": {"C": [0.1, 1, 10]},
        "SVM": {"C": [0.1, 1], "kernel": ['linear', 'rbf']},
        "RandomForest": {"n_estimators": [100, 200]}
    }

    for name, model in models.items():
        print(f"Tuning {name}...")
        best_model = tune_model(model, param_grids[name], X_train, y_train)
        tuned_models[name] = best_model
        joblib.dump(best_model, f"models/{name}.pkl")

    joblib.dump(vectorizer, "models/vectorizer.pkl")
    return tuned_models, X_test, y_test
