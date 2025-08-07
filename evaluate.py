from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import pandas as pd

class_names = ['bipolar', 'schizophrenia', 'healthy']

def evaluate_models(X_test, y_test):
    results = {}
    for name in ['LogisticRegression', 'SVM', 'RandomForest']:
        model = joblib.load(f"models/{name}.pkl")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=class_names)
        roc = roc_auc_score(pd.get_dummies(y_test), y_proba, multi_class='ovr')
        results[name] = {"report": report, "confusion_matrix": cm.tolist(), "roc_auc": roc}
    return results
