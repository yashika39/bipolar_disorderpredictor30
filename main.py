from train import train_models
from evaluate import evaluate_models

def main():
    models, X_test, y_test = train_models()
    results = evaluate_models(X_test, y_test)
    for name, metrics in results.items():
        print(f"\n=== {name} ===")
        print("Classification Report:", metrics['report'])
        print("Confusion Matrix:", metrics['confusion_matrix'])
        print("ROC-AUC Score:", metrics['roc_auc'])

if __name__ == '__main__':
    main()
