from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def get_models():
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "RandomForest": RandomForestClassifier()
    }
    return models

def tune_model(model, param_grid, X_train, y_train):
    grid = GridSearchCV(model, param_grid, cv=5, scoring='f1_macro')
    grid.fit(X_train, y_train)
    return grid.best_estimator_
