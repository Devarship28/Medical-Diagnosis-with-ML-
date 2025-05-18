from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=500),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    return {'report': report, 'confusion_matrix': cm, 'roc_auc': roc_auc}

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)