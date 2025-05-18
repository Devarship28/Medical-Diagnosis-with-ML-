import os
from src.data_preprocessing import load_data, preprocess_data, train_val_test_split
from src.modeling import train_models, evaluate_model, save_model

def main():
    data_path = "data/diabetes.csv"
    target_col = "Outcome"
    model_dir = "models"
    model_path = os.path.join(model_dir, "best_model.joblib")

    # Create models directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Load and preprocess data
    df = load_data(data_path)
    X, y = preprocess_data(df, target_col)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    # Train models
    models = train_models(X_train, y_train)

    # Evaluate models and select best based on roc_auc
    best_model = None
    best_score = -1
    for name, model in models.items():
        metrics = evaluate_model(model, X_val, y_val)
        roc_auc = metrics['roc_auc'] if metrics['roc_auc'] is not None else 0
        print(f"{name} ROC AUC: {roc_auc}")
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = model

    if best_model is not None:
        save_model(best_model, model_path)
        print(f"Best model saved to {model_path}")
    else:
        print("No best model found to save.")

if __name__ == "__main__":
    main()
