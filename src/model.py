import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from preprocessing import preprocess_data


def define_and_train_models(X, y):
    """
    Define and train models.
    
    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    
    Returns:
    dict: A dictionary of trained model names and their instances.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "SVM": SVC(probability=True),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(eval_metric='logloss', verbosity=0),
        "CatBoost": CatBoostClassifier(verbose=0),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB()
    }

    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X, y)
        trained_models[name] = model

    print("All models have been trained successfully.\n")
    return trained_models


def save_models(trained_models, models_dir="../models/"):
    """
    Save trained models to the specified directory.
    
    Parameters:
    trained_models (dict): Dictionary of trained model names and instances.
    models_dir (str): Directory to save the models.
    
    Returns:
    None
    """
    # Ensure the directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    for name, model in trained_models.items():
        model_path = os.path.join(models_dir, f"{name.replace(' ', '_')}.pkl")
        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)
        print(f"Saved {name} to {model_path}")

    print("All models have been saved successfully.")


