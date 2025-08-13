import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class CustomCategoricalEncoder(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            X[col] = X[col].fillna('missing')
            if col in ['rbc', 'pc']:
                X[col] = X[col].apply(lambda x: 1 if x.lower() == 'normal' else 0)
            elif col in ['pcc', 'ba']:
                X[col] = X[col].apply(lambda x: 1 if x.lower() == 'present' else 0)
            elif col == 'appet':
                X[col] = X[col].apply(lambda x: 1 if x.lower() == 'good' else 0)
            elif col in ['htn', 'dm', 'cad', 'pe', 'ane']:
                X[col] = X[col].apply(lambda x: 1 if x.lower() == 'yes' else 0)
            else:
                X[col] = X[col].astype('category').cat.codes
        return X

def preprocess_data(data_path, target_column):
    # Load the dataset
    data = pd.read_csv(data_path)
    data = data.reset_index(drop=True)
    data = data.drop(columns=['id'], errors='ignore')
    data = data.dropna(subset=[target_column])

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Define numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Define preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('encoder', CustomCategoricalEncoder())
    ])

    # Combine transformers into a single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Create the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('target_encoder', FunctionTransformer(lambda y: y.apply(lambda x: 0 if x == "notckd" else 1), validate=False))
    ])

    # Apply pipeline
    X_processed = pipeline.named_steps['preprocessor'].fit_transform(X)
    y_encoded = pipeline.named_steps['target_encoder'].transform(y)

    return X_processed, y_encoded
