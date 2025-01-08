import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

def preprocess_data(data_path, target_column):
    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Separate the features and the target variable
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Define the numerical transformer
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Fill NaN with mean
        ('scaler', StandardScaler())  # Standardize numerical features
    ])

    # Define the categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill NaN with mode
        ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
    ])

    # Combine both transformers into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)

    # If the target variable is categorical, encode it
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        y_encoded = y  # If it's already numerical, keep it as is

    return X_processed, y_encoded

# Example usage:
# X, y = preprocess_data('path/to/your/data.csv', 'classification')

