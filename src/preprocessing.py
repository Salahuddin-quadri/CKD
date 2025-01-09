import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from scipy.sparse import issparse

def preprocess_data(data_path, target_column):
    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Separate the features
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # numerical transformer
    numerical_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)), 
        ('scaler', StandardScaler())  
    ])

    # categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), 
        ('encoder', OneHotEncoder(handle_unknown='ignore'))  
    ])

    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X).toarray()
    
    # encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X_processed, y_encoded
