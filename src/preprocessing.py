import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(data_path, target_column):
    # Load the dataset
    data = pd.read_csv(data_path)
    data = data.reset_index(drop=True)
    data = data.drop(columns=['id'])
    
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
    def encode_categorical(df, categorical_cols):
        for col in categorical_cols:
            df[col] = LabelEncoder().fit_transform(df[col])
        return df

    # Preprocess numerical columns
    X[numerical_cols] = numerical_transformer.fit_transform(X[numerical_cols])
    
    # Preprocess categorical columns
    X = encode_categorical(X, categorical_cols)
    
    # encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X.values, y_encoded
