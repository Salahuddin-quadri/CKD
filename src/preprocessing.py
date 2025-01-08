import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data_path, target_column):
    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Separate the features and the target variable
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    
    
    return X,y

