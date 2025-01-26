import pandas as pd
import os
import sys

sys.path.append('c:\\Users\\salah\\CKD\\src')

from model import define_and_train_models , save_models
from preprocessing import preprocess_data

DATA_PATH = '../data/raw.csv'
TARGET_COLUMN = 'classification' 

# Preprocessing
X, y = preprocess_data(DATA_PATH , TARGET_COLUMN)

print(y)

#train the models
trained_models = define_and_train_models(X, y)

#save the models
save_models(trained_models, models_dir="../models/")