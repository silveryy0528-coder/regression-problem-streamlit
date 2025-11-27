#%% imports
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from custom_transformers import RareCategoryGrouper


#%% load data
data_folder = r"C:\Users\YanGuo\Documents\predict-salary-streamlit"
data = pd.read_csv(os.path.join(data_folder, 'vehicle_emissions.csv'))

# %% create features and target
X = data.drop(columns=['CO2_Emissions', 'Model_Year', 'Model'])
y = data['CO2_Emissions']

# split data into numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
make_rare = ['Make']
transmission_rare = ['Transmission']
categorical_cols = ['Vehicle_Class']

# %% start the pipeline with encoding
numerical_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)
categorical_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ]
)
categorical_rare_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('rare_grouper', RareCategoryGrouper(min_freq=0.01, other_label='Other')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ]
)

# join the pipelins together
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),    # name, pipeline, columns
        ('cat', categorical_pipeline, categorical_cols),
        ('make_rare', categorical_rare_pipeline, make_rare),
        ('transmission_rare', categorical_rare_pipeline, transmission_rare)
    ]
)

pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ]
)

# %% split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# check what is encoded
encoded_columns = pipeline.named_steps['preprocessor'].named_transformers_['make_rare'] \
    ['ohe'].get_feature_names_out(make_rare)

rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.2f}")
print(f"MAE: {mae:.2f}")

# %%
joblib.dump(pipeline, os.path.join(data_folder, 'emission_model_pipeline.pkl'))

# %%
