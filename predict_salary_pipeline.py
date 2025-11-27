#%% imports
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from custom_transformers import RareCategoryGrouper, QuantileClipper, MultiHotEncoder, GenericOrdinalEncoder


def clean_education(X):
    if 'Bachelor' in X:
        return 'Bachelor degree'
    if 'Master' in X:
        return 'Master degree'
    if 'Professional' in X:
        return 'Post grad'
    return 'Less than a Bachelor'


#%% load data
data_folder = r"C:\Users\YanGuo\Documents\predict-salary-streamlit"
data = pd.read_csv(os.path.join(data_folder, 'stackoverflow_salary.csv'))
data.dropna(subset=['ConvertedCompYearly'], inplace=True)

clipper = QuantileClipper(lower=0.05, upper=0.95)
data['ConvertedCompYearly'] = clipper.fit_transform(data['ConvertedCompYearly'])
data['YearsCodePro'] = data['YearsCodePro'] \
    .replace('Less than 1 year', '0.5').replace('More than 50 years', '50').astype(float)
data['OrgSize'] = data['OrgSize'].replace([\
    '2 to 9 employees', '10 to 19 employees', 'Just me - I am a freelancer, sole proprietor, etc.', \
    "I don’t know"], 'Other').fillna('Other')

#%%
country_pipeline = Pipeline(
    steps=[
        ('rare_grouper', RareCategoryGrouper(min_freq=0.01, other_label='Other')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ]
)
years_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('clipper', QuantileClipper(lower=0.01, upper=0.99))
    ]
)
employment_pipeline = Pipeline(
    steps=[
        ('mhe', MultiHotEncoder(delimiter=';'))
    ]
)
org_maps = {'OrgSize': [
    '20 to 99 employees',
    '100 to 499 employees',
    '500 to 999 employees',
    '1,000 to 4,999 employees',
    '5,000 to 9,999 employees',
    '10,000 or more employees',
    'Other']}
organization_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ore', GenericOrdinalEncoder(mappings=org_maps))
    ]
)
age_maps = {'Age': [
    '18-24 years old',
    '25-34 years old',
    '35-44 years old',
    '45-54 years old',
    '55-64 years old',
    'Other']}
age_pipeline = Pipeline(
    steps=[
        ('rare_grouper', RareCategoryGrouper(min_freq=0.01, other_label='Other')),
        ('ore', GenericOrdinalEncoder(mappings=age_maps))
    ]
)

X = data[[
    'Country',
    'YearsCodePro',
    'Employment',
    'OrgSize',
    # 'Age'
]]
y = data['ConvertedCompYearly']

# join the pipelins together
preprocessor = ColumnTransformer(
    transformers=[
        ('country', country_pipeline, ['Country']),
        ('years', years_pipeline, ['YearsCodePro']),
        ('employment', employment_pipeline, ['Employment']),
        ('organization', organization_pipeline, ['OrgSize'])
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

# encoded_columns = pipeline.named_steps['preprocessor'].named_transformers_['age'] \
#     ['ore']
# print(encoded_columns)

rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.2f}")
print(f"MAE: {mae:.2f}")

# %%
joblib.dump(pipeline, os.path.join(data_folder, 'salary_model_pipeline.pkl'))
