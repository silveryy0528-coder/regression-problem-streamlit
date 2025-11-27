import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '..')
from utils.custom_transformers import RareCategoryGrouper


def load_model():
    model = joblib.load('salary_model_pipeline.pkl')
    return model


def show_predict_page():
    st.title("SW Developer Salary Prediction")

    st.write("""### Enter the personal details to predict salary in USD""")
    pipeline = load_model()

    df = pd.read_csv('stackoverflow_salary.csv')
    df['Country'] = df['Country'].fillna('Other')

    country_options = sorted(df['Country'].unique())
    country = st.selectbox("Country", options=country_options)

    years = st.slider("Years of coding", min_value=0.5, max_value=50.0, value=5.0)

    options = pipeline.named_steps['preprocessor'].named_transformers_['employment'] \
        ['mhe'].get_feature_names_out(['Employment'])
    selected = st.multiselect(
        label='Select your employment type',
        options=options,
    )
    employment = ";".join(selected)

    org_options = [
        '20 to 99 employees',
        '100 to 499 employees',
        '500 to 999 employees',
        '1,000 to 4,999 employees',
        '5,000 to 9,999 employees',
        '10,000 or more employees',
        'Other']
    org_size = st.selectbox('Organization size', options=org_options)

    age_options = [
        '18-24 years old',
        '25-34 years old',
        '35-44 years old',
        '45-54 years old',
        '55-64 years old',
        'Other']
    age = st.selectbox('Age', options=age_options)

    ok = st.button("Predict salary")
    if ok:
        input_data = np.array([[country, years, employment, org_size, age]])
        input_df = pd.DataFrame(input_data, columns=['Country', 'YearsCodePro', 'Employment', 'OrgSize', 'Age'])
        pred_salary = pipeline.predict(input_df)
        st.subheader("The predicted salary is: {:.2f} USD".format(pred_salary[0]))
