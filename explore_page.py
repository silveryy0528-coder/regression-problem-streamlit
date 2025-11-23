import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


@st.cache_resource
def load_data():
    data_folder = r"C:\Users\YanGuo\Documents\predict-salary-streamlit"
    data = pd.read_csv(os.path.join(data_folder, 'vehicle_emissions.csv'))
    return data


def show_explore_page():
    st.title("Explore Vehicle Emissions Data")

    data = load_data()
    st.write("### Dataset Overview")
    st.dataframe(data.head())

    figsize = (8, 5)

    st.write("### Distribution of CO2 Emissions")
    plt.figure(figsize=figsize)
    sns.histplot(data['CO2_Emissions'], bins=50, kde=True, color='skyblue')
    plt.title('Distribution of CO2 Emissions')
    plt.xlabel('CO2 Emissions')
    plt.ylabel('Count')
    st.pyplot(plt)

    plt.figure(figsize=figsize)
    sns.countplot(
        y='Transmission', data=data, order=data['Transmission'].value_counts().index,
        hue='Transmission', legend=False)
    plt.title(f'Number of respondents by transmission type')
    plt.xlabel('Count')
    plt.ylabel(f'Transmission Type')
    st.pyplot(plt)

    plt.figure(figsize=figsize)
    sns.violinplot(
        data=data,
        x='Vehicle_Class',
        y='CO2_Emissions',
        inner=None,
        cut=0,
        palette='pastel',
    )
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)

