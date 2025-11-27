import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def shorten(label, max_len=20):
    return label if len(label) <= max_len else label[:max_len] + "…"


@st.cache_resource
def load_data():
    data = pd.read_csv('salary_data_cleaned.csv')
    return data

def show_explore_page():
    st.title("Explore Stack Overflow SW Developer Salary Data")

    figsize = (10, 5)
    data = load_data()

    st.write("### Salary histogram")
    sns.histplot(data['ConvertedCompYearly'], bins=50, kde=True, color='skyblue')
    plt.title('Distribution of salary in USD')
    plt.xlabel('Salary in USD')
    plt.ylabel('Count')
    st.pyplot(plt)

    st.write("### Country distribution")
    country_counts = data['Country'].value_counts().head(20)
    labels = [shorten(lbl) for lbl in country_counts.index]
    values = country_counts.values
    df_plot = pd.DataFrame({'Country': labels, 'Count': values})
    sns.barplot(x='Country', y='Count', data=df_plot, hue='Country', legend=False)
    plt.xlabel("Countries")
    plt.ylabel("Counts")
    plt.title("Count plot of the top 20 countries")
    plt.xticks(rotation=45, ha='right')   # rotate labels if long
    plt.tight_layout()
    st.pyplot(plt)

    st.write('### Age vs. salary')
    plt.figure(figsize=figsize)
    sns.boxplot(
        data=data,
        x='Age',
        y='ConvertedCompYearly',
        hue='Age',
        legend=False
    )
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)

    st.write('### Country vs. salary')
    top_countries = data['Country'].value_counts().head(8).index
    df_top = data[data['Country'].isin(top_countries)].copy()
    df_top['Country_short'] = df_top['Country'].apply(shorten)
    plt.figure(figsize=figsize)
    sns.violinplot(
        x='Country_short',
        y='ConvertedCompYearly',
        data=df_top,
        inner='quartile',   # shows median and quartiles
        hue='Country_short',
        legend=False
    )
    plt.xlabel("Country")
    plt.ylabel("Salary (Yearly)")
    plt.title("Salary Distribution by Top 5 Countries")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    st.pyplot(plt)
