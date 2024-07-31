import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ppscore as pps
from feature_engine.encoding import OneHotEncoder
from src.data_management import load_data
sns.set_style("whitegrid")

# load data and transform attrition
df = load_data()
df['Attrition'] = df['Attrition'].replace({"Yes":1, "No":0})

vars_to_consider = ['TotalWorkingYears', 'YearsAtCompany',
                    'YearsWithCurrManager', 'OverTime',
                    'YearsInCurrentRole', 'JobLevel',
                    'MaritalStatus', 'JobRole', 'Age',
                    'MonthlyIncome', 'StockOptionLevel']

df_eda = df.filter(list(vars_to_consider))
df_eda['Attrition'] = df['Attrition']

def encode_df():
    """
    one hot encoder to transform data from categorical to numerical
    """
    encoder = OneHotEncoder(variables=df.columns[df.dtypes=='object'].to_list(), drop_last=True)
    df_ohe = encoder.fit_transform(df)
    return df_ohe


def correlation(method):
    """
    perform the correlation
    args: correlation method
    returns: variables strongly correlated to attrition
    """
    print(encode_df())
    corr = encode_df().corr(method=method)['Attrition']\
           .sort_values(key=abs, ascending=False)[1:].head(10)
    df_corr = corr.to_frame()
    corr_variables = df_corr.index.to_list()
    return corr_variables


def heatmap_corr(matrix):
    """
    plots the heatmap - this function was adopted from EDA tools lesson
    """
    fig, ax = plt.subplots(figsize=(16,16))
    heatmap = sns.heatmap(matrix, annot=True, annot_kws={"size": 6},
                    cmap='rocket_r', linecolor='lightgrey')
    heatmap.set(xlabel=None, ylabel=None)
    st.pyplot(fig)


def plot_categorical(df, col, target_var):
    fig, axes = plt.subplots(figsize=(12, 5))
    sns.countplot(data=df, x=col, hue=target_var,
                  order=df[col].value_counts().index)
    plt.xticks(rotation=90)
    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)


def plot_numerical(df, col, target_var):
    fig, axes = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x=col, hue=target_var, kde=True, element="step")
    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)



def page_correlation_body():

    # hard copied from churned customer study notebook
    vars_to_study = ['Contract', 'InternetService',
                    'OnlineSecurity', 'TechSupport', 'tenure']

    st.write("### Feature Correlation Analysis")

    st.write(
        f"* [Business Requirement and Dataset](#business-requirement-1-data-visualisation-and-correlation-study)\n"
        f"* [Correlation Analysis](#correlation-analysis)\n"
        f"* [Strongest Correlated Features](#strongest-correlated-features)\n"
        f"* [Conclusions](#conclusions)\n"
    )

    st.info(
        f"#### **Business Requirement 1**: Understanding the main"
        f"factors leading to attrition\n\n"
        f"* Approach: perform a correlation study to determine which"
        f" features correlate strongly with the target.\n"
    )

    st.write("---")


    st.write("#### Correlation Analysis")
    st.write(
        f"* Correlations were measured with Pearson's method that "
        " indicates linear relationships between numerical variables.\n"
        f"* In addition to Spearman's method which measures the monotonic"
        " relationships between variables.\n"
        f"* A Predictive power score (PPS) study was also performed"
        " to measure the prediction power between attributes.\n"
    )

    if st.checkbox("Attrition strongest correlators - Pearson method"):
        st.write(correlation('pearson'))

    if st.checkbox("Attrition strongest correlators - Spearman method"):
        st.write(correlation('spearman'))

    st.write(
        f"Since they agreed on almost all of the features, we built a set"
        " from those features and performed the PPS analysis on those"
        " features. Here are the results."
    )

    if st.checkbox("PPS heatmap"):
        pps_matrix_raw = pps.matrix(df_eda)
        pps_matrix = pps_matrix_raw.filter(['x', 'y', 'ppscore'])\
                     .pivot(columns='x', index='y', values='ppscore')
        heatmap_corr(pps_matrix)

    st.write("---")

    st.write("#### Strongest Correlated Features")
    st.write(
        f"* We can now show how attrition correlates with each variable"
        " and realize the first business requirement.\n"
    )

    feature = st.selectbox(
        "Select feature to view:", (vars_to_consider)
    )

    if df_eda[feature].dtype == 'object':
        plot_categorical(df_eda, feature, 'Attrition')
    else:
        plot_numerical(df_eda, feature, 'Attrition')


    st.success(
        f"#### Conclusions\n\n"
        f" Unfortunately, from the correlation analysis alone, it"
        f" was not very clear how the strongly correlated features really"
        f" affect attrition. As can be seen, the distrubution of both attrition"
        f" and no-attrition are very close.\n\n"
        f" One specific feature, however, stands out"
        f" as clearly negatively correlated to attrition, which is `OverTime`"
        f" The more an employee works overtime, the more they are prone to "
        f" leave the company."
    )