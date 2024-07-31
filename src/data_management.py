import streamlit as st
import pandas as pd
import joblib


# load data
@st.cache_data
def load_data():
    df = pd.read_csv(
        "outputs/datasets/collection/employee-attrition.csv")
    return df


def load_pkl_file(file_path):
    return joblib.load(filename=file_path)