import streamlit as st
import pandas as pd
from src.data_management import load_data, load_pkl_file
from src.machine_learning.predict_live import predict_attrition


def page_attrition_predictor_body():

    # load predict attrition files
    path = "outputs/ml_pipelines/v3"
    attrition_pipe_dc_fe = load_pkl_file(
        f'{path}/clf_pipeline_data_cleaning_feat_eng.pkl')
    attrition_pipe_model = load_pkl_file(
        f"{path}/clf_pipeline_model.pkl")
    attrition_features = (pd.read_csv(
                          f"{path}/X_train.csv")
                          .columns
                          .to_list()
                          )

    st.write("### Prospect attritionometer Interface")
    st.info(
        f"#### **Business Requirement 2**: Classification Model\n\n"
        f"* The client is interested in using the data to predict predicting"
        f" whether a certain employee will decide to leave the company.\n"
        f"* A machine learning model was built using a binary classification"
        f" model with the following success metrics:\n"
	    f"  * At least 80% precision for no attrition, on train and test set."
        f" Because we want to be sure tha the employee is not intending to"
        f" leave the company.\n"
	    f"  * At least 60% precision for attrition on train and test set."
    )
    st.write("---")

    # Generate Live Data
    X_live = DrawInputsWidgets()

    # predict on live data
    if st.button("Run Predictive Analysis"):
        attrition_prediction = predict_attrition(
            X_live, attrition_features, attrition_pipe_dc_fe,
            attrition_pipe_model)


def DrawInputsWidgets():

    # load dataset
    df = load_data()
    percentageMin, percentageMax = 0.4, 2.0

    # we create input widgets for our 9 best features
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col7, col8, col9 = st.columns(3)

    # We are using these features to feed the ML pipeline
    # feat_selection_vars = ['OverTime', 'YearsAtCompany', 'Age',
    #                       'JobLevel', 'StockOptionLevel', 'Department',
    #                        'JobInvolvement', 'EnvironmentSatisfaction',
    #                        'JobSatisfaction'
    #                        ]

    # create an empty DataFrame, which will be the live data
    X_live = pd.DataFrame([], index=[0])

    # from here on we draw the widget based on the variable type
    # (numerical or categorical) and set initial values
    with col1:
        feature = "OverTime"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique(),
            help='Does the employee work overime?'
        )
    X_live[feature] = st_widget

    with col2:
        feature = "YearsAtCompany"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique(),
            help='How many years did the employee spend at the company?'
        )
    X_live[feature] = st_widget

    with col3:
        feature = "Age"
        st_widget = st.number_input(
            label=feature,
            value=df[feature].median()
        )
    X_live[feature] = st_widget

    with col4:
        feature = "JobLevel"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique(),
            help='Level in the heirarchy'
        )
    X_live[feature] = st_widget

    with col5:
        feature = "StockOptionLevel"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique(),
            help='How many company stocks does the employee own?'
        )
    X_live[feature] = st_widget

    with col6:
        feature = "Department"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
    X_live[feature] = st_widget

    with col7:
        feature = "JobInvolvement"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique(),
            help='1: Not so involved, 4: Highly involved.'
        )
    X_live[feature] = st_widget

    with col8:
        feature = "EnvironmentSatisfaction"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique(),
            help='1: Low, 4: Very high.'
        )
    X_live[feature] = st_widget

    with col9:
        feature = "JobSatisfaction"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique(),
            help='1: Low, 4: Very high.'
        )
    X_live[feature] = st_widget

    return X_live
