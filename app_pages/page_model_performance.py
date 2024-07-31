import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from src.data_management import load_pkl_file
from src.machine_learning.evaluate_clf import clf_performance


def page_model_performance_body():

    path = "outputs/ml_pipelines/v3"
    dc_fe_pipeline = load_pkl_file(
        f"{path}/clf_pipeline_data_cleaning_feat_eng.pkl"
    )
    model_pipeline = load_pkl_file(
        f"{path}/clf_pipeline_model.pkl"
    )
    feat_importance = plt.imread(
        f"{path}/features_importance.png"
    )
    X_train = pd.read_csv(
        f"{path}/X_train.csv"
    )
    X_test = pd.read_csv(
        f"{path}/X_test.csv"
    )
    y_train = pd.read_csv(
        f"{path}/y_train.csv"
    )
    y_test = pd.read_csv(
        f"{path}/y_test.csv"
    )

    st.write("### ML Pipeline: Binary Classification")

    st.info(
        f"The model success metrics are:\n"
        f"* At least 80% precision for no-attrition (to predict that"
        f" the employee will stay).\n\n"
        f"* At least 60% precision for attrition (to predict that"
        f" the employee will leave)"
    )

    st.write("---")
    st.write(f"#### ML Pipelines")
    st.write(f"For this model there were 2 ML Pipelines arrange in series:\n")

    st.write(f"* The first pipeline is responsible for data cleaning and\
             feature engineering.\n")
    st.write(dc_fe_pipeline)

    st.write(f"* The second pipeline is responsible for feature scaling and\
             modeling.\n")
    st.write(model_pipeline)

    st.write("---")
    st.write(f"#### Feature Importance")
    st.write(f"* The most important features used for training the model were\
             as follows:\n")
    st.write(X_train.columns.to_list())
    st.image(feat_importance)

    st.write("---")
    st.write(f"#### Model Performance")
    st.success(
        f"The model performed well but could not generalize well to"
        f" predict attrition in  the unseen data"
        f" Therefore, it did not pass all the acceptance criteria."
        f" It scored the following metrics:\n"
        f"* Precision on No-Attrition: 99% and 88% on the train and"
        f" test sets, respectively.\n"
        f"* Precision on Attrition: 99% and 40% on the train and"
        f" test sets, respectively.\n"
        )

    st.write("---")

    clf_performance(X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    pipeline=model_pipeline,
                    label_map=["No Attrition", "Attrition"])
