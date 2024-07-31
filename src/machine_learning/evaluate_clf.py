import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def confusion_matrix_and_report(X, y, pipeline, label_map):
    """
    Display the confusion matrix and report
    """
    prediction = pipeline.predict(X)

    st.write('#### Confusion Matrix')
    st.code(pd.DataFrame(confusion_matrix(y_true=prediction, y_pred=y),
                         columns=[["Actual " + sub for sub in label_map]],
                         index=[["Prediction " + sub for sub in label_map]]
                         ))

    st.write('#### Classification Report')
    st.code(classification_report(y, prediction, target_names=label_map), "\n")


def clf_performance(X_train, y_train, X_test, y_test, pipeline, label_map):
    """
    Pretty printing function for the confusion_matrix_and_report function
    """
    st.info("Train Set")
    confusion_matrix_and_report(X_train, y_train, pipeline, label_map)

    st.info("Test Set")
    confusion_matrix_and_report(X_test, y_test, pipeline, label_map)
