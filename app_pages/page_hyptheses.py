import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.data_management import load_data
sns.set_style("whitegrid")

def page_hypotheses_body():

    # load data
    df = load_data()
    
    # load train dataset
    def load_train():
        X_train = pd.read_csv(
            f"outputs/ml_pipelines/v3/X_train.csv")
        return X_train
    
    features_importance = plt.imread(
        f"outputs/ml_pipelines/v3/features_importance.png")

    st.write(
        f"* [Hypothesis 1](#hypothesis-1)\n"
        f"* [Hypothesis 2](#hypothesis-2)\n"
        f"* [Hypothesis 3](#hypothesis-3)\n"
    )

    st.write("### Hypothesis 1")
    st.warning(
        "* We suspect that monthly income is the main feature"
        " to predict attrition."
    )
    st.success(
        f" * According to the correlation analysis, MonthlyIncome and"
        f" Attrition are slightly correlated.\n"
        f" * According to the feature"
        f" selection algorithm in the ML pipeline, MonthlyIncome is not"
        f" a main feeature to predict Attrition."
        f" * A simple histogram plot (toggle) shows that they are"
        f" indeed correlated but only to a certain level of income,"
        f" which is around 10k, after which the correlation is lost."
    )

    # inspect data
    if st.checkbox("Inspect MonthyIncome-Attrition relation"):
        show_plot(load_data(), 'MonthlyIncome', 'Attrition')

    st.write("---")

    st.write("### Hypothesis 2")
    st.warning(
        "* We suspect that men tend to leave the workforce"
        " more often than women."
    )
    st.success(
        " * That was found out to be false. Men and women show very similar"
        " almost equivalent attrition and no-attrition rates. If we"
        " normalizes their count in the dataset, we find that the"
        " ratio of a woman leaving the workforce from the total women"
        " subset is the same as for men."
        " * Other findings from correlation analysis and feature selection"
        " show that gender is not in anyway correlated to attrition."
    )

    # inspect data
    if st.checkbox("Inspect Gender-Attrition relation"):
        show_plot(load_data(), 'Gender', 'Attrition', 
                  stat="probability", multiple="stack")

    st.write("---")

    st.write("### Hypothesis 3")
    st.warning(
        "* We suspect that only few features affect attrition"
    )
    st.success(
        " This was found out to be relatively true. According to"
        " feature selection in the ML pipeline, only 9 variables"
        " out of the 34 are good predictors of attrition"
    )
    
    # inspect data
    if st.checkbox("Inspect most important features\
                   (highest to lowest)"):
        st.write(load_train().columns.to_list())
        st.image(features_importance)


def show_plot(df, col, target_var, **kwargs):
    fig, axes = plt.subplots(figsize=(12, 5))
    sns.histplot(data=df, x=col, hue=target_var, **kwargs)
    plt.title(f"{col}-{target_var}", fontsize=20, y=1.05)
    st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()