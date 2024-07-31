import streamlit as st
import pandas as pd


dataset = pd.read_csv(
    f"outputs/datasets/collection/employee-attrition.csv").head(5)

variables = []

def page_summary_body():

    st.write(
        "* [Project Summary](#project-summary)\n"
        "* [Project Dataset](#project-dataset)\n"
        "* [Business Requirements](#business-requirements)"
    )

    st.write("### Project Summary")

    st.info(
        " The attrition predictor predicts whether an employee will"
        " remain in the workforce according to multiple factors like"
        " demographics, work culture, etc. Attrition in this context"
        " could be voulantary as well as involuntary leave from an"
        " organization for unpredictable or uncontrollable reasons."
        " With a high attrition rate, a company is likely to shrink in size."
        " Employee attrition leads to significant costs for a business,"
        " including the cost of business disruption, hiring and training"
        " new staff. Therefore, there is great business interest in"
        " understanding the drivers of, and minimizing staff attrition."
        " Managing and understanding attrition is pivotal for organizations"
        " to ensure a stable and engaged workforce."
        " containing individual customer data on the products and services"
        " (like internet type, online security, online backup, tech support),"
        " account information (like contract type, payment method," 
        " monthly charges) and profile (like gender, partner, dependents)."
        )

    st.write("### Project Dataset")
        
    st.info(
        " The dataset can be found on "
        " [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)"
        " and it consists of 1470 rows and 35 columns i.e. a total of"
        " 51450 observations. 9 columns are categorical of Object (or string)"
        " type while the rest (26 columns) are numerical of integer type."
        " The following summary was obtained from `ProfileReport` imported"
        " from `ydata_profiling` library."
    )

    st.write("The dataset looks like this")

    st.dataframe(dataset)

    # Link to README file, so the users can have access to full project documentation
    st.write(
        f"* For explanation of variable meanings and further info"
        f"on the dataset please visit and **read** the "
        f"[Project README file]"
        f"(https://github.com/mtelewa/attrition-predictor).")
    

    # copied from README file - "Business Requirements" section
    st.success(
        " The project has 2 business requirements:\n"
        " 1. The client is interested in understanding the main factors"
        " leading to attrition "
        " 2. The client is interested in predicting whether a certain"
        " employee will decide to leave the company."
        )

        