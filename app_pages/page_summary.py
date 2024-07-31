import streamlit as st
import pandas as pd
from src.data_management import load_data

# load data
df = load_data()


def page_summary_body():

    st.write(
        "* [Project Summary](#project-summary)\n"
        "* [Project Dataset](#project-dataset)\n"
        "* [Business Requirements](#business-requirements)"
    )

    st.write("### Project Summary")

    st.info(
        f" The attrition predictor predicts whether an employee will"
        f" remain in the workforce according to multiple factors like"
        f" demographics, work culture, etc. Attrition in this context"
        f" could be voulantary as well as involuntary leave from an"
        f" organization for unpredictable or uncontrollable reasons."
        f" With a high attrition rate, a company is likely to shrink in size."
        f" \n\nEmployee attrition leads to significant costs for a business,"
        f" including the cost of business disruption, hiring and training"
        f" new staff. Therefore, there is great business interest in"
        f" understanding the drivers of, and minimizing staff attrition."
        f" Managing and understanding attrition is pivotal for organizations"
        f" to ensure a stable and engaged workforce."
        f" containing individual customer data on the products and services"
        f" (like internet type, online security, online backup, tech support),"
        f" account information (like contract type, payment method,"
        f" monthly charges) and profile (like gender, partner, dependents)."
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

    st.dataframe(load_data())

    # Link to README file, so the users can have acces
    # to full project documentation
    st.write(
        f"* For explanation of variable meanings and further info"
        f"on the dataset please visit and **read** the "
        f"[Project README file]"
        f"(https://github.com/mtelewa/attrition-predictor)."
    )

    # copied from README file - "Business Requirements" section
    st.success(
        " The project has 2 business requirements:\n"
        " 1. The client is interested in understanding the main factors"
        " leading to attrition\n "
        " 2. The client is interested in predicting whether a certain"
        " employee will decide to leave the company."
    )
