import streamlit as st

strong_correlators = ['TotalWorkingYears', 'YearsAtCompany',
                       'YearsWithCurrManager', 'OverTime',
                       'YearsInCurrentRole', 'JobLevel',
                       'MaritalStatus', 'JobRole', 'Age',
                       'MonthlyIncome', 'StockOptionLevel'
                     ]

feat_selection_vars = ['OverTime', 'YearsAtCompany', 'Age',
                       'JobLevel', 'StockOptionLevel', 'Department',
                       'JobInvolvement', 'EnvironmentSatisfaction',
                       'JobSatisfaction'
                       ]

def intersection(lst1, lst2):
    """
    From: https://www.geeksforgeeks.org/python-intersection-two-lists/
    """
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

common_vars = intersection(strong_correlators, feat_selection_vars)

def page_conclusion_body():

    st.write("### Project Conclusions")

    st.success(
        f"#### Business Requirements\n\n"
        f"*Business Requirement 1* - This requirement was met through"
        f" the use of an exploratory data analysis and various correlations"
        f" and plots. The correlations did not show clearly how the features"
        f" affect attrition. However, we see that variables strongly"
        f" correlated to attrition agree to a great extent with the variables"
        f" from the feature selection step in the ML pipeline."
        f" The variables that showed stron correlatoion and "
        f" were seleected as important features are: {common_vars}\n\n"
        f"*Business Requirement 2* - This requirement was met through"
        f" the use of a ML binary classification model.\n"
        f"* For precision on no-attrition, that is an employee will"
        f" stay in the compay, the model scored"
        f" 99% and 88% on the train and test datasets, respectively.\n"
        f"* For precision on attrition, that is an employee will leave"
        f" the company, the model scored 99% and 88% on the"
        f" train and test datasets, respectively."
    )

    st.info(
        f"#### Project Outcomes\n\n"
        f" It is clear that the model can stillbe further trained and"
        f" overfitting can be minimized, such that the model"
        f" to perform better on the unseen data. However, the dataset itself"
        f" could be difficult to extract data from and solve the"
        f"  classification problem. That is clear from the weak correlaions"
        f" of many variables withthe target. \n\n"
        f" One can still predict with high precision that an employee"
        f" is likely to stay in the company, which is a useful metric"
        f" to reflect on the general employee conditions."
    )