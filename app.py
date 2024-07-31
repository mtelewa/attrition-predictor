import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_hyptheses import page_hypotheses_body
from app_pages.page_correlation_analysis import page_correlation_body
from app_pages.page_attrition_predictor import page_attrition_predictor_body
from app_pages.page_model_performance import page_model_performance_body
from app_pages.page_project_summary import page_project_summary_body

app = MultiPage(app_name= "Attrition Predictor") # Create an instance of the app 

# Add your app pages here using .add_page()
app.add_page("Project Summary", page_summary_body)
app.add_page("Project Hypotheses", page_hypotheses_body)
app.add_page("Project Correlation Analysis", page_correlation_body)
app.add_page("Attrition Predictor", page_attrition_predictor_body)
app.add_page("Model Performance", page_model_performance_body)
app.add_page("Project Summary", page_project_summary_body)

app.run() # Run the app
