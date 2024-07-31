import streamlit as st


def predict_attrition(X_live, attrition_features, attrition_pipeline_dc_fe,
                      attrition_pipeline_model):

    # from live data, subset features related to this pipeline
    X_live_attrition = X_live.filter(attrition_features)

    # apply data cleaning / feat engine pipeline to live data
    X_live_attrition_dc_fe = attrition_pipeline_dc_fe\
        .transform(X_live_attrition)

    # predict
    attrition_prediction = attrition_pipeline_model\
        .predict(X_live_attrition_dc_fe)
    attrition_prediction_proba = attrition_pipeline_model.predict_proba(
        X_live_attrition_dc_fe)

    # Create a logic to display the results
    attrition_prob = attrition_prediction_proba[0, attrition_prediction][0]*100
    if attrition_prediction == 1:
        attrition_result = 'will leave'
    else:
        attrition_result = 'will not leave'

    statement = (
        f'### There is {attrition_prob.round(1)}% probability '
        f'that this employee **{attrition_result} the company**.')

    st.write(statement)

    return attrition_prediction
