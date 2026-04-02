import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved assets
with open('triage_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

st.set_page_config(page_title='Patient Triage AI', page_icon='⚕️')
st.title('⚕️ Patient Triage System')

# User Inputs
st.sidebar.header('Patient Profile')
age = st.sidebar.number_input('Age', 0, 120, 25)
bp = st.sidebar.selectbox('Blood Pressure', ['Low', 'Normal', 'High'])
chol = st.sidebar.selectbox('Cholesterol', ['Normal', 'High'])

# Pre-process inputs
bp_map = {'Low': 0, 'Normal': 1, 'High': 2}
chol_val = 1 if chol == 'High' else 0

st.subheader('Select Symptoms')
all_features = model.feature_names_in_
symptom_cols = [c for c in all_features if c not in ['Age', 'Blood Pressure', 'Cholesterol Level'] and c != 'Gender']
selected = st.multiselect('Symptoms', symptom_cols)

if st.button('Run Triage Prediction'):
    input_df = pd.DataFrame(0, index=[0], columns=all_features)
    input_df.loc[0, ['Age', 'Blood Pressure', 'Cholesterol Level']] = [age, bp_map[bp], chol_val]
    for s in selected: input_df.loc[0, s] = 1
    
    pred = model.predict(input_df)[0]
    label = le.inverse_transform([pred])[0]
    conf = np.max(model.predict_proba(input_df)) * 100
    
    color = 'red' if label == 'Emergency' else 'orange' if label == 'GP Appointment' else 'green'
    st.markdown(f'### Priority: :{color}[{label}]')
    st.write(f'Confidence Score: {conf:.1f}%')