import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ── Page Configuration ──
st.set_page_config(
    page_title='Patient Triage System',
    page_icon='🏥',
    layout='centered',
    initial_sidebar_state='collapsed'
)

# ── Load Model and Encoder ──
@st.cache_resource
def load_model():
    model = joblib.load('triage_model_with_smote_pipeline.joblib')
    le    = joblib.load('label_encoder.joblib')
    return model, le

model, le = load_model()

# ── Get Feature Columns ──
@st.cache_resource
def get_feature_cols():
    demographic_cols = ['Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']
    all_features     = list(model.named_steps['scaler'].feature_names_in_)
    symptom_cols     = [f for f in all_features if f not in demographic_cols]
    return all_features, symptom_cols

all_features, symptom_cols = get_feature_cols()

# ── Header ──
st.markdown("""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem 0;'>
        <h1 style='font-size:2rem; margin-bottom:0.3rem;'>🏥 Patient Triage System</h1>
        <p style='color:gray; font-size:0.95rem;'>
            Answer the questions below to receive a triage recommendation
        </p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# ── STEP 1: About You ──
# User Inputs
st.sidebar.header('Step 1 — About you')
age = st.sidebar.number_input('Age', 0, 120, 25)
gender = st.sidebar.radio(
    'What is your gender?',
    ['Female', 'Male'],
    horizontal=True
)
blood_pressure = st.sidebar.selectbox('Blood Pressure', ['Normal', 'Low', 'High'])
cholesterol = st.sidebar.selectbox(
    'What is your cholesterol level? (if known)',
    ['Normal', 'High'],
    help='Select Normal if you are unsure'
)

# st.subheader("Step 1 — About you")

# age = st.slider('How old are you?', 0, 100, 30)

# gender = st.radio(
#     'What is your gender?',
#     ['Female', 'Male'],
#     horizontal=True
# )

# blood_pressure = st.selectbox(
#     'What is your blood pressure level? (if known)',
#     ['Normal', 'Low', 'High'],
#     help='Select Normal if you are unsure'
# )

# cholesterol = st.selectbox(
#     'What is your cholesterol level? (if known)',
#     ['Normal', 'High'],
#     help='Select Normal if you are unsure'
# )

st.divider()

# ── STEP 2: How are you feeling ──
st.subheader("Step 2 — How are you feeling right now?")
st.caption("Tick everything that applies. You do not need to know the medical name — just describe what you feel.")

# Core symptoms with plain English labels
fever              = st.checkbox('I have a high temperature or feel feverish')
cough              = st.checkbox('I have a cough')
fatigue            = st.checkbox('I feel unusually tired or exhausted')
difficulty_breathing = st.checkbox('I am having difficulty breathing or feel short of breath')

st.markdown("**Any other symptoms?**")
st.caption("Select as many as apply from the list below.")

# Clean up symptom names for display
def clean_label(s):
    return s.replace('_', ' ').strip().capitalize()

symptom_display = {clean_label(s): s for s in sorted(symptom_cols)}

selected_display = st.multiselect(
    'Additional symptoms:',
    options=list(symptom_display.keys()),
    placeholder='Start typing or scroll to find symptoms...'
)

selected_raw = [symptom_display[d] for d in selected_display]

# Count
core_count    = sum([fever, cough, fatigue, difficulty_breathing])
total_symptoms = core_count + len(selected_raw)

if total_symptoms > 0:
    st.caption(f"You have selected **{total_symptoms}** symptom(s) in total.")

st.divider()

# ── STEP 3: Predict ──
st.subheader("Step 3 — Get your triage recommendation")
st.caption(
    "Based on your symptoms and information, the system will suggest "
    "the most appropriate level of care."
)

predict_btn = st.button(
    'Get My Triage Recommendation',
    use_container_width=True,
    type='primary'
)

# ── Prediction Logic ──
if predict_btn:
    if total_symptoms == 0:
        st.warning("Please select at least one symptom before continuing.")
    else:
        gender_val = 1 if gender == 'Male' else 0
        bp_val     = {'Low': 0, 'Normal': 1, 'High': 2}[blood_pressure]
        chol_val   = 1 if cholesterol == 'High' else 0

        input_data = pd.DataFrame(0, index=[0], columns=all_features)
        input_data['Age']               = age
        input_data['Gender']            = gender_val
        input_data['Blood Pressure']    = bp_val
        input_data['Cholesterol Level'] = chol_val

        if fever and 'fever' in input_data.columns:
            input_data['fever'] = 1
        if cough and 'cough' in input_data.columns:
            input_data['cough'] = 1
        if fatigue and 'fatigue' in input_data.columns:
            input_data['fatigue'] = 1
        if difficulty_breathing and 'difficulty_breathing' in input_data.columns:
            input_data['difficulty_breathing'] = 1

        for symp in selected_raw:
            if symp in input_data.columns:
                input_data[symp] = 1

        prediction   = model.predict(input_data)[0]
        proba        = model.predict_proba(input_data)[0]
        result_label = le.inverse_transform([prediction])[0]
        confidence   = np.max(proba) * 100

        # ── Result Card ──
        st.divider()

        if result_label == 'Emergency':
            color    = '#c0392b'
            bg_color = '#fdecea'
            icon     = '🚨'
            headline = 'Seek emergency care now'
            action   = (
                'Your symptoms suggest you may need urgent medical attention. '
                'Please call **999** or go to your nearest **A&E** immediately. '
                'Do not drive yourself if you feel unwell.'
            )
        elif result_label == 'GP Appointment':
            color    = '#d35400'
            bg_color = '#fef5ec'
            icon     = '📅'
            headline = 'Book a GP appointment'
            action   = (
                'Your symptoms suggest you should see a doctor within the next **1 to 5 days**. '
                'Contact your GP surgery to book an appointment, or use **NHS 111** online '
                'if you cannot get an appointment quickly.'
            )
        else:
            color    = '#1e8449'
            bg_color = '#eafaf1'
            icon     = '🏠'
            headline = 'Self-care at home'
            action   = (
                'Your symptoms suggest you can manage this at home with rest, fluids, '
                'and over-the-counter remedies. If your symptoms worsen or do not improve '
                'within a few days, contact your GP or call **NHS 111**.'
            )

        st.markdown(f"""
            <div style='
                background-color: {bg_color};
                border-left: 6px solid {color};
                border-radius: 0 8px 8px 0;
                padding: 1.4rem 1.6rem;
                margin-bottom: 1rem;
            '>
                <h2 style='color:{color}; margin:0 0 0.5rem 0; font-size:1.5rem;'>
                    {icon} {headline}
                </h2>
                <p style='margin:0; font-size:1rem; line-height:1.6; color:#2c3e50;'>
                    {action}
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Confidence row
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Confidence", f"{confidence:.1f}%")
        with c2:
            st.metric("Symptoms assessed", total_symptoms)
        with c3:
            st.metric("Recommendation", result_label)

        # Probability breakdown
        st.markdown("**How confident is the model in each category?**")
        classes = le.classes_
        prob_df = pd.DataFrame({
            'Triage Level': classes,
            'Probability (%)': [round(p * 100, 2) for p in proba]
        }).sort_values('Probability (%)', ascending=False)

        st.bar_chart(
            prob_df.set_index('Triage Level')['Probability (%)'],
            use_container_width=True
        )

        # Symptoms submitted
        with st.expander("View symptoms you submitted"):
            core_list = []
            if fever: core_list.append('High temperature / fever')
            if cough: core_list.append('Cough')
            if fatigue: core_list.append('Fatigue / tiredness')
            if difficulty_breathing: core_list.append('Difficulty breathing')
            full_list = core_list + selected_display
            for s in full_list:
                st.write(f"• {s}")

        st.divider()
        st.caption(
            "⚠️ This recommendation is generated by a machine learning model "
            "trained on synthetic healthcare data. It is for decision-support only "
            "and does not replace professional medical advice."
        )

# ── About This Model ──
st.divider()
with st.expander("ℹ️ How accurate is this tool?"):
    st.markdown("#### Model performance")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Accuracy", "87.77%")
    with c2:
        st.metric("Macro F1", "0.8798")
    with c3:
        st.metric("CV F1", "0.8414")
    with c4:
        st.metric("Stability", "±0.0073")

    st.markdown("""
    This tool uses a **Random Forest** machine learning model trained on over 8,400
    patient records. Three models were compared — Logistic Regression (62%),
    XGBoost (85%), and Random Forest (87.77%). Random Forest was selected as the
    most accurate and stable.

    The most important safety finding: the model **never** misclassified an Emergency
    case as Self-Care. When uncertain, it defaults to the middle category (GP Appointment),
    which is the safer direction.
    """)

    st.caption(
        "Developed for COM 763 Advanced Machine Learning — Wrexham University. "
        "Research prototype only — not validated for clinical use."
    )

# ── Footer ──
st.divider()
st.markdown("""
    <div style='text-align:center; color:gray; font-size:0.8rem; padding-bottom:1rem;'>
        COM 763 Advanced Machine Learning — Wrexham University |
        Model: Random Forest | Accuracy: 87.77% | Macro F1: 0.8798
    </div>
""", unsafe_allow_html=True)