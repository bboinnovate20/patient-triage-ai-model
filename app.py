import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ── Page Configuration ──
st.set_page_config(
    page_title='Patient Triage System',
    page_icon='🏥',
    layout='wide',
    initial_sidebar_state='expanded'
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
    return all_features, symptom_cols, demographic_cols

all_features, symptom_cols, demographic_cols = get_feature_cols()

# ── Symptom Groups for UI ──
respiratory_symptoms = [s for s in symptom_cols if any(k in s.lower() for k in [
    'cough', 'breath', 'chest', 'wheez', 'mucus', 'phlegm', 'throat',
    'sinus', 'nasal', 'runny', 'congestion', 'asthma', 'pneumon'
])]
gastrointestinal_symptoms = [s for s in symptom_cols if any(k in s.lower() for k in [
    'nausea', 'vomit', 'abdomen', 'abdominal', 'diarrhoe', 'stomach',
    'gastro', 'bowel', 'indigestion', 'acidity', 'ulcer', 'appetite'
])]
neurological_symptoms = [s for s in symptom_cols if any(k in s.lower() for k in [
    'headache', 'dizz', 'migraine', 'seizure', 'paralys', 'confusion',
    'memory', 'altered', 'sensori', 'stiff_neck', 'tremors', 'anxiety'
])]
skin_symptoms = [s for s in symptom_cols if any(k in s.lower() for k in [
    'rash', 'itch', 'skin', 'blister', 'peeling', 'pus', 'dischrom',
    'nodal', 'eruption', 'acne', 'bruising'
])]
general_symptoms = [s for s in symptom_cols if s not in
    respiratory_symptoms + gastrointestinal_symptoms +
    neurological_symptoms + skin_symptoms
]

# ── Header ──
st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <h1 style='font-size:2.2rem; margin-bottom:0.2rem;'>🏥 Patient Triage System</h1>
        <p style='color:gray; font-size:1rem;'>
            AI-powered symptom assessment to guide appropriate level of care
        </p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# ── Disclaimer Banner ──
st.warning(
    "⚠️ **Important:** This tool is for decision-support purposes only. "
    "It does not replace professional medical advice. "
    "If you are experiencing a life-threatening emergency, call **999** immediately."
)

st.divider()

# ── Layout: Two Columns ──
left_col, right_col = st.columns([1, 2])

# ── LEFT: Demographics ──
with left_col:
    st.subheader("👤 Patient Demographics")

    age = st.slider('Age', 0, 100, 30, help='Patient age in years')

    gender = st.selectbox('Gender', ['Female', 'Male'])

    blood_pressure = st.selectbox(
        'Blood Pressure',
        ['Low', 'Normal', 'High'],
        help='Patient reported or measured blood pressure level'
    )

    cholesterol = st.selectbox(
        'Cholesterol Level',
        ['Normal', 'High'],
        help='Patient reported or measured cholesterol level'
    )

    st.divider()

    # Core symptoms always shown
    st.subheader("🌡️ Core Symptoms")
    fever              = st.checkbox('Fever')
    cough              = st.checkbox('Cough')
    fatigue            = st.checkbox('Fatigue')
    difficulty_breathing = st.checkbox('Difficulty Breathing')

    # Symptom count indicator
    core_count = sum([fever, cough, fatigue, difficulty_breathing])
    st.caption(f"Core symptoms selected: **{core_count}**")

# ── RIGHT: Symptom Groups ──
with right_col:
    st.subheader("🩺 Additional Symptoms")
    st.caption("Select all symptoms the patient is currently experiencing.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🫁 Respiratory",
        "🫃 Gastrointestinal",
        "🧠 Neurological",
        "🧴 Skin",
        "⚡ General"
    ])

    with tab1:
        if respiratory_symptoms:
            resp_selected = st.multiselect(
                'Respiratory symptoms:',
                sorted(respiratory_symptoms),
                key='resp'
            )
        else:
            resp_selected = []
            st.info('No respiratory symptoms found in feature set.')

    with tab2:
        if gastrointestinal_symptoms:
            gi_selected = st.multiselect(
                'Gastrointestinal symptoms:',
                sorted(gastrointestinal_symptoms),
                key='gi'
            )
        else:
            gi_selected = []
            st.info('No gastrointestinal symptoms found in feature set.')

    with tab3:
        if neurological_symptoms:
            neuro_selected = st.multiselect(
                'Neurological symptoms:',
                sorted(neurological_symptoms),
                key='neuro'
            )
        else:
            neuro_selected = []
            st.info('No neurological symptoms found in feature set.')

    with tab4:
        if skin_symptoms:
            skin_selected = st.multiselect(
                'Skin symptoms:',
                sorted(skin_symptoms),
                key='skin'
            )
        else:
            skin_selected = []
            st.info('No skin symptoms found in feature set.')

    with tab5:
        if general_symptoms:
            gen_selected = st.multiselect(
                'General symptoms:',
                sorted(general_symptoms),
                key='gen'
            )
        else:
            gen_selected = []
            st.info('No general symptoms found in feature set.')

    # Total symptom count
    all_selected = (
        resp_selected + gi_selected +
        neuro_selected + skin_selected + gen_selected
    )
    total_symptoms = core_count + len(all_selected)
    st.caption(f"Total symptoms selected: **{total_symptoms}**")

    st.divider()

    # ── Predict Button ──
    predict_btn = st.button(
        'Predict Triage Level',
        use_container_width=True,
        type='primary'
    )

# ── Prediction Logic ──
if predict_btn:
    if total_symptoms == 0:
        st.error("Please select at least one symptom before predicting.")
    else:
        # Encode demographics
        gender_val = 1 if gender == 'Male' else 0
        bp_val     = {'Low': 0, 'Normal': 1, 'High': 2}[blood_pressure]
        chol_val   = 1 if cholesterol == 'High' else 0

        # Build input vector
        input_data = pd.DataFrame(0, index=[0], columns=all_features)
        input_data['Age']              = age
        input_data['Gender']           = gender_val
        input_data['Blood Pressure']   = bp_val
        input_data['Cholesterol Level'] = chol_val

        # Core symptoms
        if fever and 'fever' in input_data.columns:
            input_data['fever'] = 1
        if cough and 'cough' in input_data.columns:
            input_data['cough'] = 1
        if fatigue and 'fatigue' in input_data.columns:
            input_data['fatigue'] = 1
        if difficulty_breathing and 'difficulty_breathing' in input_data.columns:
            input_data['difficulty_breathing'] = 1

        # Additional symptoms
        for symp in all_selected:
            if symp in input_data.columns:
                input_data[symp] = 1

        # Predict
        prediction   = model.predict(input_data)[0]
        proba        = model.predict_proba(input_data)[0]
        result_label = le.inverse_transform([prediction])[0]
        confidence   = np.max(proba) * 100

        # ── Result Display ──
        st.divider()
        st.subheader("📋 Triage Result")

        # Colour and icon by urgency
        if result_label == 'Emergency':
            color     = '#e74c3c'
            bg_color  = '#fdecea'
            
        elif result_label == 'GP Appointment':
            color     = '#e67e22'
            bg_color  = '#fef9f0'
            
        else:
            color     = '#27ae60'
            bg_color  = '#eafaf1'
            

        # Result card
        st.markdown(f"""
            <div style='
                background-color: {bg_color};
                border-left: 6px solid {color};
                border-radius: 8px;
                padding: 1.2rem 1.5rem;
                margin-bottom: 1rem;
            '>
                <h2 style='color:{color}; margin:0 0 0.4rem 0;'>
                    {result_label}
                </h2>
                
            </div>
        """, unsafe_allow_html=True)

        # Confidence and metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence", f"{confidence:.1f}%")
        with col2:
            st.metric("Symptoms Assessed", total_symptoms)
        with col3:
            st.metric("Triage Level", result_label)

        # ── Probability Breakdown ──
        st.subheader("📊 Class Probabilities")
        classes = le.classes_
        prob_df = pd.DataFrame({
            'Triage Level': classes,
            'Probability (%)': [round(p * 100, 2) for p in proba]
        }).sort_values('Probability (%)', ascending=False)

        st.dataframe(
            prob_df,
            use_container_width=True,
            hide_index=True
        )

        # Probability bar chart
        st.bar_chart(
            prob_df.set_index('Triage Level')['Probability (%)'],
            use_container_width=True
        )

        # ── Symptoms Summary ──
        st.subheader("📝 Symptoms Submitted")
        core_list = []
        if fever: core_list.append('fever')
        if cough: core_list.append('cough')
        if fatigue: core_list.append('fatigue')
        if difficulty_breathing: core_list.append('difficulty_breathing')

        full_list = core_list + all_selected
        if full_list:
            symptom_display = pd.DataFrame({
                'Symptom': full_list,
                'Status': ['Present'] * len(full_list)
            })
            st.dataframe(
                symptom_display,
                use_container_width=True,
                hide_index=True
            )

        # ── Disclaimer ──
        st.divider()
        st.caption(
            "⚠️ This prediction is generated by a machine learning model trained on "
            "synthetic healthcare data. It is intended for academic and decision-support "
            "purposes only and does not constitute medical advice. Always consult a "
            "qualified healthcare professional for diagnosis and treatment."
        )


        # ── Model Performance Info Panel ──
with st.expander("ℹ️ About this model — accuracy and performance", expanded=False):

    st.markdown("#### How accurate is this triage system?")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Accuracy", "87.77%", help="Correct predictions on test data")
    with col2:
        st.metric("Macro F1 Score", "0.8798", help="Balanced metric across all 3 classes")
    with col3:
        st.metric("CV F1 Mean", "0.8414", help="5-fold cross-validation result")
    with col4:
        st.metric("CV Stability", "±0.0073", help="Low variance = consistent results")

    st.divider()

    st.markdown("#### How it was built")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        **Model selected**
        Random Forest Classifier trained on 8,410 patient records
        across two merged symptom datasets.
        """)

    with col_b:
        st.markdown("""
        **Three models compared**
        Logistic Regression (62%), XGBoost (85.29%),
        Random Forest (87.77%). Random Forest won on all metrics.
        """)

    with col_c:
        st.markdown("""
        **Class balance**
        SMOTE was applied to balance Emergency, GP Appointment,
        and Self-Care classes to exactly 33.3% each.
        """)

    st.divider()

    st.markdown("#### Per-class recall")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric(
            "Emergency recall", "87.1%",
            delta="0 missed as Self-Care",
            delta_color="normal"
        )
    with r2:
        st.metric(
            "GP Appointment recall", "93.7%",
            delta="Strongest class",
            delta_color="normal"
        )
    with r3:
        st.metric(
            "Self-Care recall", "82.5%",
            delta="10 sent to GP (safe)",
            delta_color="normal"
        )

    st.divider()
    st.caption(
        "This model was developed for COM 763 Advanced Machine Learning at Wrexham University. "
        "It is a research prototype and should not be used as a substitute for clinical judgement."
    )

# ── Footer ──
st.divider()
st.markdown("""
    <div style='text-align:center; color:gray; font-size:0.8rem; padding-bottom:1rem;'>
        Model: Random Forest | Accuracy: 87.77% | Macro F1: 0.8798
    </div>
""", unsafe_allow_html=True)