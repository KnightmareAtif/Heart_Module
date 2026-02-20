import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("Heart Disease Predictor")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    with open('xgb_model.pkl', 'rb') as f:
        xgb = pickle.load(f)
    with open('rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('dt_model.pkl', 'rb') as f:
        dt = pickle.load(f)
    with open('log_model.pkl', 'rb') as f:
        log = pickle.load(f)
    return xgb, rf, dt, log

try:
    xgb_model, rf_model, dt_model, log_model = load_models()
except FileNotFoundError:
    st.error("Model files not found. Please upload them to the directory.")
    st.stop()

# Define the EXACT 11 features the model was trained on
feature_names = [
    'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
    'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
]

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])

# --- TAB 1: PREDICT ---
with tab1:
    st.header("Patient Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=150, value=50)
        sex = st.selectbox("Gender", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", ["Atypical Angina (ATA)", "Non-Anginal Pain (NAP)", "Asymptomatic (ASY)", "Typical Angina (TA)"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        
    with col2:
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
        restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality (ST)", "Left Ventricular Hypertrophy (LVH)"])
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
        
    with col3:
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=-5.0, max_value=10.0, value=0.0, step=0.1)
        slope = st.selectbox("Slope of the Peak ST Segment", ["Upsloping", "Flat", "Downsloping"])

    if st.button("Submit"):
        # Map values properly (matching notebook encoding)
        sex_val = 0 if sex == "Male" else 1
        cp_dict = {"Atypical Angina (ATA)": 0, "Non-Anginal Pain (NAP)": 1, "Asymptomatic (ASY)": 2, "Typical Angina (TA)": 3}
        fbs_val = 1 if fbs == "True" else 0
        restecg_dict = {"Normal": 0, "ST-T Wave Abnormality (ST)": 1, "Left Ventricular Hypertrophy (LVH)": 2}
        exang_val = 1 if exang == "Yes" else 0
        slope_dict = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
        
        # Build strict DataFrame with only the 11 variables to prevent 13-feature ValueError
        input_data = pd.DataFrame([[
            age, sex_val, cp_dict[cp], trestbps, chol, fbs_val, 
            restecg_dict[restecg], thalach, exang_val, oldpeak, slope_dict[slope]
        ]], columns=feature_names)
        
        st.subheader("Results & Model Thinking")
        models = {
            "Decision Tree Classifier": dt_model,
            "Logistic Regression": log_model,
            "Random Forest": rf_model,
            "XGBoost Classifier": xgb_model
        }
        
        cols = st.columns(len(models))
        for i, (name, model) in enumerate(models.items()):
            pred = model.predict(input_data)[0]
            
            try:
                probabilities = model.predict_proba(input_data)[0]
                prob_no_disease = probabilities[0]
                prob_disease = probabilities[1]
            except AttributeError:
                prob_disease = 1.0 if pred == 1 else 0.0
                prob_no_disease = 1.0 - prob_disease

            result_text = "Heart Disease Detected" if pred == 1 else "No Heart Disease"
            color = "red" if pred == 1 else "green"
            confidence = prob_disease if pred == 1 else prob_no_disease
            
            with cols[i]:
                st.markdown(f"**{name}**")
                st.markdown(f"<h5 style='color: {color};'>{result_text}</h5>", unsafe_allow_html=True)
                st.write(f"**Model Confidence:** {confidence * 100:.1f}%")
                
                st.caption("Model Probabilities:")
                st.progress(float(prob_disease), text=f"Disease: {prob_disease*100:.1f}%")
                st.progress(float(prob_no_disease), text=f"No Disease: {prob_no_disease*100:.1f}%")

# --- TAB 2: BULK PREDICT ---
with tab2:
    st.header("Bulk Predict")
    st.markdown("""
    **Instructions:**
    1. No NaN values are allowed.
    2. Upload a CSV containing at least these 11 features: `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`.
    3. Categorical variables must be numerically encoded identical to the model training.
    """)
    
    uploaded_file = st.file_uploader("Upload CSV File (Max 200MB)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            
            # Select only the 11 required columns (drops extra features if present)
            try:
                features_to_predict = input_df[feature_names]
            except KeyError as e:
                st.error(f"Missing required columns in CSV. Details: {e}")
                st.stop()
            
            st.write("### Data Preview")
            st.dataframe(input_df.head(), width='stretch')
            
            predictions = xgb_model.predict(features_to_predict) 
            input_df['Prediction'] = predictions
            input_df['Result'] = input_df['Prediction'].map({0: 'No Disease', 1: 'Disease Detected'})
            
            st.write("### Prediction Results")
            st.dataframe(input_df, width='stretch')
            
            csv = input_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Prediction CSV",
                data=csv,
                file_name='bulk_predictions_results.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"Error processing the file. Make sure columns match the 11 trained features strictly. Error details: {e}")

# --- TAB 3: MODEL INFORMATION ---
with tab3:
    st.header("Model Performance Information")
    
    model_data = {
        'Decision Tree': 80.97,
        'Logistic Regression': 85.86,
        'Random Forest': 85.86,
        'XGBoost': 86.41
    }
    
    df_acc = pd.DataFrame({
        'Models': list(model_data.keys()),
        'Accuracy': list(model_data.values())
    })
    
    fig = px.bar(
        df_acc, 
        x='Models', 
        y='Accuracy', 
        title='Machine Learning Model Accuracies (%)',
        color='Models',
        text='Accuracy'
    )
    
    fig.update_layout(yaxis_title="Accuracy (%)", xaxis_title="Models")
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    
    st.plotly_chart(fig, width='stretch')