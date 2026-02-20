import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import base64

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

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])

# --- TAB 1: PREDICT ---
with tab1:
    st.header("Patient Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=150, value=50)
        sex = st.selectbox("Gender", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
        restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        
    with col3:
        slope = st.selectbox("Slope of the Peak ST Segment", ["Upsloping", "Flat", "Downsloping"])
        ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=4, value=0)
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"])

    if st.button("Submit"):
        sex_val = 1 if sex == "Male" else 0
        cp_dict = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
        fbs_val = 1 if fbs == "True" else 0
        restecg_dict = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
        exang_val = 1 if exang == "Yes" else 0
        slope_dict = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
        thal_dict = {"Normal": 1, "Fixed Defect": 2, "Reversable Defect": 3}
        
        input_data = np.array([[
            age, sex_val, cp_dict[cp], trestbps, chol, fbs_val, 
            restecg_dict[restecg], thalach, exang_val, oldpeak, 
            slope_dict[slope], ca, thal_dict[thal]
        ]])
        
        # NOTE: Removed the scaler transformation step
        
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
            
            # Get prediction probabilities for "Model Thinking"
            try:
                probabilities = model.predict_proba(input_data)[0]
                prob_no_disease = probabilities[0]
                prob_disease = probabilities[1]
            except AttributeError:
                # Fallback if model doesn't support predict_proba
                prob_disease = 1.0 if pred == 1 else 0.0
                prob_no_disease = 1.0 - prob_disease

            result_text = "Heart Disease Detected" if pred == 1 else "No Heart Disease"
            color = "red" if pred == 1 else "green"
            confidence = prob_disease if pred == 1 else prob_no_disease
            
            with cols[i]:
                st.markdown(f"**{name}**")
                st.markdown(f"<h5 style='color: {color};'>{result_text}</h5>", unsafe_allow_html=True)
                st.write(f"**Model Confidence:** {confidence * 100:.1f}%")
                
                # Show probability distribution
                st.caption("Model Probabilities:")
                st.progress(float(prob_disease), text=f"Disease: {prob_disease*100:.1f}%")
                st.progress(float(prob_no_disease), text=f"No Disease: {prob_no_disease*100:.1f}%")

# --- TAB 2: BULK PREDICT ---
with tab2:
    st.header("Bulk Predict")
    st.markdown("""
    **Instructions:**
    1. No NaN values are allowed.
    2. Total 13 features must be present in exactly this order: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`.
    3. Categorical variables must be correctly encoded numerically.
    """)
    
    uploaded_file = st.file_uploader("Upload CSV File (Max 200MB)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(input_df.head())
            
            # NOTE: Removed the scaler transformation step
            predictions = xgb_model.predict(input_df) 
            input_df['Prediction_XGB'] = predictions
            
            st.write("### Prediction Results")
            st.dataframe(input_df)
            
            csv = input_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Prediction CSV",
                data=csv,
                file_name='bulk_predictions_results.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.warning(f"Please make sure the uploaded CSV file has correct columns and data format. Error: {e}")

# --- TAB 3: MODEL INFORMATION ---
with tab3:
    st.header("Model Performance Information")
    
    model_data = {
        'Decision Tree': 80.9,
        'Logistic Regression': 85.8,
        'Random Forest': 85.9,
        'XGBoost': 86.4
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
    
    st.plotly_chart(fig, use_container_width=True)