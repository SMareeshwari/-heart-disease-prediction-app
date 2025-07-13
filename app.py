import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# Load trained model
model = pickle.load(open('heart_model.pkl', 'rb'))

# Load test data for evaluation
test_data = pd.read_csv("heart_test_data.csv")
X_test = test_data.drop("target", axis=1)
y_test = test_data["target"]

# Sidebar info
st.sidebar.title("About This App")
st.sidebar.markdown("""
This app predicts the likelihood of heart disease using a machine learning model trained on clinical data.
- Built with 🐍 Python, 🧠 Scikit-learn, and 🎨 Streamlit.
- Created for educational/demo use only.
""")
st.sidebar.markdown("👨‍💻 Developed by: **Mareeshwari**")

# Main layout
st.title("❤️ Heart Disease Risk Prediction")
st.markdown("This tool helps estimate heart disease risk based on clinical attributes. Please enter the details below.")

st.divider()
st.subheader("🔍 Patient Information")

# Group inputs into columns for cleaner UI
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, value=120)
    chol = st.number_input("Cholesterol Level", 100, 600, value=200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])

with col2:
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", 50, 250, value=150)
    exang = st.radio("Exercise Induced Angina", [0, 1])
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (Thal)", [0, 1, 2, 3])

# Tooltip-style info
st.info("ℹ️ *Oldpeak* is the ST depression induced by exercise. \n\n*Thal* refers to a blood disorder called thalassemia.")

# Format input for prediction
input_data = np.array([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)

    st.subheader("🩺 Prediction Result")
    if prediction[0] == 1:
        st.error("⚠️ High risk of Heart Disease. Please consult a medical professional.")

        # Doctor consultation link (Practo)
        st.markdown("""
        <a href="https://www.practo.com/consult" target="_blank">
            <button style='background-color: #ff4b4b; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;'>
                👨‍⚕️ Book Consultation with a Doctor
            </button>
        </a>
        """, unsafe_allow_html=True)

    else:
        st.success("✅ Low risk of Heart Disease. Keep maintaining a healthy lifestyle!")

    # Optional Evaluation
    if st.checkbox("Show model evaluation metrics"):
        st.divider()
        st.subheader("📊 Model Evaluation")

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Confusion Matrix
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        st.pyplot(fig_cm)

        # ROC Curve
        st.markdown("### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Receiver Operating Characteristic')
        ax_roc.legend(loc='lower right')
        st.pyplot(fig_roc)

        # Classification Report
        st.markdown("### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in report.items():
            if label in ['0', '1']:
                st.write(f"**Class {label}**")
                st.write(f"- Precision: {metrics['precision']:.2f}")
                st.write(f"- Recall: {metrics['recall']:.2f}")
                st.write(f"- F1-Score: {metrics['f1-score']:.2f}")

# Disclaimer
st.divider()
st.warning("⚠️ This tool is for demonstration purposes only. It is not a substitute for professional medical advice.")
