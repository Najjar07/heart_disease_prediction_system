import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score
import os

# --- Load the trained model ---
model_path = "rf_heart_disease_Model.pkl"
rf_model = joblib.load(model_path)

st.markdown("<h1>❤️ Heart Disease Prediction System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#4a4a4a;'>Clinical Decision Support Tool powered by Random Forest</p>",
    unsafe_allow_html=True
)

#st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
#st.title("❤️ Heart Disease Prediction App")
#st.write("Predict whether a patient is at risk of heart disease using a Random Forest model.")

# ==============================
# Hospital Theme Styling
# ==============================
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #f4f8fb;
}

/* Title Styling */
h1 {
    color: #0b3c5d;
    text-align: center;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #e8f1f8;
    padding: 20px;
}

/* Buttons */
.stButton>button {
    background-color: #0b3c5d;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #145a86;
    color: white;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background-color: white;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
}

/* Footer */
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Sidebar: Patient Input
# ==============================
st.sidebar.header("Patient Data Input")

# Patient Name
patient_name = st.sidebar.text_input("Patient Name")

# Manual full feature input
age = st.sidebar.number_input("Age", value=30)
sex = st.sidebar.selectbox(
    "Sex (1 = Male, 0 = Female)",
    [1, 0],
    format_func=lambda x: "Male (1)" if x == 1 else "Female (0)"
)

chest_pain = st.sidebar.selectbox(
    "Chest Pain Type",
    [0,1,2,3],
    help="""
    0 = Typical Angina
    1 = Atypical Angina
    2 = Non-anginal Pain
    3 = Asymptomatic
    """
)

resting_bp = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", value=120)
cholesterol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", value=200)

fasting_bs = st.sidebar.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [1,0],
    format_func=lambda x: "Yes (1)" if x==1 else "No (0)"
)

rest_ecg = st.sidebar.selectbox(
    "Resting Electrocardiographic Results",
    [0,1,2],
    help="""
    0 = Normal
    1 = ST-T abnormality
    2 = Left ventricular hypertrophy
    """
)

max_hr = st.sidebar.number_input("Maximum Heart Rate Achieved", value=150)

exercise_angina = st.sidebar.selectbox(
    "Exercise Induced Angina",
    [1,0],
    format_func=lambda x: "Yes (1)" if x==1 else "No (0)"
)

st_depression = st.sidebar.number_input("ST Depression (Oldpeak)", value=1.0)

slope = st.sidebar.selectbox(
    "Slope of Peak Exercise ST Segment",
    [0,1,2],
    help="""
    0 = Upsloping
    1 = Flat
    2 = Downsloping
    """
)

major_vessels = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0,1,2,3])
thalassemia = st.sidebar.selectbox(
    "Thalassemia",
    [0,1,2,3],
    help="""
    0 = Normal
    1 = Fixed Defect
    2 = Reversible Defect
    3 = Unknown
    """
)

# --- Predict Button at Bottom ---
predict_button = st.sidebar.button("Predict")

# Create dataframe
input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [chest_pain],
    "trestbps": [resting_bp],
    "chol": [cholesterol],
    "fbs": [fasting_bs],
    "restecg": [rest_ecg],
    "thalach": [max_hr],
    "exang": [exercise_angina],
    "oldpeak": [st_depression],
    "slope": [slope],
    "ca": [major_vessels],
    "thal": [thalassemia]
})

# ==============================
# Prediction Section
# ==============================
if predict_button:

    st.subheader("Input Patient Data")
    display_df = input_df.copy()
    display_df.insert(0, "Patient Name", patient_name)
    st.dataframe(display_df)

    prediction = rf_model.predict(input_df)[0]
    prediction_proba = rf_model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning(f"⚠️ High Risk of Heart Disease ({prediction_proba*100:.2f}%)")
        result_label = "High Risk"
    else:
        st.success(f"✅ Low Risk of Heart Disease ({(1-prediction_proba)*100:.2f}%)")
        result_label = "Low Risk"

    # ==============================
    # Save to CSV
    # ==============================
    csv_file = "prediction_history.csv"

    save_df = display_df.copy()
    save_df["Prediction"] = result_label
    save_df["Probability"] = prediction_proba

    if os.path.exists(csv_file):
        save_df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        save_df.to_csv(csv_file, index=False)

    st.info("Prediction saved to prediction_history.csv")

    # ==============================
    # Feature Importance
    # ==============================
    st.subheader("Feature Importance")

    importances = rf_model.feature_importances_
    features = rf_model.feature_names_in_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([features[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# ==============================
# Model Evaluation
# ==============================
st.subheader("Model Evaluation (Test Set)")

file_path = r"C:\Users\LENOVO\Desktop\sk\heart_disease.csv"
df = pd.read_csv(file_path)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred = rf_model.predict(X_test)
y_pred_prob = rf_model.predict_proba(X_test)[:,1]

st.write("**Accuracy:**", f"{accuracy_score(y_test, y_pred):.4f}")
st.write("**ROC-AUC Score:**", f"{roc_auc_score(y_test, y_pred_prob):.4f}")
st.write("**Classification Report:**")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5,4))
ax.imshow(cm, cmap='Blues')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha='center', va='center')

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# Cross-validation
st.subheader("5-Fold Cross-Validation Accuracy")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='accuracy')
st.write("CV Accuracy per fold:", np.round(cv_scores,4))
st.write(f"Mean CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ==============================
# Footer
# ==============================
#st.write("---")
#st.write("Model powered by Random Forest.")
#st.write("For demonstration and educational purposes only.")
#st.write("**Najari Umar Jibril – Machine Learning Engineer**")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#6c757d;'>"
    "Model powered by Random Forest | For demonstration and educational purposes only."
    "<br><strong>Najari Umar Jibril – Machine Learning Engineer</strong>"
    "</div>",
    unsafe_allow_html=True

)
