# heart_disease_prediction_system
Clinical Decision Support Tool for Heart Disease Prediction using Random Forest, and Streamlit.

❤️ Heart Disease Prediction System
Clinical Decision Support Tool Powered by Random Forest

Author: Najari Umar Jibril
Role: Machine Learning Engineer

📌 Project Overview

This project is a Heart Disease Prediction System built using a Random Forest classifier.
The application allows healthcare professionals to input patient clinical data and receive:
Risk prediction (High Risk / Low Risk)
Probability score
Model performance metrics
Feature importance visualization
Confusion matrix
5-fold cross-validation results
CSV logging of predictions

This tool is designed for demonstration and educational purposes only.

🏥 Problem Statement
Heart disease remains one of the leading causes of death worldwide.
Early prediction using machine learning can assist clinicians in identifying high-risk patients and making informed decisions.
This project demonstrates how machine learning can support clinical decision-making.

🧠 Machine Learning Approach
Model Used
Random Forest Classifier

Why Random Forest?
Handles nonlinear relationships
Reduces overfitting compared to single decision trees
Provides feature importance
Robust and interpretable

📊 Dataset
The dataset contains clinical attributes such as:
Age
Sex (1 = Male, 0 = Female)
Chest Pain Type
Resting Blood Pressure
Serum Cholesterol
Fasting Blood Sugar
Resting ECG
Maximum Heart Rate Achieved
Exercise Induced Angina
ST Depression
Slope of ST Segment
Number of Major Vessels
Thalassemia

Target Variable:
1 = Presence of heart disease
0 = No heart disease

🚀 Features of the Application
1️⃣ Patient Prediction
Accepts full clinical feature input
Displays probability-based risk level

2️⃣ Model Evaluation
Accuracy
Precision
Recall
F1 Score
ROC-AUC Score
Confusion Matrix

3️⃣ Cross Validation
5-Fold Stratified Cross Validation

4️⃣ Feature Importance
Displays impact of each feature on prediction

5️⃣ Prediction Logging
Automatically saves predictions to CSV file

6️⃣ Professional Hospital-Themed UI
Clean medical dashboard interface
Sidebar patient input
Structured evaluation panels

📈 Model Performance (Example)
Metric	Score
Accuracy	~97%
ROC-AUC	~0.99
Precision	High
Recall	High
F1 Score	High
(Note: Actual performance depends on dataset version.)

🛠️ Technologies Used
Python
Streamlit
Scikit-learn
Pandas
NumPy
Matplotlib
Joblib

💻 How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/Najjar07/heart-disease-prediction.git
cd heart-disease-prediction

2️⃣ Install dependencies
pip install -r requirements.txt

Or manually install:
pip install streamlit scikit-learn pandas numpy matplotlib joblib

3️⃣ Run the app
streamlit run app.py
📂 Project Structure
├── Heart_Disease_mrrf_app.py
├── rf_heart_disease_Model.pkl
├── heart_disease.csv
├── prediction_history.csv
├── README.md

⚠️ Disclaimer
This system is intended for:
Educational purposes
Demonstration of machine learning concepts
Portfolio presentation
It is NOT intended for real-world medical diagnosis.
Always consult licensed medical professionals for clinical decisions.

🌍 Future Improvements
Add SHAP Explainability
Add ROC Curve visualization
Add database storage (SQLite/PostgreSQL)
Hyperparameter tuning
Model comparison (Logistic Regression, XGBoost)
API deployment using FastAPI

👨‍💻 Author
Najari Umar Jibril
Machine Learning Engineer
Specializing in:
Predictive Modeling
Healthcare Data Analysis
AI Deployment
