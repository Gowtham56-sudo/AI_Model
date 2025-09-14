# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier  # CHANGED: Imported XGBoost
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="MSME Early Warning Dashboard",
    page_icon="⚠️",
    layout="wide"
)

# --- AI Model Function ---
def run_model_analysis(df):
    # Feature Engineering
    df['Consumption_Drop_Percentage'] = ((df['Mar_Consumption'] - df['Apr_Consumption']) / (df['Mar_Consumption'] + 1)) * 100
    
    # Labeling
    df['Is_Sick'] = df['Consumption_Drop_Percentage'].apply(lambda x: 1 if x > 30 else 0)
    
    # Prepare Data for Model
    features = ['Age', 'Employees', 'Consumption_Drop_Percentage']
    target = 'Is_Sick'
    X = df[features]
    y = df[target]
    
    # Split, Train, and Predict
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # CHANGED: Initialized XGBClassifier instead of RandomForest
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on the entire dataset to display results
    df['Prediction'] = model.predict(X)
    
    # Calculate metrics
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    
    return df, accuracy, precision, recall

# --- Dashboard UI ---
st.title("🚀 AI Early Warning System with XGBoost")
st.markdown("This dashboard uses the powerful **XGBoost** model to predict potentially sick MSMEs.")

# Load data
try:
    data = pd.read_csv('sample_msme_data_50.csv', on_bad_lines='skip')
except FileNotFoundError:
    st.error("Error: 'sample_msme_data_50.csv' not found. Please make sure it's in the same folder.")
    st.stop()

# Sidebar
st.sidebar.header("Dashboard Controls")
if st.sidebar.button("Run XGBoost Analysis"):
    with st.spinner('Running XGBoost model... Please wait.'):
        time.sleep(2) # Just for dramatic effect
        # Call the function to run the analysis
        results_df, accuracy, precision, recall = run_model_analysis(data)
        st.sidebar.success("Analysis Complete!")

        st.subheader("📊 Model Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy*100:.2f}%")
        col2.metric("Precision", f"{precision*100:.2f}%")
        col3.metric("Recall", f"{recall*100:.2f}%")
        
        st.subheader("🚩 Flagged MSMEs (Potentially Sick)")
        flagged_msmes = results_df[results_df['Prediction'] == 1]
        st.dataframe(flagged_msmes[['MSME_ID', 'BusinessType', 'Employees', 'Mar_Consumption', 'Apr_Consumption', 'Consumption_Drop_Percentage']])

        st.subheader("✅ Healthy MSMEs")
        healthy_msmes = results_df[results_df['Prediction'] == 0]
        st.dataframe(healthy_msmes[['MSME_ID', 'BusinessType', 'Employees', 'Mar_Consumption', 'Apr_Consumption']])

else:
    st.info("Click the 'Run XGBoost Analysis' button on the sidebar to start.")

# Option to show raw data
if st.checkbox("Show Raw Sample Data"):
    st.subheader("Raw Data")
    st.dataframe(data)    git config --global user.name "Gowtham T"
    git config --global user.email gowtham.t20062@gmail.com