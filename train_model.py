# Import necessary libraries
import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import numpy as np
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="MSME Real-time Health Dashboard",
    page_icon="🩺",
    layout="wide"
)

# --- AI Model & Data Processing Function ---
@st.cache_data # Cache the data and model to run faster
def run_ai_analysis(df):
    # 1. Feature Engineering
    df['Consumption_Drop_Percentage'] = ((df['Mar_Consumption'] - df['Apr_Consumption']) / (df['Mar_Consumption'] + 1)) * 100
    df['Is_Sick_Label'] = df['Consumption_Drop_Percentage'].apply(lambda x: 1 if x > 30 else 0)
    
    # 2. Prepare Data for Model
    features = ['Age', 'Employees', 'Consumption_Drop_Percentage']
    target = 'Is_Sick_Label'
    X = df[features]
    y = df[target]
    
    # 3. Train the XGBoost Model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X, y) # Train on the full dataset for this prototype
    
    # 4. Generate Risk Score (from 0 to 100)
    # predict_proba gives probability for class 0 and 1. We take probability for class 1 (Sick).
    probabilities = model.predict_proba(X)[:, 1]
    df['Risk_Score'] = np.round(probabilities * 100, 2)
    
    # 5. Assign Health Status based on Risk Score
    def get_health_status(score):
        if score > 70:
            return "🔴 High Risk"
        elif 40 <= score <= 70:
            return "🟡 Medium Risk"
        else:
            return "🟢 Low Risk"
            
    df['Health_Status'] = df['Risk_Score'].apply(get_health_status)
    
    return df.sort_values(by='Risk_Score', ascending=False)

# --- Dashboard UI ---
st.title("🩺 MSME Real-time Health Dashboard")
st.markdown("An advanced AI dashboard to monitor MSME health, generate reports, and provide real-time alerts.")

# --- Load Data ---
try:
    data = pd.read_csv('sample_msme_data_50.csv', on_bad_lines='skip')
except FileNotFoundError:
    st.error("Error: 'sample_msme_data_50.csv' not found. Please make sure it's in the same folder.")
    st.stop()

# --- Main Dashboard ---
st.subheader("MSME Health Overview")

# Run the AI analysis
results_df = run_ai_analysis(data)

# Display the main table with color-coded indicators
st.dataframe(results_df[['MSME_ID', 'Health_Status', 'Risk_Score', 'BusinessType', 'Employees', 'Consumption_Drop_Percentage']])

# --- 2. Detailed Reporting System ---
st.sidebar.title("📄 Detailed Reporting")
selected_msme = st.sidebar.selectbox("Select an MSME for a detailed report:", results_df['MSME_ID'])

if selected_msme:
    st.subheader(f"Detailed Report for: {selected_msme}")
    
    msme_data = results_df[results_df['MSME_ID'] == selected_msme].iloc[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Health Status", msme_data['Health_Status'])
        st.metric("Risk Score", f"{msme_data['Risk_Score']}%")
        
        # Anomaly and Risk Factor Analysis
        st.write("**Risk Factor Analysis:**")
        if msme_data['Health_Status'] == "🔴 High Risk":
            st.error(f"- Critical drop in electricity consumption of {msme_data['Consumption_Drop_Percentage']:.2f}%.")
        elif msme_data['Health_Status'] == "🟡 Medium Risk":
            st.warning(f"- Notable drop in electricity consumption of {msme_data['Consumption_Drop_Percentage']:.2f}%. Monitoring suggested.")
        else:
            st.success("- No significant risk factors detected. Operations appear stable.")

    with col2:
        # Historical Performance Chart
        st.write("**Historical Electricity Consumption:**")
        consumption_data = msme_data[['Jan_Consumption', 'Feb_Consumption', 'Mar_Consumption', 'Apr_Consumption']]
        consumption_df = pd.DataFrame({
            'Month': ['Jan_Consumption', 'Feb_Consumption', 'Mar_Consumption', 'Apr_Consumption'],
            'Consumption (Units)': consumption_data.values
        })
        
        fig = px.line(consumption_df, x='Month', y='Consumption (Units)', title=f"Consumption Trend for {selected_msme}", markers=True)
        st.plotly_chart(fig, use_container_width=True)

# --- 3. Real-time Alert System (Simulation) ---
st.sidebar.title("🚨 Real-time Alerts")
high_risk_alerts = results_df[results_df['Health_Status'] == "🔴 High Risk"]

if not high_risk_alerts.empty:
    for index, row in high_risk_alerts.iterrows():
        st.sidebar.error(
            f"**ALERT:** {row['MSME_ID']} is at HIGH RISK with a score of {row['Risk_Score']}%. "
            f"Immediate attention required due to a {row['Consumption_Drop_Percentage']:.2f}% consumption drop."
        )
else:
    st.sidebar.success("No high-risk alerts at the moment. All MSMEs are stable.")