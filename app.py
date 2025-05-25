from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# Set Streamlit layout to wide
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# Load the trained model
try:
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Could not load model: {e}")


# Load the MinMaxScaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the input features for the model
feature_names = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "EstimatedSalary", "Geography_France", "Geography_Germany", "Geography_Spain",
    "Gender_Female", "Gender_Male", "HasCrCard_0", "HasCrCard_1",
    "IsActiveMember_0", "IsActiveMember_1"
]

# Columns requiring scaling
scale_vars = ["CreditScore", "EstimatedSalary", "Tenure", "Balance", "Age", "NumOfProducts"]

# Updated default values
default_values = [
    600, 30, 2, 8000, 2, 60000,
    True, False, False, True, False, False, True, False, True
]

# Sidebar setup
st.sidebar.image("Pic 1.PNG", use_container_width=True)  # Display Pic 1

st.sidebar.header("User Inputs")

# Collect user inputs
user_inputs = {}
for i, feature in enumerate(feature_names):
    if feature in scale_vars:
        user_inputs[feature] = st.sidebar.number_input(
            feature, value=default_values[i], step=1 if isinstance(default_values[i], int) else 0.01
        )
    elif isinstance(default_values[i], bool):
        user_inputs[feature] = st.sidebar.checkbox(feature, value=default_values[i])
    else:
        user_inputs[feature] = st.sidebar.number_input(
            feature, value=default_values[i], step=1
        )

# Convert inputs to a DataFrame
input_data = pd.DataFrame([user_inputs])

# Apply MinMaxScaler to the required columns
input_data_scaled = input_data.copy()
input_data_scaled[scale_vars] = scaler.transform(input_data[scale_vars])

# Submit button
submitted = st.sidebar.button("🔍 Predict")

# App Header
st.title("Customer Churn Prediction")

# Page Layout
left_col, right_col = st.columns(2)

# Left Page: Feature Importance
if submitted:
    with left_col:
        # st.header("Feature Importance")
        st.header("Prediction")
        input_data = pd.DataFrame([user_inputs])
        input_data_scaled = input_data.copy()
        input_data_scaled[scale_vars] = scaler.transform(input_data[scale_vars])

        probabilities = model.predict_proba(input_data_scaled)[0]
        prediction = model.predict(input_data_scaled)[0]
        prediction_label = "Churned" if prediction == 1 else "Retain"
        # Add simple visual feedback
        st.progress(int(probabilities[1] * 100))
        st.metric(label="Churn Probability", value=f"{probabilities[1]:.2%}")
        st.metric(label="Retention Probability", value=f"{probabilities[0]:.2%}")

    # Right Page: Prediction
    with right_col:
        # st.header("Prediction")
        # Get the predicted probabilities and label
        probabilities = model.predict_proba(input_data_scaled)[0]
        prediction = model.predict(input_data_scaled)[0]
        # Map prediction to label
        prediction_label = "Churned" if prediction == 1 else "Retain"
        
        # Optional pie chart
        fig = px.pie(
            names=["Retain", "Churn"],
            values=[probabilities[0], probabilities[1]],
            color_discrete_sequence=["green", "red"],
            title="Prediction Breakdown"
        )
        st.plotly_chart(fig)
        st.markdown(f" Output: Based on given values the model predict the customer can be **{prediction_label}** ")
