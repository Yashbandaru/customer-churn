import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load trained model
with open("churn_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to leave the service.")

st.divider()
st.subheader("Enter Customer Details")

with st.form("churn_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])

        senior = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (months)",
            0,
            72,
            12
        )

        phone = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No phone service", "No", "Yes"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

        backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

        device = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )

    with col2:
        support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

        tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

        movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        use_container_width=True
    )

if submitted:

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")

    st.metric(
        "Estimated Churn Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    st.caption(
        "Predictions are estimates and should be interpreted carefully."
    )