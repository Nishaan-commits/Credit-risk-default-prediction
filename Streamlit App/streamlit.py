import streamlit as st
import pandas as pd
from feature_engineering import FeatureEngineeringTransformer
import joblib # type: ignore

@st.cache_resource
def load_pipeline():
    return joblib.load("credit_full_pipeline.pkl")

pipeline = load_pipeline()
st.set_page_config(layout="wide")

st.title('German Credit Risk Prediction')
st.image("""https://cdn.educba.com/academy/wp-content/uploads/2021/02/Credit-Risk.jpg""")
st.header('Enter Customer Details:')

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
        .block-container {
        max-width: 900px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# We expose semantic financial ranges to users and internally map them to categorical codes expected by the model.
personal_status_map = {
    "Male: Divorced/Seperated":"A91",
    "Female: Divorced/Seperated/Married":"A92",
    "Male: Single":"A93",
    "Male: Married/Widowed":"A94",
    "Female: Single":"A95",
}
foreign_worker_map = {
    "Yes":"A201",
    "No":"A202",
}
employment_map = {
    "Unemployed":"A71",
    "Employment since < 1 Year":"A72",
    "1 Year <= Employment since < 4 Years":"A73",
    "4 Years <= Employment since < 7 Years":"A74",
    "Employment since >= 7 Years":"A75",
}
Job_map = {
    "Unemployed/ Unskilled - Non resident": "A171",
    "Unskilled - Resident":"A172",
    "Skilled Employee/ Official":"A173",
    "Management/ self-employed/ highly qualified employee/ officer":"A174",
}
checking_status_map = {
    "Balance < 0 DM": "A11",
    "0 ≤ Balance < 200 DM": "A12",
    "Balance ≥ 200 DM / salary ≥ 1 year": "A13",
    "No checking account": "A14",
}
savings_map = {
    "Savings < 100 DM": "A61",
    "100 ≤ Savings < 500 DM": "A62",
    "500 ≤ Savings < 1000 DM": "A63",
    "Savings ≥ 1000 DM": "A64",
    "Unknown / no savings account": "A65",
}
property_map = {
    "Real Estate":"A121",
    "Building society savings agreement/ life insurance":"A122",
    "Car or Other(Not selected in Purpose)":"A123",
    "Unknown or No Property":"A124",
}
telephone_map = {
    "None":"A191",
    "Yes, registered under Customer's Name":"A192",
}
credit_history_map = {
    "No credits taken / all credits paid back duly": "A30",
    "All credits at this bank paid back duly": "A31",
    "Existing credits paid back duly till now": "A32",
    "Delay in paying off in the past": "A33",
    "Critical account / other credits existing": "A34",
}
debtors_map = {
    "None":"A101",
    "Co-applicant":"A102",
    "Guarantor":"A193",
}
other_installment_plans_map = {
    "Bank":"A141",
    "Stores":"A142",
    "None":"A143",
}
housing_map = {
    "Rent":"A151",
    "Own":"A152",
    "For Free":"A153",
}
purpose_map = {
    "Car (new)": "A40",
    "Car (used)": "A41",
    "Furniture / equipment": "A42",
    "Radio / television": "A43",
    "Domestic appliances": "A44",
    "Repairs": "A45",
    "Education": "A46",
    "Vacation": "A47",
    "Retraining": "A48",
    "Business": "A49",
    "Others": "A410",
}





with st.form("Credit Form"):
    with st.expander("Applicant Profile", expanded= True):
        st.caption("Who is the applicant?")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=80, value=30)
        personal_status_label = st.selectbox(
            "Personal Status and Sex",
            list(personal_status_map.keys())
        )
        with col2:
            foreign_worker_label = st.selectbox(
                "Foreign Worker",
                list(foreign_worker_map.keys())
            )
        people_liable = st.number_input("Number of people being liable to provide maintenance for", min_value=0, max_value=2)

    with st.expander("Employment and Stability", expanded = True):
        st.caption("How stable is the applicant’s income?")
  
        employment_label = st.selectbox(
            "Present Employment Since",
            list(employment_map.keys())
        )
        Job_label = st.selectbox(
            "Job",
            list(Job_map.keys())
        )

    with st.expander("Financial Position", expanded=True):
        st.caption("Current financial strength & buffers")
        col4, col5 = st.columns(2)
        with col4:
            checking_label = st.selectbox(
                "Status of existing checking account",
                list(checking_status_map.keys())
            )
            savings_label = st.selectbox(
                "Savings account / bonds",
                list(savings_map.keys())
            )
        with col5:
            property_label = st.selectbox(
                "Property",
                list(property_map.keys())
            )
            telephone_label = st.selectbox(
                "Telephone",
                list(telephone_map.keys())
            )
    with st.expander("Credit History and Obligations"):
        st.caption("Past behavior and existing liabilities")
        col6, col7 = st.columns(2)
        credit_history_label = st.selectbox(
            "Credit History",
            list(credit_history_map.keys())
        )
        number_credits = st.number_input("Number of existing credits at this bank", min_value=1, max_value=4)
        with col6:
            debtors_label = st.selectbox(
                "Other Debtors or Guarantors",
                list(debtors_map.keys())
            )
        with col7:
            other_installment_plans_label = st.selectbox(
                "Other Installment Plans",
                list(other_installment_plans_map.keys())
            )
    with st.expander("Living Situation", expanded=True):
        st.caption("Housing Stability")
        col1, col2 = st.columns(2)
        with col1:
            housing_label = st.selectbox(
                "Housing",
                list(housing_map.keys())
            )
        with col2:
            residence_since = st.number_input("Present residence since", min_value=1, max_value=4)
    with st.expander("Loan / Credit Request Details", expanded=True):
        st.caption("Details of the Requested Loan")
        col9, col10 = st.columns(2)
        with col9:
            amount = st.number_input("Loan Amount", min_value=250, max_value = 20000)
            purpose_label = st.selectbox(
                "Purpose of the loan",
                list(purpose_map.keys())
            )
        with col10:
            duration = st.number_input("Duration (months)", min_value=4, max_value=72)
            installment_rate = st.number_input("Installment Rate (Category)", min_value=1, max_value=4)
    

    submitted = st.form_submit_button("🔍 Predict Credit Risk")
                    
personal_status = personal_status_map[personal_status_label]
status = checking_status_map[checking_label]
purpose = purpose_map[purpose_label]
credit_history = credit_history_map[credit_history_label]
savings = savings_map[savings_label]
employment = employment_map[employment_label]
debtors = debtors_map[debtors_label]
property = property_map[property_label]
other_installment_plans = other_installment_plans_map[other_installment_plans_label]
housing = housing_map[housing_label]
job = Job_map[Job_label]
telephone = telephone_map[telephone_label]
foreign_worker = foreign_worker_map[foreign_worker_label]

input_df = pd.DataFrame([{
    "status": status,
    "duration": duration,
    "credit_history": credit_history,
    "purpose": purpose,
    "amount": amount,
    "savings": savings,
    "employment": employment,
    "installment_rate":installment_rate,
    "personal_status":personal_status,
    "debtors":debtors,
    "residence_since":residence_since,
    "property":property,
    "age":age,
    "other_installment_plans":other_installment_plans,
    "housing":housing,
    "number_credits":number_credits,
    "job":job,
    "people_liable":people_liable,
    "telephone":telephone,
    "foreign_worker":foreign_worker,
}])



# We decouple probability estimation from decision threshold, allowing business-driven risk calibration
THRESHOLD = 0.55

if submitted:
    with st.expander("Model Input (for transparency)"):
        st.dataframe(input_df)

    proba = pipeline.predict_proba(input_df)[0][1]
    pred = int(proba >= THRESHOLD)

    st.divider()

    if pred == 1:
        st.error(
            f"🚨 High Credit Risk\n\n"
            f"Probability: **{proba:.2f}**\n"
            f"Decision Threshold: **{THRESHOLD}**"
        )
    else:
        st.success(
            f"✅ Low Credit Risk\n\n"
            f"Probability: **{proba:.2f}**\n"
            f"Decision Threshold: **{THRESHOLD}**"
        )

st.markdown("---")
st.caption("Built with Streamlit, scikit-learn & XGBoost")
st.caption("Credit Risk Modeling")
