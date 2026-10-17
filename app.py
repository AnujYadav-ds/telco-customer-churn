
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# 2. PROJECT PATHS AND DATA
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "Reports"
DATA_PATH = PROJECT_ROOT / "Data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

df = load_data(str(DATA_PATH))


# Load the saved churn prediction pipeline
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model_v1.pkl"

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.stop()

model = load_model(str(MODEL_PATH))


# --------------------------------------------------
# 3. DARK THEME STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #273449;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 750;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .kpi-card {
        background: linear-gradient(135deg, #172338, #111827);
        border: 1px solid #29364b;
        border-radius: 14px;
        padding: 20px;
        min-height: 125px;
    }

    .kpi-label {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 10px;
    }

    .kpi-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 750;
    }

    .kpi-accent {
        color: #fb7185;
    }

    div[data-testid="stMetric"] {
        background: #172338;
        border: 1px solid #29364b;
        border-radius: 12px;
        padding: 16px;
    }

    hr {
        border-color: #29364b;
    }
    
/* Improve contrast for filter labels and KPI metrics */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"],
[data-testid="stCaptionContainer"] p {
    color: #e2e8f0 !important;
}

[data-testid="stMultiSelect"] label {
    color: #e2e8f0 !important;
}

[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background-color: #172338;
    border-color: #475569;
}

[data-testid="stMultiSelect"] input {
    color: #f8fafc !important;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

[data-testid="stMetricValue"] {
    color: #f8fafc !important;
}

/* Make the CSV download button readable */
.stDownloadButton button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 8px !important;
}

.stDownloadButton button p,
.stDownloadButton button span {
    color: #ffffff !important;
}

.stDownloadButton button:hover {
    background-color: #1d4ed8 !important;
    border-color: #60a5fa !important;
}

/* Consistent, readable button styling */
.stButton > button {
    background-color: #172338 !important;
    color: #f8fafc !important;
    border: 1px solid #475569 !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    transition: background-color 0.2s ease, border-color 0.2s ease;
}

.stButton > button p,
.stButton > button span {
    color: #f8fafc !important;
}

.stButton > button:hover {
    background-color: #2563eb !important;
    border-color: #60a5fa !important;
    color: #ffffff !important;
}

.stButton > button:focus:not(:active) {
    border-color: #60a5fa !important;
    color: #ffffff !important;
}

/* Match Streamlit's header to the dashboard's dark theme */
[data-testid="stHeader"] {
    background: #0b1120 !important;
}

[data-testid="stToolbar"] {
    background: transparent !important;
}

[data-testid="stDecoration"] {
    background: #0b1120 !important;
}

[data-testid="stHeader"] button,
[data-testid="stHeader"] svg {
    color: #e5e7eb !important;
    fill: #e5e7eb;
}

</style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 📡 CHURN INTELLIGENCE")
    st.caption("TELCO CUSTOMER ANALYTICS")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Static EDA Insights",
            "Interactive Analytics",
            "Churn Prediction",
            "Model Insights",
        ],
        index=0,
    )

    st.divider()
    st.caption("Machine Learning Project")
    st.caption("Telco Customer Churn")

# --------------------------------------------------
# 5. CALCULATE OVERVIEW KPIs
# --------------------------------------------------
total_customers = len(df)
churned_customers = int((df["Churn"] == "Yes").sum())
retained_customers = int((df["Churn"] == "No").sum())
churn_rate = (
    churned_customers / total_customers * 100
    if total_customers
    else 0
)

# --------------------------------------------------
# 6. OVERVIEW PAGE
# --------------------------------------------------
if page == "Overview":
    st.markdown(
        '<div class="main-title">Customer Churn Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">'
        'Understand customer behaviour, uncover churn patterns, '
        'and explore machine learning predictions.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 Business Overview")

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        (c1, "TOTAL CUSTOMERS", f"{total_customers:,}", ""),
        (c2, "CHURNED CUSTOMERS", f"{churned_customers:,}", "kpi-accent"),
        (c3, "CHURN RATE", f"{churn_rate:.2f}%", "kpi-accent"),
        (c4, "RETAINED CUSTOMERS", f"{retained_customers:,}", ""),
    ]

    for column, label, value, accent in cards:
        with column:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value {accent}">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown("### 🎯 Project Objective")

    st.markdown(
        """
        This project analyses customer churn in the telecommunications
        industry and develops a machine learning model to identify
        customers who may be at risk of leaving.

        **Explore the dashboard:**
        - **Static EDA Insights:** Review the saved exploratory charts
          and their business findings.
        - **Interactive Analytics:** Filter customer segments and
          explore changing churn patterns.
        - **Churn Prediction:** Test individual customer profiles.
        - **Model Insights:** Review evaluation metrics and SHAP
          explanations.
        """
    )

    st.info(
        "Use the navigation menu in the sidebar to explore each section. "
        "The remaining sections will be implemented next."
    )


elif page == "Static EDA Insights":
    st.title("📈 Exploratory Data Analysis")
    st.markdown(
        "### Understanding the drivers of customer churn"
    )
    st.markdown(
        "These charts summarize findings from the original "
        "Telco Customer Churn dataset."
    )

    st.divider()

    eda_insights = [
        (
            "Contract Type vs Churn",
            "Month-to-month customers show substantially higher churn "
            "than customers on longer-term contracts.",
            "insight_01_contract_type_churn.png",
        ),
        (
            "Tenure vs Churn",
            "Customers with shorter tenure are more likely to churn, "
            "highlighting the importance of early customer retention.",
            "insight_02_tenure_churn.png",
        ),
        (
            "Internet Service vs Churn",
            "Churn patterns differ across internet service categories. "
            "Service type can help identify customer segments for "
            "further investigation.",
            "insight_03_internet_service_churn.png",
        ),
        (
            "Payment Method vs Churn",
            "Electronic-check customers exhibit higher churn in this "
            "dataset than several other payment-method groups.",
            "insight_04_payment_method_churn.png",
        ),
        (
            "Online Security vs Churn",
            "Customers without online security show a higher churn "
            "tendency, making security adoption worth investigating.",
            "insight_05_online_security_churn.png",
        ),
        (
            "Monthly Charges vs Churn",
            "Churn varies with monthly charges. Pricing and perceived "
            "value may be useful areas for further analysis.",
            "insight_06_monthly_charges_churn.png",
        ),
        (
            "Dependents vs Churn",
            "Churn differs between customers with and without "
            "dependents, revealing a potentially useful customer segment.",
            "insight_07_dependents_churn.png",
        ),
        (
            "Partner Status vs Churn",
            "Partner status is associated with differences in churn "
            "behaviour across customer groups.",
            "insight_08_partner_churn.png",
        ),
        (
            "Senior Citizens vs Churn",
            "Churn patterns differ between senior and non-senior "
            "customers, which may inform tailored retention strategies.",
            "insight_09_senior_citizen_churn.png",
        ),
        (
            "Contract Type & Internet Service",
            "Combining contract and internet-service segments helps "
            "explore how churn patterns vary across customer groups.",
            "insight_10_contract_internet_churn.png",
        ),
    ]

    for start in range(0, len(eda_insights), 2):
        columns = st.columns(2, gap="large")

        for column, insight in zip(
            columns,
            eda_insights[start:start + 2]
        ):
            title, description, filename = insight
            chart_path = PROJECT_ROOT / "Reports" / filename

            with column:
                with st.container(border=True):
                    st.markdown(f"#### {title}")

                    if chart_path.exists():
                        st.image(
                            str(chart_path),
                            use_container_width=True,
                        )
                    else:
                        st.error(f"Chart not found: {filename}")

                    st.markdown(description)

    st.divider()
    st.caption(
        "Source: Telco Customer Churn dataset | "
        "EDA findings are descriptive associations, not proof of causation."
    )



elif page == "Interactive Analytics":
    import plotly.express as px

    st.title("🔎 Interactive Customer Analytics")
    st.markdown(
        "Explore customer segments and discover how churn patterns change."
    )

    st.divider()

    # Work with a separate copy so the original dataset remains unchanged
    analytics_df = df.copy()

    # Convert TotalCharges for numeric analysis
    analytics_df["TotalCharges"] = pd.to_numeric(
        analytics_df["TotalCharges"], errors="coerce"
    )

    # Create tenure groups using the same boundaries as preprocessing.py
    analytics_df["tenure_group"] = pd.cut(
        analytics_df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=[
            "0-6 months",
            "7-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months",
        ],
    )

    # Sidebar filters for interactive exploration
    st.subheader("🎛️ Customer Segment Filters")

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    with filter_col1:
        contract_options = sorted(analytics_df["Contract"].dropna().unique())
        selected_contracts = st.multiselect(
            "Contract type",
            contract_options,
            default=contract_options,
            key="analytics_contracts",
        )

    with filter_col2:
        internet_options = sorted(
            analytics_df["InternetService"].dropna().unique()
        )
        selected_internet = st.multiselect(
            "Internet service",
            internet_options,
            default=internet_options,
            key="analytics_internet",
        )

    with filter_col3:
        tenure_options = [
            "0-6 months",
            "7-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months",
        ]
        selected_tenure = st.multiselect(
            "Tenure group",
            tenure_options,
            default=tenure_options,
            key="analytics_tenure",
        )

    with filter_col4:
        payment_options = sorted(
            analytics_df["PaymentMethod"].dropna().unique()
        )
        selected_payment = st.multiselect(
            "Payment method",
            payment_options,
            default=payment_options,
            key="analytics_payment",
        )

    # Apply filters
    filtered_df = analytics_df[
        analytics_df["Contract"].isin(selected_contracts)
        & analytics_df["InternetService"].isin(selected_internet)
        & analytics_df["tenure_group"].astype(str).isin(selected_tenure)
        & analytics_df["PaymentMethod"].isin(selected_payment)
    ].copy()

    # Dynamic KPI calculations
    total_filtered = len(filtered_df)
    churned_filtered = int((filtered_df["Churn"] == "Yes").sum())
    retained_filtered = int((filtered_df["Churn"] == "No").sum())
    churn_rate_filtered = (
        churned_filtered / total_filtered * 100
        if total_filtered
        else 0
    )

    st.markdown("### 📊 Filtered Segment Overview")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Customers", f"{total_filtered:,}")
    k2.metric("Churned", f"{churned_filtered:,}")
    k3.metric("Churn Rate", f"{churn_rate_filtered:.2f}%")
    k4.metric("Retained", f"{retained_filtered:,}")

    if filtered_df.empty:
        st.warning(
            "No customers match this combination of filters. "
            "Select additional categories to see results."
        )
    else:
        st.caption(
            f"Showing {total_filtered:,} of {len(analytics_df):,} customers."
        )

        # Prepare reusable chart styling
        chart_template = "plotly_dark"

        st.divider()
        st.subheader("📈 Churn Patterns")

        # First row: contract and internet service
        chart1, chart2 = st.columns(2)

        with chart1:
            contract_summary = (
                filtered_df.groupby("Contract", observed=True)
                .agg(
                    Customers=("Churn", "size"),
                    Churned=("Churn", lambda s: (s == "Yes").sum()),
                )
                .reset_index()
            )
            contract_summary["Churn Rate (%)"] = (
                contract_summary["Churned"]
                / contract_summary["Customers"]
                * 100
            )

            fig_contract = px.bar(
                contract_summary,
                x="Contract",
                y="Churn Rate (%)",
                color="Churn Rate (%)",
                color_continuous_scale="Tealrose",
                hover_data=["Customers", "Churned"],
                title="Churn Rate by Contract",
                template=chart_template,
            )
            fig_contract.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Contract Type",
                yaxis_title="Churn Rate (%)",
            )
            st.plotly_chart(
                fig_contract, use_container_width=True
            )

        with chart2:
            internet_summary = (
                filtered_df.groupby("InternetService", observed=True)
                .agg(
                    Customers=("Churn", "size"),
                    Churned=("Churn", lambda s: (s == "Yes").sum()),
                )
                .reset_index()
            )
            internet_summary["Churn Rate (%)"] = (
                internet_summary["Churned"]
                / internet_summary["Customers"]
                * 100
            )

            fig_internet = px.bar(
                internet_summary,
                x="InternetService",
                y="Churn Rate (%)",
                color="Churn Rate (%)",
                color_continuous_scale="Tealrose",
                hover_data=["Customers", "Churned"],
                title="Churn Rate by Internet Service",
                template=chart_template,
            )
            fig_internet.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Internet Service",
                yaxis_title="Churn Rate (%)",
            )
            st.plotly_chart(
                fig_internet, use_container_width=True
            )

        # Second row: tenure and payment method
        chart3, chart4 = st.columns(2)

        with chart3:
            tenure_summary = (
                filtered_df.groupby("tenure_group", observed=False)
                .agg(
                    Customers=("Churn", "size"),
                    Churned=("Churn", lambda s: (s == "Yes").sum()),
                )
                .reset_index()
            )
            tenure_summary = tenure_summary[
                tenure_summary["Customers"] > 0
            ]
            tenure_summary["Churn Rate (%)"] = (
                tenure_summary["Churned"]
                / tenure_summary["Customers"]
                * 100
            )

            fig_tenure = px.line(
                tenure_summary,
                x="tenure_group",
                y="Churn Rate (%)",
                markers=True,
                hover_data=["Customers", "Churned"],
                title="Churn Rate by Tenure",
                template=chart_template,
            )
            fig_tenure.update_layout(
                xaxis_title="Tenure Group",
                yaxis_title="Churn Rate (%)",
            )
            st.plotly_chart(
                fig_tenure, use_container_width=True
            )

        with chart4:
            payment_summary = (
                filtered_df.groupby("PaymentMethod", observed=True)
                .agg(
                    Customers=("Churn", "size"),
                    Churned=("Churn", lambda s: (s == "Yes").sum()),
                )
                .reset_index()
            )
            payment_summary["Churn Rate (%)"] = (
                payment_summary["Churned"]
                / payment_summary["Customers"]
                * 100
            )

            fig_payment = px.bar(
                payment_summary,
                x="PaymentMethod",
                y="Churn Rate (%)",
                color="Churn Rate (%)",
                color_continuous_scale="Tealrose",
                hover_data=["Customers", "Churned"],
                title="Churn Rate by Payment Method",
                template=chart_template,
            )
            fig_payment.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Payment Method",
                yaxis_title="Churn Rate (%)",
            )
            st.plotly_chart(
                fig_payment, use_container_width=True
            )

        st.divider()

        # Filtered dataset preview and download
        with st.expander("🔍 View filtered customer data"):
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True,
            )

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download filtered data as CSV",
            data=csv_data,
            file_name="filtered_telco_churn_data.csv",
            mime="text/csv",
        )



elif page == "Churn Prediction":
    st.title("🤖 Churn Prediction Studio")
    st.markdown(
        "Predict customer churn using the trained machine learning pipeline."
    )
    st.divider()

    st.subheader("👤 Manual Customer Prediction")
    st.caption(
        "Enter a customer's service and account details to estimate churn risk."
    )

    # Keep only input widgets and the submit button inside the form.
    with st.form("manual_customer_form"):
        st.markdown("#### 📋 Customer & Account Details")
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
            partner = st.selectbox("Has Partner?", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"]
            )
            internet_service = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox(
                "Paperless Billing", ["Yes", "No"]
            )

        st.markdown("#### 🛡️ Services & Support")
        col3, col4 = st.columns(2)

        with col3:
            online_security = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )

        with col4:
            streaming_movies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=1.0,
            )
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=20000.0,
                value=840.0,
                step=10.0,
            )

        submitted = st.form_submit_button(
            "🔍 Predict Churn Risk",
            use_container_width=True,
            type="primary",
        )

    def add_engineered_features(customer_df):
        """Create the same engineered features used during model training."""
        customer_df = customer_df.copy()
        customer_df["tenure_group"] = pd.cut(
            customer_df["tenure"],
            bins=[-1, 6, 12, 24, 48, 72],
            labels=[
                "0-6 months",
                "7-12 months",
                "13-24 months",
                "25-48 months",
                "49-72 months",
            ],
        )
        customer_df["num_services"] = (
         customer_df["PhoneService"].eq("Yes").astype(int)
          + customer_df["MultipleLines"].eq("Yes").astype(int)
          + customer_df["InternetService"].ne("No").astype(int)
          + customer_df["OnlineSecurity"].eq("Yes").astype(int)
          + customer_df["OnlineBackup"].eq("Yes").astype(int)
          + customer_df["DeviceProtection"].eq("Yes").astype(int)
          + customer_df["TechSupport"].eq("Yes").astype(int)
          + customer_df["StreamingTV"].eq("Yes").astype(int)
          + customer_df["StreamingMovies"].eq("Yes").astype(int)
        )
        return customer_df


    def show_prediction(customer_df, heading="Prediction Result", actual_label=None):
        """Run the saved pipeline and display the prediction and probability."""
        prepared = add_engineered_features(customer_df)
        model.steps[-1][1].multi_class = "auto"
        prediction = int(model.predict(prepared)[0])
        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(prepared)[0]
            classes = list(model.classes_)
            if 1 in classes:
                probability = float(probabilities[classes.index(1)]) * 100

        st.markdown(f"### {heading}")
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            if prediction == 1:
                st.error("🚨 Model prediction: Likely to Churn")
            else:
                st.success("✅ Model prediction: Likely to Stay")
        with result_col2:
            if probability is not None:
                st.metric("Estimated Churn Probability", f"{probability:.1f}%")

        if actual_label is not None:
            st.info(f"Actual dataset label: **{actual_label}**")

    if submitted:
        manual_customer = pd.DataFrame(
            [{
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }]
        )
        show_prediction(manual_customer, "Manual Customer Result")

    st.divider()
    st.subheader("🧪 Test the Model")
    st.caption(
        "Try known churners, retained customers, or a random customer from the dataset."
    )

    test_col1, test_col2, test_col3 = st.columns(3)
    with test_col1:
        test_churn = st.button(
            "🚨 Test Churn Customer", use_container_width=True
        )
    with test_col2:
        test_retained = st.button(
            "✅ Test Retained Customer", use_container_width=True
        )
    with test_col3:
        test_random = st.button(
            "🎲 Test Random Customer", use_container_width=True
        )

    # Dataset test buttons are deliberately outside st.form.
    if test_churn or test_retained or test_random:
        if test_churn:
            candidates = df[df["Churn"].astype(str).str.strip().str.lower() == "yes"]
            heading = "Known Churn Customer Test"
        elif test_retained:
            candidates = df[df["Churn"].astype(str).str.strip().str.lower() == "no"]
            heading = "Known Retained Customer Test"
        else:
            candidates = df
            heading = "Random Customer Test"

        if candidates.empty:
            st.warning("No suitable customer records were found in the dataset.")
        else:
            sample = candidates.sample(n=1).copy()
            actual_label = str(sample["Churn"].iloc[0])
            customer_features = sample.drop(columns=["customerID", "Churn"], errors="ignore")
            customer_features["TotalCharges"] = pd.to_numeric(
                customer_features["TotalCharges"], errors="coerce"
            )
            show_prediction(
                customer_features,
                heading=heading,
                actual_label=actual_label,
            )



elif page == "Model Insights":
    st.title("🧠 Model Performance & Explainability")
    st.markdown(
        "Evaluate model performance, compare tuned models, "
        "and understand the factors influencing churn predictions."
    )

    # Report paths
    tuned_path = REPORTS_DIR / "tuned_model_comparison.csv"
    cv_path = REPORTS_DIR / "cross_validation_results.csv"
    test_path = REPORTS_DIR / "test_set_results.csv"
    shap_summary_path = REPORTS_DIR / "shap_summary_plot.png"
    shap_waterfall_path = REPORTS_DIR / "shap_waterfall_customer_1639.png"

    # Load reports safely
    def load_report(path):
        if path.exists():
            return pd.read_csv(path)
        return None

    tuned_results = load_report(tuned_path)
    cv_results = load_report(cv_path)
    test_results = load_report(test_path)

    # Model comparison
    st.subheader("📊 Tuned Model Comparison")

    if tuned_results is not None and not tuned_results.empty:
        st.dataframe(
            tuned_results,
            use_container_width=True,
            hide_index=True
        )

        # Show available test metrics for the selected model.
        # Metric values are read from the report rather than hardcoded.
        model_column = next(
            (
                col for col in tuned_results.columns
                if col.strip().lower() in
                ["model", "model_name", "classifier"]
            ),
            None
        )

        if model_column:
            st.caption(
                
                "model comparison"
            )
    else:
        st.warning("tuned_model_comparison.csv was not found or is empty.")

    # Cross-validation results
    st.subheader("🔁 Cross-Validation Results")

    if cv_results is not None and not cv_results.empty:
        st.dataframe(
            cv_results,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("cross_validation_results.csv was not found or is empty.")

    # Test-set results
    st.subheader("🎯 Test-Set Evaluation")

    if test_results is not None and not test_results.empty:
        st.dataframe(
            test_results,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("test_set_results.csv was not found or is empty.")

    # SHAP explanations
    st.divider()
    st.subheader("🔍 Model Explainability with SHAP")

    st.markdown(
        """
        **SHAP** helps explain how individual features influence model
        predictions. These plots explain model behaviour; they do not
        establish that a feature causes customer churn.
        """
    )

    if shap_summary_path.exists():
        st.markdown("#### Overall Feature Importance")
        st.caption(
            "Summary across customers: features with larger SHAP impact "
            "generally have a stronger influence on model predictions."
        )
        st.image(
            str(shap_summary_path),
            width=700,
        )
    else:
        st.warning("SHAP summary plot was not found.")

    if shap_waterfall_path.exists():
        st.markdown("#### Individual Customer Explanation")
        st.caption(
            "Example customer: row index 1639 from the test data. "
            "This plot explains how features contributed to that "
            "individual prediction."
        )
        st.image(
            str(shap_waterfall_path),
            width=700,
        )
    else:
        st.warning("SHAP waterfall plot was not found.")


# --------------------------------------------------
# 7. FOOTER
# --------------------------------------------------
st.divider()
st.caption("Telco Customer Churn | Machine Learning & Analytics")
