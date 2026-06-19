# ---------------- IMPORTS ----------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import xgboost as xgb

from statsmodels.tsa.arima.model import ARIMA


import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Retail AI Intelligence",
    layout="wide",
    page_icon="🛒"
)

# ---------------- UI STYLE ----------------
st.markdown("""
<style>

/* ==========================================================
   EXECUTIVE AI RETAIL INTELLIGENCE PLATFORM
   CONSULTING + ENTERPRISE GRADE UI
========================================================== */

:root{

    --bg-primary:#050816;
    --bg-secondary:#0b1220;
    --bg-card:rgba(255,255,255,0.04);

    --cyan:#22d3ee;
    --purple:#8b5cf6;
    --pink:#ec4899;

    --text-primary:#ffffff;
    --text-secondary:#94a3b8;

    --border:rgba(255,255,255,0.08);

}

/* ==========================================================
   APP BACKGROUND
========================================================== */

.stApp{

    background:
    radial-gradient(circle at top left,
        rgba(139,92,246,.15),
        transparent 35%),

    radial-gradient(circle at bottom right,
        rgba(34,211,238,.12),
        transparent 35%),

    linear-gradient(
        135deg,
        #050816,
        #0b1220,
        #111827
    );

    color:white;
}

/* ==========================================================
   MAIN WRAPPER
========================================================== */

.block-container{

    max-width:1400px;

    padding-top:1rem;

    background:
    rgba(255,255,255,0.02);

    backdrop-filter:blur(18px);

    border-radius:28px;

    border:1px solid rgba(255,255,255,.05);

    box-shadow:
    0 20px 60px rgba(0,0,0,.45);
}

/* ==========================================================
   PREMIUM TITLE
========================================================== */

h1{

    text-align:center;

    font-size:3.2rem;

    font-weight:900;

    letter-spacing:-1px;

    background:
    linear-gradient(
        90deg,
        #ffffff,
        #c4b5fd,
        #67e8f9
    );

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;

    margin-bottom:15px;
}

h1::after{

    content:"";

    display:block;

    width:220px;

    height:4px;

    margin:14px auto;

    border-radius:999px;

    background:
    linear-gradient(
        90deg,
        transparent,
        var(--purple),
        var(--cyan),
        transparent
    );
}

/* ==========================================================
   SECTION HEADERS
========================================================== */

h2,h3{

    color:white !important;

    padding-left:15px;

    border-left:5px solid var(--purple);

    margin-top:25px;

    font-weight:700;
}

/* ==========================================================
   SIDEBAR
========================================================== */

section[data-testid="stSidebar"]{

    background:
    linear-gradient(
        180deg,
        #0f172a,
        #111827
    );

    border-right:
    1px solid rgba(255,255,255,.08);

    box-shadow:
    inset -1px 0 0 rgba(255,255,255,.05);
}

/* ==========================================================
   KPI CARDS
========================================================== */

.metric-card{

    background:
    linear-gradient(
        145deg,
        rgba(255,255,255,.06),
        rgba(255,255,255,.02)
    );

    border-radius:24px;

    padding:28px;

    border:1px solid rgba(255,255,255,.08);

    backdrop-filter:blur(20px);

    transition:.35s ease;

    position:relative;

    overflow:hidden;
}

.metric-card::before{

    content:"";

    position:absolute;

    top:0;
    left:0;

    width:100%;
    height:4px;

    background:
    linear-gradient(
        90deg,
        var(--cyan),
        var(--purple),
        var(--pink)
    );
}

.metric-card:hover{

    transform:translateY(-6px);

    box-shadow:
    0 15px 40px rgba(139,92,246,.25);
}

/* ==========================================================
   TABS
========================================================== */

div[data-baseweb="tab-list"]{

    background:
    rgba(255,255,255,.03);

    border-radius:18px;

    padding:8px;

    border:1px solid rgba(255,255,255,.05);
}

button[data-baseweb="tab"]{

    border-radius:14px !important;

    font-weight:700 !important;

    color:#cbd5e1 !important;
}

button[data-baseweb="tab"][aria-selected="true"]{

    background:
    linear-gradient(
        135deg,
        #7c3aed,
        #9333ea
    ) !important;

    color:white !important;

    box-shadow:
    0 8px 25px rgba(124,58,237,.35);
}

/* ==========================================================
   TAB PANELS
========================================================== */

div[role="tabpanel"]{

    background:
    rgba(255,255,255,.03);

    border-radius:24px;

    padding:28px;

    border:
    1px solid rgba(255,255,255,.05);

    backdrop-filter:blur(12px);
}

/* ==========================================================
   EXECUTIVE BUTTONS
========================================================== */

.stButton > button{

    border:none;

    border-radius:14px;

    height:50px;

    font-weight:700;

    background:
    linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );

    transition:.3s;
}

.stButton > button:hover{

    transform:translateY(-2px);

    box-shadow:
    0 10px 30px rgba(124,58,237,.35);
}

/* ==========================================================
   PLOTLY CHART CONTAINER
========================================================== */

[data-testid="stPlotlyChart"]{

    background:
    rgba(255,255,255,.03);

    border-radius:24px;

    border:
    1px solid rgba(255,255,255,.06);

    padding:12px;

    box-shadow:
    0 10px 40px rgba(0,0,0,.25);
}

/* ==========================================================
   DATAFRAME
========================================================== */

[data-testid="stDataFrame"]{

    border-radius:18px;

    overflow:hidden;

    border:
    1px solid rgba(255,255,255,.08);
}

/* ==========================================================
   FILE UPLOADER
========================================================== */

[data-testid="stFileUploader"]{

    border:
    2px dashed rgba(139,92,246,.35);

    border-radius:20px;

    background:
    rgba(255,255,255,.02);
}

/* ==========================================================
   SCROLLBAR
========================================================== */

::-webkit-scrollbar{
    width:8px;
}

::-webkit-scrollbar-thumb{

    border-radius:999px;

    background:
    linear-gradient(
        var(--purple),
        var(--cyan)
    );
}

/* ==========================================================
   PREMIUM KPI CARDS
========================================================== */

.metric-card {

    background:
    linear-gradient(
        135deg,
        rgba(34,211,238,0.18),
        rgba(168,85,247,0.18),
        rgba(236,72,153,0.18)
    );

    backdrop-filter: blur(20px);

    border: 1px solid rgba(255,255,255,.08);

    border-radius: 24px;

    padding: 24px;

    min-height: 140px;

    position: relative;

    overflow: hidden;

    transition: all .35s ease;

    box-shadow:
        0 10px 30px rgba(0,0,0,.35),
        0 0 30px rgba(168,85,247,.15);
}

/* Top Accent Line */

.metric-card::before {

    content:"";

    position:absolute;

    top:0;
    left:0;

    width:100%;
    height:4px;

    background:
    linear-gradient(
        90deg,
        #22d3ee,
        #a855f7,
        #ec4899
    );
}

/* Glow Effect */

.metric-card::after {

    content:"";

    position:absolute;

    top:-50%;
    right:-50%;

    width:180px;
    height:180px;

    background:
    radial-gradient(
        rgba(168,85,247,.25),
        transparent 70%
    );

    pointer-events:none;
}

/* Hover */

.metric-card:hover {

    transform:
        translateY(-8px)
        scale(1.02);

    box-shadow:
        0 20px 50px rgba(168,85,247,.35),
        0 0 40px rgba(34,211,238,.15);
}
</style>
""", unsafe_allow_html=True) 

st.title("🚀 Retail AI Intelligence Platform")

# ==========================================================
# 📂 DATASET UPLOAD & INGESTION
# ==========================================================

# Upload retail dataset (CSV format only)
uploaded_file = st.file_uploader(
    "Upload your retail dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    # ======================================================
    # 📖 READ DATASET
    # ======================================================

    try:
        df = pd.read_csv(uploaded_file, encoding="latin1")

    except Exception as error:
        st.error(f"Unable to read CSV file: {error}")
        st.stop()

    # ======================================================
    # 🔍 AUTOMATIC COLUMN DETECTION
    # ======================================================

    def find_column(keywords):
        """
        Automatically identify dataset columns based on
        common retail naming conventions.
        """
        for column in df.columns:
            for keyword in keywords:
                if keyword in column.lower():
                    return column
        return None

    # Detect important columns
    date_col = find_column(["date", "time"])
    amount_col = find_column(["amount", "sales", "revenue", "price"])
    category_col = find_column(["category", "product", "item"])
    gender_col = find_column(["gender", "sex"])
    transaction_col = find_column(["transaction", "order", "bill", "invoice"])
    customer_col = find_column(["customer", "user", "client"])

    # Validate required fields
    if not date_col or not amount_col:
        st.error("Dataset must contain both Date and Sales-related columns.")
        st.stop()

    # ======================================================
    # 🧹 DATA STANDARDIZATION & CLEANING
    # ======================================================

    # Rename detected columns into standard names
    df.rename(
        columns={
            date_col: "Date",
            amount_col: "Total Amount"
        },
        inplace=True
    )

    # Remove duplicate column names if present
    df = df.loc[:, ~df.columns.duplicated()]

    # Convert date column to datetime format
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Handle accidental duplicate sales columns
    if isinstance(df["Total Amount"], pd.DataFrame):
        df["Total Amount"] = df["Total Amount"].iloc[:, 0]

    # Convert sales values into numeric format
    df["Total Amount"] = pd.to_numeric(
        df["Total Amount"],
        errors="coerce"
    )

    # Remove invalid records
    df.dropna(inplace=True)

    # ======================================================
    # 🏷 CREATE STANDARD BUSINESS FIELDS
    # ======================================================

    # Product Category
    df["Product Category"] = (
        df[category_col]
        if category_col
        else "General"
    )

    # Customer Gender
    df["Gender"] = (
        df[gender_col]
        if gender_col
        else "Unknown"
    )

    # Transaction Identifier
    if transaction_col:
        df["Transaction ID"] = df[transaction_col]
    else:
        df["Transaction ID"] = range(len(df))

    # Customer Identifier
    if customer_col:
        df["Customer ID"] = df[customer_col]
    else:
        df["Customer ID"] = 0

    # ======================================================
    # 🎛 SMART FILTER CONTROL PANEL
    # ======================================================

    st.sidebar.title("🎛 Smart Control Panel")

    # ----------------------------------
    # Category Search
    # ----------------------------------
    search_text = st.sidebar.text_input(
        "🔍 Search Category"
    )

    categories = df["Product Category"].unique()

    if search_text:
        categories = [
            category
            for category in categories
            if search_text.lower() in str(category).lower()
        ]

    # ----------------------------------
    # Multi-Select Filters
    # ----------------------------------
    selected_categories = st.sidebar.multiselect(
        "📦 Product Category",
        categories
    )

    selected_gender = st.sidebar.multiselect(
        "👤 Gender",
        df["Gender"].unique()
    )

    # ----------------------------------
    # Date Range Filter
    # ----------------------------------
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        [
            df["Date"].min(),
            df["Date"].max()
        ]
    )

    # ======================================================
    # ⚡ QUICK FILTERS
    # ======================================================

    st.sidebar.markdown("### ⚡ Quick Filters")

    # Last 7 Days Filter
    if st.sidebar.button("Last 7 Days"):
        df = df[
            df["Date"] >= (
                df["Date"].max() -
                pd.Timedelta(days=7)
            )
        ]

    # Top Revenue Category Filter
    if st.sidebar.button("Top Category"):

        top_category = (
            df.groupby("Product Category")["Total Amount"]
            .sum()
            .idxmax()
        )

        df = df[
            df["Product Category"] == top_category
        ]

    # ======================================================
    # 🎯 APPLY USER FILTERS
    # ======================================================

    filtered_df = df.copy()

    # Category Filter
    if selected_categories:
        filtered_df = filtered_df[
            filtered_df["Product Category"].isin(
                selected_categories
            )
        ]

    # Gender Filter
    if selected_gender:
        filtered_df = filtered_df[
            filtered_df["Gender"].isin(
                selected_gender
            )
        ]

    # Date Range Filter
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["Date"] >= pd.to_datetime(date_range[0])) &
            (filtered_df["Date"] <= pd.to_datetime(date_range[1]))
        ]

    # ======================================================
    # 📊 KEY PERFORMANCE INDICATORS (KPIs)
    # ======================================================

    st.subheader("📊 Key Metrics")

    total_revenue = filtered_df["Total Amount"].sum()

    total_transactions = (
        filtered_df["Transaction ID"]
        .nunique()
    )

    average_order_value = (
        total_revenue /
        max(total_transactions, 1)
    )

    total_customers = (
        filtered_df["Customer ID"]
        .nunique()
    )

    # KPI Cards Layout
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
       st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:14px;color:#94a3b8;">💰 Total Revenue</div>
            <div style="font-size:32px;font-weight:800;color:white;margin-top:10px;">
                ${total_revenue:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with kpi2:
       st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:14px;color:#94a3b8;">🛒 Transactions</div>
            <div style="font-size:32px;font-weight:800;color:white;margin-top:10px;">
                {total_transactions:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with kpi3:
       st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:14px;color:#94a3b8;">📦 Avg Order Value</div>
            <div style="font-size:32px;font-weight:800;color:white;margin-top:10px;">
                ${average_order_value:,.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with kpi4:
       st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:14px;color:#94a3b8;">👥 Customers</div>
            <div style="font-size:32px;font-weight:800;color:white;margin-top:10px;">
                {total_customers:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    def generate_executive_insights(df, filtered_df):
        
        insights = []

        total_revenue = filtered_df["Total Amount"].sum()

        avg_daily = (
        filtered_df.groupby("Date")["Total Amount"].sum().mean()
        )

        top_category = (
        filtered_df.groupby("Product Category")["Total Amount"]
        .sum()
        .idxmax()
        )

        category_share = (
        filtered_df.groupby("Product Category")["Total Amount"]
        .sum()
        .max()
        / total_revenue
        ) * 100

        # ---------------- EXECUTIVE SUMMARY ----------------
        insights.append(
        f"📊 Revenue performance shows total sales of "
        f"${total_revenue:,.0f} with an average daily run-rate of "
        f"${avg_daily:,.0f}."
    )

    # ---------------- DRIVER INSIGHT ----------------
        insights.append(
        f"📦 Primary growth driver is '{top_category}', "
        f"contributing {category_share:.1f}% of total revenue."
    )

    # ---------------- RISK ANALYSIS ----------------
        volatility = filtered_df.groupby("Date")["Total Amount"].std().mean()

        if volatility > filtered_df["Total Amount"].mean() * 0.5:
          insights.append(
            "⚠️ High revenue volatility detected — demand is unstable "
            "and may require pricing or inventory stabilization."
        )
        else:
          insights.append( 
            "✅ Revenue pattern is stable with low volatility risk."
        )

    # ---------------- OPPORTUNITY ----------------
        low_categories = (
          filtered_df.groupby("Product Category")["Total Amount"]
        .sum()
        .sort_values()
        .head(1)
        .index[0]
    )

        insights.append(
        f"💡 Upside opportunity identified in '{low_categories}' category "
        "through targeted promotions or bundling strategies."
    )

    # ---------------- ACTION RECOMMENDATIONS ----------------
        insights.append(
        "🎯 Recommended Action: Focus marketing investment on top category, "
        "optimize pricing strategy, and monitor weekly demand shifts."
    )

        return insights

    # ==========================================================
    # 📑 APPLICATION NAVIGATION TABS
    # ==========================================================

    tabs = st.tabs([
        "📊 Dashboard",
        "📂 Dataset Overview",
        "📈 Forecast",
        "🤖 ML Models",
        "⚖️ Model Comparison",
        "🔮 Predictor",
        "🚨 Anomaly Detection",
        "💡 Reinforcement Learning",
        "🧠 Deep Reinforcement Learning"
    ])


        # ======================================================
    # 📊 DASHBOARD
    # ======================================================

    with tabs[0]:

        category_revenue = (
            filtered_df
            .groupby("Product Category")["Total Amount"]
            .sum()
            .sort_values(ascending=False)
        )

        top_category = (
            category_revenue.idxmax()
            if not category_revenue.empty
            else "N/A"
        )

        # --------------------------------------------------
        # CATEGORY PERFORMANCE
        # --------------------------------------------------

        st.subheader("📦 Product Category Performance")

        left_col, right_col = st.columns(2)

        with left_col:

            revenue_chart = px.bar(
                category_revenue.reset_index(),
                x="Product Category",
                y="Total Amount",
                text_auto=".2s",
                title="Revenue by Category"
            )

            revenue_chart.update_layout(
                template="plotly_dark",
                height=450,
                xaxis_title="Category",
                yaxis_title="Revenue"
            )

            st.plotly_chart(
                revenue_chart,
                use_container_width=True
            )

        with right_col:

            category_share_chart = px.pie(
                values=category_revenue.values,
                names=category_revenue.index,
                hole=0.60,
                title="Revenue Contribution"
            )

            category_share_chart.update_layout(
                template="plotly_dark",
                height=450
            )

            st.plotly_chart(
                category_share_chart,
                use_container_width=True
            )

        # --------------------------------------------------
        # DAILY REVENUE TREND
        # --------------------------------------------------

        st.subheader("📈 Revenue Trend Analysis")

        daily_revenue = (
            filtered_df
            .groupby("Date")["Total Amount"]
            .sum()
        )

        moving_average_7d = (
            daily_revenue
            .rolling(window=7)
            .mean()
        )

        moving_average_30d = (
            daily_revenue
            .rolling(window=30)
            .mean()
        )

        trend_chart = go.Figure()

        trend_chart.add_trace(
            go.Scatter(
                x=daily_revenue.index,
                y=daily_revenue.values,
                mode="lines",
                name="Revenue"
            )
        )

        trend_chart.add_trace(
            go.Scatter(
                x=moving_average_7d.index,
                y=moving_average_7d.values,
                mode="lines",
                name="7-Day Avg"
            )
        )

        trend_chart.add_trace(
            go.Scatter(
                x=moving_average_30d.index,
                y=moving_average_30d.values,
                mode="lines",
                name="30-Day Avg"
            )
        )

        trend_chart.update_layout(
            template="plotly_dark",
            height=550,
            hovermode="x unified",
            title="Revenue Trend Over Time"
        )

        st.plotly_chart(
            trend_chart,
            use_container_width=True
        )

        # --------------------------------------------------
        # MONTHLY TREND ANALYSIS
        # --------------------------------------------------

        st.subheader("📅 Monthly Revenue Analysis")

        monthly_revenue = (
            filtered_df
            .groupby(
                pd.Grouper(
                    key="Date",
                    freq="M"
                )
            )["Total Amount"]
            .sum()
        )

        monthly_chart = px.line(
            monthly_revenue,
            title="Monthly Revenue Trend",
            markers=True
        )

        monthly_chart.update_layout(
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(
            monthly_chart,
            use_container_width=True
        )

        # --------------------------------------------------
        # CATEGORY PERFORMANCE TABLE
        # --------------------------------------------------

        st.subheader("📋 Category Performance Breakdown")

        category_table = (
            filtered_df
            .groupby("Product Category")
            .agg(
                Revenue=("Total Amount", "sum"),
                Orders=("Transaction ID", "count")
            )
            .sort_values(
                "Revenue",
                ascending=False
            )
        )

        st.dataframe(
            category_table,
            use_container_width=True
        )

        # --------------------------------------------------
        # EXECUTIVE INSIGHTS
        # --------------------------------------------------

        st.subheader("📌 Executive Insights")

        if not category_revenue.empty:

            top_category_sales = category_revenue.max()

            contribution_pct = (
                top_category_sales
                /
                category_revenue.sum()
            ) * 100

            best_day = daily_revenue.idxmax()

            best_day_sales = daily_revenue.max()

            st.markdown(
                f"""
                <div class="metric-card">

                <h4>Business Performance Summary</h4>

                <p>
                Top Performing Category:
                <b>{top_category}</b>
                </p>

                <p>
                Category Contribution:
                <b>{contribution_pct:.1f}%</b>
                of total revenue
                </p>

                <p>
                Highest Revenue Day:
                <b>{best_day.date()}</b>
                (${best_day_sales:,.0f})
                </p>

                <p>
                Recommendation:
                Focus inventory allocation,
                marketing campaigns,
                and promotional spending
                on high-performing categories.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ======================================================
    # 📂 DATA QUALITY & EXPLORATORY ANALYSIS
    # ======================================================

    with tabs[1]:

        st.subheader("📂 Data Quality & Exploratory Analysis")

        # --------------------------------------------------
        # DATASET HEALTH SUMMARY
        # --------------------------------------------------

        total_rows, total_columns = df.shape

        missing_cells = df.isnull().sum().sum()

        duplicate_rows = df.duplicated().sum()

        numeric_columns = (
            df.select_dtypes(include=np.number)
            .columns
        )

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>{total_rows:,}</h3>
                    <p>Total Records</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with kpi2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>{total_columns}</h3>
                    <p>Total Features</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with kpi3:

            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>{missing_cells:,}</h3>
                    <p>Missing Values</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with kpi4:

            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>{duplicate_rows:,}</h3>
                    <p>Duplicate Records</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")

        # --------------------------------------------------
        # DATA TYPES
        # --------------------------------------------------

        st.subheader("🔍 Data Structure")

        dtype_df = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes.astype(str).values
        })

        st.dataframe(
            dtype_df,
            use_container_width=True
        )

        # --------------------------------------------------
        # MISSING VALUE ANALYSIS
        # --------------------------------------------------

        st.subheader("❗ Missing Value Assessment")

        missing_values = (
            df.isnull()
            .sum()
            .sort_values(ascending=False)
        )

        missing_values = (
            missing_values[missing_values > 0]
        )

        if not missing_values.empty:

            missing_chart = px.bar(
                x=missing_values.index,
                y=missing_values.values,
                title="Missing Values by Feature"
            )

            missing_chart.update_layout(
                template="plotly_dark",
                height=450
            )

            st.plotly_chart(
                missing_chart,
                use_container_width=True
            )

        else:

            st.success(
                "✅ Dataset contains no missing values."
            )

        # --------------------------------------------------
        # DESCRIPTIVE STATISTICS
        # --------------------------------------------------

        st.subheader("📊 Statistical Profile")

        st.dataframe(
            df.describe().round(2),
            use_container_width=True
        )

        # --------------------------------------------------
        # OUTLIER ANALYSIS
        # --------------------------------------------------

        st.subheader("🚨 Outlier Assessment")

        outlier_results = []

        for column in numeric_columns:

            q1 = df[column].quantile(0.25)

            q3 = df[column].quantile(0.75)

            iqr = q3 - q1

            lower_bound = (
                q1 - (1.5 * iqr)
            )

            upper_bound = (
                q3 + (1.5 * iqr)
            )

            outlier_count = len(
                df[
                    (df[column] < lower_bound)
                    |
                    (df[column] > upper_bound)
                ]
            )

            outlier_results.append({
                "Feature": column,
                "Outliers": outlier_count
            })

        outlier_df = pd.DataFrame(
            outlier_results
        )

        outlier_chart = px.bar(
            outlier_df,
            x="Feature",
            y="Outliers",
            title="Outlier Distribution"
        )

        outlier_chart.update_layout(
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(
            outlier_chart,
            use_container_width=True
        )

        st.dataframe(
            outlier_df,
            use_container_width=True
        )

        # --------------------------------------------------
        # FEATURE DISTRIBUTIONS
        # --------------------------------------------------

        st.subheader("📈 Feature Distribution Analysis")

        selected_feature = st.selectbox(
            "Select Numerical Feature",
            numeric_columns
        )

        distribution_chart = px.histogram(
            df,
            x=selected_feature,
            nbins=40,
            marginal="box",
            title=f"{selected_feature} Distribution"
        )

        distribution_chart.update_layout(
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(
            distribution_chart,
            use_container_width=True
        )

        # --------------------------------------------------
        # CORRELATION ANALYSIS
        # --------------------------------------------------

        st.subheader("🔗 Correlation Analysis")

        if len(numeric_columns) > 1:

            correlation_matrix = (
                df[numeric_columns]
                .corr()
            )

            correlation_chart = px.imshow(
                correlation_matrix,
                text_auto=".2f",
                aspect="auto",
                title="Feature Correlation Matrix"
            )

            correlation_chart.update_layout(
                template="plotly_dark",
                height=650
            )

            st.plotly_chart(
                correlation_chart,
                use_container_width=True
            )

        # ======================================================
    # 📈 SALES FORECASTING TAB
    # ======================================================

    with tabs[2]:

        st.subheader("📈 Sales Forecasting")

        # --------------------------------------------------
        # Prepare Daily Time Series Data
        # --------------------------------------------------

        daily_sales_series = (
            filtered_df
            .groupby("Date")["Total Amount"]
            .sum()
            .asfreq("D")
            .fillna(0)
        )

        smoothed_sales = (
            daily_sales_series
            .rolling(window=7)
            .mean()
            .bfill()
        )

        # --------------------------------------------------
        # Forecast Generation
        # --------------------------------------------------

        if len(smoothed_sales) == 0:

            st.warning(
                "No data available for forecasting."
            )

            forecast_values = pd.Series(dtype=float)

            confidence_interval = pd.DataFrame()

        elif len(smoothed_sales) > 20:

            forecast_model = ARIMA(
                smoothed_sales,
                order=(2, 1, 2)
            ).fit()

            forecast_result = (
                forecast_model
                .get_forecast(steps=30)
            )

            forecast_values = (
                forecast_result
                .predicted_mean
            )

            confidence_interval = (
                forecast_result
                .conf_int()
            )

        else:

            forecast_values = pd.Series(
                [smoothed_sales.mean()] * 30,
                index=pd.date_range(
                    start=smoothed_sales.index.max()
                    + pd.Timedelta(days=1),
                    periods=30,
                    freq="D"
                )
            )

            confidence_interval = pd.DataFrame()

        # --------------------------------------------------
        # KPI ROW
        # --------------------------------------------------

        col1, col2, col3 = st.columns(3)

        forecast_revenue = float(forecast_values.sum())
        avg_forecast = float(forecast_values.mean())

        recent_mean = float(smoothed_sales.tail(30).mean()) if len(smoothed_sales) > 0 else 1
        forecast_mean = float(forecast_values.mean())

        trend = ((forecast_mean - recent_mean) / max(recent_mean, 1)) * 100


        with col1:
            st.markdown(
        "<div class='metric-card'>"
        "<h4>📊 Forecast Revenue</h4>"
        f"<h2>${forecast_revenue:,.1f}</h2>"
        "</div>",
        unsafe_allow_html=True
    )

        with col2:
             st.markdown(
        "<div class='metric-card'>"
        "<h4>📈 Daily Forecast</h4>"
        f"<h2>${avg_forecast:,.1f}</h2>"
        "</div>",
        unsafe_allow_html=True
    )

        with col3:
             st.markdown(
        "<div class='metric-card'>"
        "<h4>🚀 Growth Trend</h4>"
        f"<h2>{trend:.1f}%</h2>"
        "</div>",
        unsafe_allow_html=True
    )

        # --------------------------------------------------
        # Forecast Chart
        # --------------------------------------------------

        forecast_chart = go.Figure()

        forecast_chart.add_trace(
            go.Scatter(
                x=smoothed_sales.index,
                y=smoothed_sales.values,
                mode="lines",
                name="Historical",
                line=dict(width=4)
            )
        )

        forecast_chart.add_trace(
            go.Scatter(
                x=forecast_values.index,
                y=forecast_values.values,
                mode="lines",
                name="Forecast",
                line=dict(
                    width=4,
                    dash="dash"
                )
            )
        )

        if not confidence_interval.empty:

            forecast_chart.add_trace(
                go.Scatter(
                    x=list(forecast_values.index)
                    + list(forecast_values.index[::-1]),

                    y=list(confidence_interval.iloc[:, 0])
                    + list(confidence_interval.iloc[:, 1][::-1]),

                    fill="toself",

                    line=dict(width=0),

                    hoverinfo="skip",

                    showlegend=False
                )
            )

        forecast_chart.update_layout(
            template="plotly_dark",
            height=600,
            hovermode="x unified",
            margin=dict(
                l=20,
                r=20,
                t=30,
                b=20
            ),
            legend=dict(
                orientation="h",
                y=1.05
            )
        )

        st.plotly_chart(
            forecast_chart,
            use_container_width=True
        )

        # --------------------------------------------------
        # Forecast Data
        # --------------------------------------------------

        forecast_df = pd.DataFrame({
            "Date": forecast_values.index,
            "Forecast Sales": forecast_values.values.round(2)
        })

        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=True
        )

    # ======================================================
    # 🤖 MACHINE LEARNING PERFORMANCE CENTER
    # ======================================================

    with tabs[3]:

        st.subheader("🤖 Machine Learning Performance Center")

        # --------------------------------------------------
        # FEATURE ENGINEERING
        # --------------------------------------------------

        ml_df = filtered_df.copy()

        ml_df["Day"] = ml_df["Date"].dt.day

        ml_df["Month"] = ml_df["Date"].dt.month

        ml_df = ml_df.drop(
            ["Transaction ID", "Date"],
            axis=1,
            errors="ignore"
        )

        ml_df = (
            pd.get_dummies(
                ml_df,
                drop_first=True
            )
            .fillna(0)
        )

        # --------------------------------------------------
        # VALIDATION
        # --------------------------------------------------

        if len(ml_df) < 20:

            st.warning(
                "Insufficient data available for model training."
            )

            st.stop()

        # --------------------------------------------------
        # FEATURES & TARGET
        # --------------------------------------------------

        X = ml_df.drop(
            "Total Amount",
            axis=1
        )

        y = ml_df["Total Amount"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        # --------------------------------------------------
        # MODELS
        # --------------------------------------------------

        models = {

            "Random Forest": RandomForestRegressor(
                n_estimators=200,
                random_state=42
            ),

            "XGBoost": xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=200,
                random_state=42
            ),

            "Support Vector Regression": SVR()
        }

        model_results = {}

        trained_models = {}

        # --------------------------------------------------
        # TRAINING
        # --------------------------------------------------

        with st.spinner(
            "Training machine learning models..."
        ):

            for model_name, model in models.items():

                model.fit(
                    X_train,
                    y_train
                )

                y_pred = model.predict(
                    X_test
                )

                trained_models[model_name] = model

                model_results[model_name] = {

                    "MAE":
                    mean_absolute_error(
                        y_test,
                        y_pred
                    ),

                    "RMSE":
                    np.sqrt(
                        mean_squared_error(
                            y_test,
                            y_pred
                        )
                    ),

                    "R² Score":
                    r2_score(
                        y_test,
                        y_pred
                    )
                }

        # --------------------------------------------------
        # PERFORMANCE SUMMARY
        # --------------------------------------------------

        st.subheader(
            "📊 Model Performance Comparison"
        )

        results_df = (
            pd.DataFrame(model_results)
            .T
            .reset_index()
            .rename(
                columns={
                    "index": "Model"
                }
            )
        )

        st.dataframe(
            results_df.round(4),
            use_container_width=True
        )

        # --------------------------------------------------
        # MODEL COMPARISON CHART
        # --------------------------------------------------

        performance_chart = px.bar(
            results_df,
            x="Model",
            y="R² Score",
            text_auto=".3f",
            title="Model Accuracy Comparison"
        )

        performance_chart.update_layout(
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(
            performance_chart,
            use_container_width=True
        )

        # --------------------------------------------------
        # BEST MODEL
        # --------------------------------------------------

        best_model_name = results_df.loc[
            results_df["R² Score"].idxmax(),
            "Model"
        ]

        best_model = trained_models[
            best_model_name
        ]

        st.markdown(
            f"""
            <div class="metric-card">

            <h3>🏆 Best Performing Model</h3>

            <h2>{best_model_name}</h2>

            <p>
            Highest predictive performance
            based on R² evaluation metric.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.session_state["model"] = (
            best_model
        )

        st.session_state["columns"] = (
            X.columns
        )

        # --------------------------------------------------
        # ACTUAL VS PREDICTED
        # --------------------------------------------------

        st.subheader(
            "📈 Actual vs Predicted Performance"
        )

        y_pred_best = best_model.predict(
            X_test
        )

        actual_vs_pred = go.Figure()

        actual_vs_pred.add_trace(
            go.Scatter(
                x=y_test,
                y=y_pred_best,
                mode="markers",
                name="Predictions"
            )
        )

        actual_vs_pred.update_layout(
            template="plotly_dark",
            height=550,
            xaxis_title="Actual Sales",
            yaxis_title="Predicted Sales"
        )

        st.plotly_chart(
            actual_vs_pred,
            use_container_width=True
        )

        # --------------------------------------------------
        # FEATURE IMPORTANCE
        # --------------------------------------------------

        st.subheader(
            "🧠 Feature Importance Analysis"
        )

        try:

            if hasattr(
                best_model,
                "feature_importances_"
            ):

                feature_importance = pd.Series(
                    best_model.feature_importances_,
                    index=X.columns
                )

                feature_importance = (
                    feature_importance
                    .sort_values(
                        ascending=False
                    )
                    .head(15)
                )

                importance_chart = px.bar(
                    x=feature_importance.values,
                    y=feature_importance.index,
                    orientation="h",
                    title="Top Drivers of Sales"
                )

                importance_chart.update_layout(
                    template="plotly_dark",
                    height=600
                )

                st.plotly_chart(
                    importance_chart,
                    use_container_width=True
                )

            else:

                st.info(
                    "Feature importance is not available for this model."
                )

        except Exception:

            st.warning(
                "Unable to generate feature importance."
            )

        # --------------------------------------------------
        # BUSINESS INSIGHTS
        # --------------------------------------------------

        st.subheader(
            "📌 Business Insights"
        )

        st.markdown(
            f"""
            <div class="metric-card">

            <h4>AI Model Findings</h4>

            <p>
            ✔ Best Model:
            <b>{best_model_name}</b>
            </p>

            <p>
            ✔ Strongest predictors were identified
            through feature importance analysis.
            </p>

            <p>
            ✔ Predictive analytics can support
            demand forecasting, inventory planning,
            and revenue optimization.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    # ======================================================
    # ⚖️ MODEL COMPARISON
    # ======================================================

    with tabs[4]:

        st.subheader("⚖️ Model Comparison Dashboard")

        # --------------------------------------------------
        # MODEL SUMMARY
        # --------------------------------------------------

        best_model_name = results_df.loc[
            results_df["R² Score"].idxmax(),
            "Model"
        ]

        best_r2 = results_df["R² Score"].max()

        lowest_rmse = results_df["RMSE"].min()

        k1, k2, k3 = st.columns(3)

        with k1:

            st.metric(
                "🏆 Best Model",
                best_model_name
            )

        with k2:

            st.metric(
                "📈 Best R²",
                f"{best_r2:.3f}"
            )

        with k3:

            st.metric(
                "🎯 Lowest RMSE",
                f"{lowest_rmse:.2f}"
            )

        st.markdown("---")

        # --------------------------------------------------
        # PERFORMANCE COMPARISON
        # --------------------------------------------------

        comparison_chart = go.Figure()

        comparison_chart.add_trace(
            go.Bar(
                name="MAE",
                x=results_df["Model"],
                y=results_df["MAE"]
            )
        )

        comparison_chart.add_trace(
            go.Bar(
                name="RMSE",
                x=results_df["Model"],
                y=results_df["RMSE"]
            )
        )

        comparison_chart.add_trace(
            go.Bar(
                name="R² Score",
                x=results_df["Model"],
                y=results_df["R² Score"]
            )
        )

        comparison_chart.update_layout(
            title="Model Performance Benchmark",
            template="plotly_dark",
            barmode="group",
            height=550,
            hovermode="x unified",
            xaxis_title="Machine Learning Models",
            yaxis_title="Performance Score",
            legend_title="Metrics"
        )

        st.plotly_chart(
            comparison_chart,
            use_container_width=True
        )

        # --------------------------------------------------
        # MODEL PERFORMANCE RADAR
        # --------------------------------------------------

        st.subheader("🎯 Model Performance Radar")

        radar_df = results_df.copy()

        radar_df["MAE Score"] = (
            radar_df["MAE"].max()
            - radar_df["MAE"]
        )

        radar_df["RMSE Score"] = (
            radar_df["RMSE"].max()
            - radar_df["RMSE"]
        )

        radar_chart = go.Figure()

        for _, row in radar_df.iterrows():

            radar_chart.add_trace(
                go.Scatterpolar(
                    r=[
                        row["R² Score"],
                        row["MAE Score"],
                        row["RMSE Score"]
                    ],
                    theta=[
                        "R² Score",
                        "MAE Score",
                        "RMSE Score"
                    ],
                    fill="toself",
                    name=row["Model"]
                )
            )

        radar_chart.update_layout(
            template="plotly_dark",
            height=650,
            title="Multi-Metric Model Evaluation",
            showlegend=True,
            polar=dict(
                radialaxis=dict(
                    visible=True
                )
            )
        )

        st.plotly_chart(
            radar_chart,
            use_container_width=True
        )

        # --------------------------------------------------
        # KEY FINDINGS
        # --------------------------------------------------

        st.markdown(
            f"""
            <div class="metric-card">

            <h3>📌 Key Findings</h3>

            <p>
            🏆 <b>{best_model_name}</b> delivered the
            strongest predictive performance across
            the evaluated machine learning models.
            </p>

            <p>
            📈 The model achieved the highest R² score,
            indicating better ability to explain sales
            variability in the dataset.
            </p>

            <p>
            🎯 Lower prediction errors support more
            accurate demand forecasting, inventory
            planning, and revenue optimization.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    # ======================================================
    # 🔮 SALES PREDICTION TAB
    # ======================================================

    with tabs[5]:

        st.subheader("🔮 Sales Predictor")

        # --------------------------------------------------
        # User Input Controls
        # --------------------------------------------------

        selected_day = st.slider(
            "Day",
            min_value=1,
            max_value=31,
            value=15
        )

        selected_month = st.slider(
            "Month",
            min_value=1,
            max_value=12,
            value=6
        )

        # --------------------------------------------------
        # Sales Prediction
        # --------------------------------------------------

        if st.button("Predict Sales"):

            prediction_sample = pd.DataFrame(
                [[selected_day, selected_month]],
                columns=["day", "month"]
            )

            # Align input features with training data
            for column in st.session_state["columns"]:

                if column not in prediction_sample.columns:
                    prediction_sample[column] = 0

            prediction_sample = prediction_sample[
                st.session_state["columns"]
            ]

            predicted_sales = (
                st.session_state["model"]
                .predict(prediction_sample)[0]
            )

            st.success(
                f"💰 Predicted Sales: ${predicted_sales:.2f}"
            )

        # --------------------------------------------------
        # What-If Scenario Analysis
        # --------------------------------------------------

        st.subheader("🔍 What-If Analysis")

        if st.checkbox("Run Scenario Simulation"):

            scenario_predictions = []

            for future_day in range(1, 8):

                scenario_sample = pd.DataFrame(
                    [[future_day, selected_month]],
                    columns=["day", "month"]
                )

                for column in st.session_state["columns"]:

                    if column not in scenario_sample.columns:
                        scenario_sample[column] = 0

                scenario_sample = scenario_sample[
                    st.session_state["columns"]
                ]

                forecast_value = (
                    st.session_state["model"]
                    .predict(scenario_sample)[0]
                )

                scenario_predictions.append(
                    forecast_value
                )

            scenario_chart = px.line(
                x=list(range(1, 8)),
                y=scenario_predictions,
                title="Next 7 Days Predicted Sales Trend",
                labels={
                    "x": "Day",
                    "y": "Predicted Sales"
                }
            )

            st.plotly_chart(
                scenario_chart,
                use_container_width=True
            )

        # ======================================================
    # 🚨 RISK & ANOMALY MONITORING
    # ======================================================

    with tabs[6]:

        st.subheader("🚨 Risk & Anomaly Monitoring")

        # --------------------------------------------------
        # DAILY SALES ANALYSIS
        # --------------------------------------------------

        daily_sales = (
            filtered_df
            .groupby("Date")["Total Amount"]
            .sum()
        )

        if len(daily_sales) < 2:

            st.warning(
                "Insufficient data available for anomaly detection."
            )

        else:

            z_scores = (
                daily_sales - daily_sales.mean()
            ) / daily_sales.std()

            anomalies = daily_sales[
                abs(z_scores) > 2
            ]

            # --------------------------------------------------
            # RISK SUMMARY KPIs
            # --------------------------------------------------

            total_days = len(daily_sales)

            anomaly_count = len(anomalies)

            risk_rate = (
                anomaly_count / total_days * 100
            )

            highest_sale = daily_sales.max()

            k1, k2, k3 = st.columns(3)

            with k1:

                st.metric(
                    "🚨 Anomalies",
                    anomaly_count
                )

            with k2:

                st.metric(
                    "📊 Risk Rate",
                    f"{risk_rate:.1f}%"
                )

            with k3:

                st.metric(
                    "💰 Peak Sales",
                    f"${highest_sale:,.0f}"
                )

            st.markdown("---")

            # --------------------------------------------------
            # ANOMALY VISUALIZATION
            # --------------------------------------------------

            anomaly_chart = go.Figure()

            anomaly_chart.add_trace(
                go.Scatter(
                    x=daily_sales.index,
                    y=daily_sales.values,
                    mode="lines",
                    name="Daily Sales"
                )
            )

            anomaly_chart.add_trace(
                go.Scatter(
                    x=anomalies.index,
                    y=anomalies.values,
                    mode="markers",
                    marker=dict(
                        size=12
                    ),
                    name="Anomalies"
                )
            )

            anomaly_chart.update_layout(
                template="plotly_dark",
                height=550,
                title="Sales Anomaly Detection",
                hovermode="x unified"
            )

            st.plotly_chart(
                anomaly_chart,
                use_container_width=True
            )

            # --------------------------------------------------
            # ANOMALY REPORT
            # --------------------------------------------------

            st.subheader(
                "📋 Risk Assessment Report"
            )

            if not anomalies.empty:

                severity_scores = (
                    abs(z_scores[anomalies.index]) * 10
                ).round(2)

                anomaly_report = pd.DataFrame({

                    "Date":
                    anomalies.index,

                    "Sales":
                    anomalies.values,

                    "Severity Score":
                    severity_scores.values
                })

                st.dataframe(
                    anomaly_report,
                    use_container_width=True
                )

                max_severity = (
                    anomaly_report[
                        "Severity Score"
                    ].max()
                )

                if max_severity > 25:

                    risk_level = "High Risk"

                elif max_severity > 15:

                    risk_level = "Medium Risk"

                else:

                    risk_level = "Low Risk"

                st.markdown(
                    f"""
                    <div class="metric-card">

                    <h3>⚠️ Risk Assessment</h3>

                    <p>
                    Current Risk Level:
                    <b>{risk_level}</b>
                    </p>

                    <p>
                    Detected anomalies may indicate
                    unusual purchasing behavior,
                    inventory discrepancies,
                    promotional spikes,
                    or operational issues.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.success(
                    "✅ No significant anomalies detected."
                )

            # --------------------------------------------------
            # SALES HEATMAP
            # --------------------------------------------------

            st.subheader(
                "🔥 Sales Activity Heatmap"
            )

            heatmap_df = filtered_df.copy()

            heatmap_df["day"] = (
                heatmap_df["Date"].dt.day
            )

            heatmap_df["month"] = (
                heatmap_df["Date"].dt.month
            )

            heatmap_matrix = (
                heatmap_df.pivot_table(
                    values="Total Amount",
                    index="day",
                    columns="month",
                    aggfunc="sum"
                )
            )

            heatmap_chart = px.imshow(
                heatmap_matrix,
                aspect="auto",
                title="Sales Distribution by Day & Month"
            )

            heatmap_chart.update_layout(
                template="plotly_dark",
                height=650
            )

            st.plotly_chart(
                heatmap_chart,
                use_container_width=True
            )

            # --------------------------------------------------
            # KEY FINDINGS
            # --------------------------------------------------

            st.markdown(
                f"""
                <div class="metric-card">

                <h3>📌 Key Findings</h3>

                <p>
                Total Anomalies Detected:
                <b>{anomaly_count}</b>
                </p>

                <p>
                Risk monitoring identified unusual
                sales patterns requiring review.
                </p>

                <p>
                Continuous anomaly tracking helps
                reduce revenue leakage and improve
                operational visibility.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )
        # ======================================================
    # 💡 AI PRICING OPTIMIZATION ENGINE
    # ======================================================

    with tabs[7]:

        st.subheader("💡 AI Pricing Optimization Engine")

        # --------------------------------------------------
        # DATA PREPARATION
        # --------------------------------------------------

        rl_df = filtered_df.copy()

        daily_sales = (
            rl_df
            .groupby("Date")["Total Amount"]
            .sum()
        )

        if len(daily_sales) < 10:

            st.warning(
                "Insufficient data available for Reinforcement Learning analysis."
            )

        else:

            demand_levels = pd.qcut(
                daily_sales,
                q=3,
                labels=[
                    "Low",
                    "Medium",
                    "High"
                ]
            )

            states = demand_levels.values

            actions = [
                "Decrease Price",
                "Keep Same",
                "Increase Price"
            ]

            state_mapping = {
                "Low": 0,
                "Medium": 1,
                "High": 2
            }

            n_states = 3
            n_actions = 3

            # --------------------------------------------------
            # Q-TABLE INITIALIZATION
            # --------------------------------------------------

            q_table = np.zeros(
                (
                    n_states,
                    n_actions
                )
            )

            learning_rate = 0.10

            discount_factor = 0.90

            exploration_rate = 0.20

            # --------------------------------------------------
            # Q-LEARNING TRAINING
            # --------------------------------------------------

            training_episodes = 200

            for _ in range(training_episodes):

                for i in range(len(states) - 1):

                    current_state = (
                        state_mapping[
                            states[i]
                        ]
                    )

                    if (
                        np.random.rand()
                        <
                        exploration_rate
                    ):

                        action = (
                            np.random.randint(
                                n_actions
                            )
                        )

                    else:

                        action = (
                            np.argmax(
                                q_table[
                                    current_state
                                ]
                            )
                        )

                    price_factor = [
                        0.90,
                        1.00,
                        1.10
                    ][action]

                    reward = (
                        daily_sales.iloc[i]
                        *
                        price_factor
                    )

                    next_state = (
                        state_mapping[
                            states[i + 1]
                        ]
                    )

                    q_table[
                        current_state,
                        action
                    ] += (
                        learning_rate
                        *
                        (
                            reward
                            +
                            discount_factor
                            *
                            np.max(
                                q_table[
                                    next_state
                                ]
                            )
                            -
                            q_table[
                                current_state,
                                action
                            ]
                        )
                    )

                        # --------------------------------------------------
            # AI DECISION CENTER
            # --------------------------------------------------

            optimal_actions = np.argmax(
                q_table,
                axis=1
            )

            action_names = [
                actions[i]
                for i in optimal_actions
            ]

            k1, k2, k3 = st.columns(3)

            with k1:

                st.markdown(
                    f"""
                    <div class="metric-card">

                    <h4>📉 Low Demand</h4>

                    <h3>{action_names[0]}</h3>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with k2:

                st.markdown(
                    f"""
                    <div class="metric-card">

                    <h4>📊 Medium Demand</h4>

                    <h3>{action_names[1]}</h3>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with k3:

                st.markdown(
                    f"""
                    <div class="metric-card">

                    <h4>📈 High Demand</h4>

                    <h3>{action_names[2]}</h3>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("---")

            # --------------------------------------------------
            # AI PRICING STRATEGY MATRIX
            # --------------------------------------------------

            st.subheader(
                "🧠 AI Pricing Strategy Matrix"
            )

            q_table_df = pd.DataFrame(
                q_table,
                index=[
                    "Low Demand",
                    "Medium Demand",
                    "High Demand"
                ],
                columns=actions
            )

            st.dataframe(
                q_table_df.round(2),
                use_container_width=True
            )

            # --------------------------------------------------
            # AI REVENUE OPTIMIZATION HEATMAP
            # --------------------------------------------------

            strategy_chart = px.imshow(
                q_table_df,
                text_auto=".1f",
                aspect="auto",
                title="AI Revenue Optimization Matrix"
            )

            strategy_chart.update_layout(
                template="plotly_dark",
                height=600
            )

            st.plotly_chart(
                strategy_chart,
                use_container_width=True
            )

            # --------------------------------------------------
            # BUSINESS RECOMMENDATION
            # --------------------------------------------------

            st.markdown(
                """
                <div class="metric-card">

                <h3>🚀 AI Pricing Recommendation</h3>

                <p>
                The Reinforcement Learning agent evaluates
                demand conditions and identifies pricing
                strategies that maximize long-term revenue.
                </p>

                <p>
                These recommendations can support dynamic
                pricing, promotion planning, inventory
                optimization, and revenue growth initiatives.
                </p>

                <p>
                AI-driven pricing decisions help retailers
                respond proactively to changing market demand.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )
        
    # ======================================================
    # 🧠 DEEP REINFORCEMENT LEARNING TAB
    # ======================================================

    with tabs[8]:

        st.subheader(
            "🧠 Deep Reinforcement Learning (DQN for Dynamic Pricing)"
        )

        # --------------------------------------------------
        # Data Preparation
        # --------------------------------------------------

        rl_df = filtered_df.copy()

        daily_sales = (
            rl_df.groupby("Date")["Total Amount"]
            .sum()
            .values
        )

        st.write(
            f"Number of days in dataset: {len(daily_sales)}"
        )

        # Normalize sales values
        sales = (
            daily_sales - np.mean(daily_sales)
        ) / (
            np.std(daily_sales) + 1e-5
        )

        # --------------------------------------------------
        # Pricing Environment
        # --------------------------------------------------

        class PricingEnv:

            def __init__(self, sales_data):

                self.sales = sales_data
                self.t = 0
                self.n = len(sales_data)
                self.inventory = 1000

            def reset(self):

                self.t = 0
                self.inventory = 1000

                return np.array([
                    self.sales[self.t],
                    self.inventory / 1000
                ])

            def step(self, action):

                current_demand = (
                    self.sales[self.t]
                )

                # Pricing Actions
                # 0 = Decrease Price
                # 1 = Keep Same
                # 2 = Increase Price

                price_change = [
                    -0.10,
                    0.00,
                    0.10
                ][action]

                # Demand Elasticity
                elasticity = 1.5

                adjusted_demand = (
                    current_demand *
                    (1 - price_change * elasticity)
                )

                adjusted_demand = max(
                    adjusted_demand,
                    0
                )

                # Revenue Calculation
                revenue = (
                    adjusted_demand *
                    (1 + price_change)
                )

                # Inventory Update
                sold_units = (
                    adjusted_demand * 10
                )

                self.inventory -= sold_units

                self.inventory = max(
                    self.inventory,
                    0
                )

                # Inventory Penalty
                inventory_penalty = 0

                if self.inventory < 100:
                    inventory_penalty = -20

                # Final Reward
                reward = (
                    revenue +
                    inventory_penalty
                )

                # Move to Next Step
                self.t += 1

                done = (
                    self.t >= self.n - 1
                    or self.inventory <= 0
                )

                if not done:

                    next_state = np.array([
                        self.sales[self.t],
                        self.inventory / 1000
                    ])

                else:

                    next_state = np.array([
                        0,
                        0
                    ])

                return (
                    next_state,
                    reward,
                    done
                )

        # --------------------------------------------------
        # Data Validation
        # --------------------------------------------------

        if len(daily_sales) < 10:

            st.warning(
                "Need at least 10 days of sales data "
                "for DQN training."
            )

            st.stop()

        env = PricingEnv(sales)

        # --------------------------------------------------
        # Deep Q-Network Architecture
        # --------------------------------------------------

        class DQN(nn.Module):

            def __init__(self):

                super().__init__()

                self.network = nn.Sequential(
                    nn.Linear(2, 64),
                    nn.ReLU(),

                    nn.Linear(64, 64),
                    nn.ReLU(),

                    nn.Linear(64, 32),
                    nn.ReLU(),

                    nn.Linear(32, 3)
                )

            def forward(self, x):

                return self.network(x)

        dqn_model = DQN()

        optimizer = optim.Adam(
            dqn_model.parameters(),
            lr=0.001
        )

        loss_function = nn.MSELoss()

        # --------------------------------------------------
        # Experience Replay Memory
        # --------------------------------------------------

        replay_memory = deque(
            maxlen=2000
        )

        gamma = 0.95

        epsilon = 1.00
        epsilon_decay = 0.995
        epsilon_min = 0.01

        # --------------------------------------------------
        # Training Controls
        # --------------------------------------------------

        training_episodes = st.slider(
            "Training Episodes",
            min_value=5,
            max_value=20,
            value=10
        )

        # --------------------------------------------------
        # DQN Training
        # --------------------------------------------------

        if st.button("Train DQN Model"):

            rewards_history = []

            for episode in range(
                training_episodes
            ):

                state = env.reset()

                total_reward = 0

                while True:

                    state_tensor = (
                        torch.FloatTensor(state)
                    )

                    # Epsilon-Greedy Policy
                    if random.random() < epsilon:

                        action = random.randint(
                            0,
                            2
                        )

                    else:

                        with torch.no_grad():

                            action = torch.argmax(
                                dqn_model(state_tensor)
                            ).item()

                    (
                        next_state,
                        reward,
                        done
                    ) = env.step(action)

                    replay_memory.append(
                        (
                            state,
                            action,
                            reward,
                            next_state,
                            done
                        )
                    )

                    state = next_state

                    total_reward += reward

                    # Experience Replay Training
                    if len(replay_memory) > 16:

                        batch = random.sample(
                            replay_memory,
                            16
                        )

                        for (
                            s,
                            a,
                            r,
                            ns,
                            d
                        ) in batch:

                            s = torch.FloatTensor(s)
                            ns = torch.FloatTensor(ns)

                            target = r

                            if not d:

                                target += (
                                    gamma *
                                    torch.max(
                                        dqn_model(ns)
                                    ).item()
                                )

                            target_values = (
                                dqn_model(s)
                                .clone()
                                .detach()
                            )

                            target_values[a] = target

                            loss = loss_function(
                                dqn_model(s),
                                target_values.detach()
                            )

                            optimizer.zero_grad()

                            loss.backward()

                            optimizer.step()

                    if done:
                        break

                rewards_history.append(
                    total_reward
                )

                if epsilon > epsilon_min:

                    epsilon *= epsilon_decay

            # --------------------------------------------------
            # Training Results
            # --------------------------------------------------

            st.write(
                "### 📈 Training Reward Progress"
            )

            learning_curve = px.line(
                y=rewards_history,
                title="DQN Learning Curve"
            )

            st.plotly_chart(
                learning_curve,
                use_container_width=True
            )

            st.success(
                "✅ Deep RL Agent trained successfully!"
            )

            # --------------------------------------------------
            # Learned Policy
            # --------------------------------------------------

            st.write(
                "### 🧠 Learned Pricing Policy"
            )

            actions_map = [
                "Decrease Price",
                "Keep Same",
                "Increase Price"
            ]

            test_demands = np.linspace(
                float(min(sales)),
                float(max(sales)),
                10
            )

            inventory_levels = [
                1.0,
                0.7,
                0.4
            ]

            policy_results = []

            for demand in test_demands:

                for inventory in inventory_levels:

                    state = torch.FloatTensor([
                        demand,
                        inventory
                    ])

                    with torch.no_grad():

                        best_action = (
                            torch.argmax(
                                dqn_model(state)
                            ).item()
                        )

                    policy_results.append({
                        "Demand":
                            round(demand, 2),

                        "Inventory":
                            int(inventory * 1000),

                        "Recommended Action":
                            actions_map[
                                best_action
                            ]
                    })

            policy_df = pd.DataFrame(
                policy_results
            )

            st.dataframe(
                policy_df,
                use_container_width=True
            )

            st.info(
                "💡 This Deep Q-Network learns "
                "dynamic pricing strategies based "
                "on demand patterns and inventory levels."
            )

# ==========================================================
# NO DATA UPLOADED
# ==========================================================

else:

    st.info(
        "Upload a dataset to begin 🚀"
    )
