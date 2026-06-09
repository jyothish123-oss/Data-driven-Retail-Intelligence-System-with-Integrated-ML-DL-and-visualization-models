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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Retail AI Intelligence",
    layout="wide",
    page_icon="🛒"
)

# ---------------- UI STYLE ----------------
st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
body {
    background: linear-gradient(135deg, #0f111a, #1c1f2e, #2a1a3f);
    color: #e6e6fa;
    font-family: 'Segoe UI', sans-serif;
}

/* ---------- HEADER ---------- */
h1 {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    background: linear-gradient(90deg, #00f5a0, #c471f5, #fa71cd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background: rgba(30, 20, 50, 0.95);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* ---------- METRIC CARDS (MERGED STYLE) ---------- */
.metric-card {
    background: linear-gradient(135deg,#1f4037,#99f2c8,#5f2c82);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    color: white;
    font-weight: bold;
    font-size: 16px;
    backdrop-filter: blur(12px);
    box-shadow: 0 6px 25px rgba(0,0,0,0.6);
    border: 1px solid rgba(255,255,255,0.1);
    transition: 0.3s;
}

.metric-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.8);
}

/* ---------- TABS ---------- */
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #aaa !important;
    font-weight: 600;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #c471f5 !important;
    border-bottom: 2px solid #c471f5 !important;
}

/* ---------- BUTTONS ---------- */
.stButton>button {
    background: linear-gradient(135deg, #00f5a0, #c471f5, #fa71cd);
    color: black;
    border-radius: 12px;
    border: none;
    padding: 10px 18px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.08);
    box-shadow: 0 0 15px rgba(196,113,245,0.6);
}

/* ---------- FILE UPLOADER ---------- */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.06);
    padding: 15px;
    border-radius: 12px;
    border: 1px dashed rgba(255,255,255,0.3);
}

/* ---------- DATAFRAME ---------- */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ---------- SCROLLBAR ---------- */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(#00f5a0, #c471f5);
    border-radius: 10px;
}

/* ---------- EXTRA PREMIUM EFFECT ---------- */
.metric-card, .stButton>button {
    backdrop-filter: blur(10px);
}

/* ---------- GLOW TEXT EFFECT ---------- */
h1::after {
    content: "";
    display: block;
    height: 2px;
    width: 120px;
    margin: 10px auto;
    background: linear-gradient(90deg,#00f5a0,#c471f5,#fa71cd);
    border-radius: 10px;
}
/* ---------- TAB BACKGROUND ENHANCEMENT ---------- */
div[data-baseweb="tab-list"] {
    background: linear-gradient(135deg, rgba(95,44,130,0.4), rgba(196,113,245,0.3));
    padding: 8px;
    border-radius: 12px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}

/* Individual Tabs */
button[data-baseweb="tab"] {
    border-radius: 10px !important;
    margin: 2px;
    transition: 0.3s;
}

/* Active Tab Glow */
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #5f2c82, #c471f5) !important;
    color: white !important;
    box-shadow: 0 0 12px rgba(196,113,245,0.7);
}

/* Hover Effect */
button[data-baseweb="tab"]:hover {
    background: rgba(196,113,245,0.2) !important;
    color: #fff !important;
}
/* ---------- GLOBAL GLASS CONTAINER STYLE ---------- */
.block-container {
    background: rgba(40, 20, 70, 0.55);
    padding: 20px;
    border-radius: 16px;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.6);
}

/* ---------- ALL SECTIONS (HEADERS / SUBHEADERS) ---------- */
h2, h3 {
    background: linear-gradient(90deg, rgba(95,44,130,0.25), rgba(196,113,245,0.25));
    padding: 8px 12px;
    border-radius: 10px;
    backdrop-filter: blur(8px);
}

/* ---------- TABS CONTENT AREA ---------- */
section.main > div {
    background: rgba(40, 20, 70, 0.35);
    border-radius: 16px;
    padding: 15px;
    backdrop-filter: blur(12px);
}

/* ---------- PLOTLY CHART CONTAINERS ---------- */
[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,0.05);
    padding: 12px;
    border-radius: 14px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}

/* ---------- DATAFRAME TABLE ---------- */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.05);
    padding: 10px;
    border-radius: 12px;
    backdrop-filter: blur(10px);
}

/* ---------- INPUTS (SLIDERS / SELECT / TEXT) ---------- */
.stSlider, .stSelectbox, .stTextInput, .stMultiSelect {
    background: rgba(255,255,255,0.05);
    padding: 10px;
    border-radius: 10px;
    backdrop-filter: blur(8px);
}

/* ---------- KPI ROW ALIGNMENT ---------- */
div[data-testid="column"] {
    background: transparent;
}

/* ---------- SUCCESS / WARNING / ERROR BOXES ---------- */
.stAlert {
    border-radius: 12px;
    backdrop-filter: blur(10px);
}

/* ---------- TAB PANELS SMOOTH LOOK ---------- */
div[role="tabpanel"] {
    background: rgba(40, 20, 70, 0.4);
    padding: 20px;
    border-radius: 16px;
    backdrop-filter: blur(12px);
}

/* ---------- EXTRA GLOW EFFECT ---------- */
.block-container, [data-testid="stPlotlyChart"], .metric-card {
    box-shadow: 0 0 25px rgba(196,113,245,0.15);
}

</style>
""", unsafe_allow_html=True)

st.title("🚀 Retail AI Intelligence Platform")

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload your retail dataset (CSV)", type=["csv"])

if file is not None:

    try:
        df = pd.read_csv(file, encoding="latin1")
    except Exception as e:
        st.error(f"Cannot read CSV: {e}")
        st.stop() 

    # ---------------- AUTO COLUMN DETECTION ----------------
    def find_col(keywords):
        for col in df.columns:
            for k in keywords:
                if k in col.lower():
                    return col
        return None

    date_col = find_col(["date", "time"])
    amount_col = find_col(["amount", "sales", "revenue", "price"])
    category_col = find_col(["category", "product", "item"])
    gender_col = find_col(["gender", "sex"])
    transaction_col = find_col(["transaction", "order", "bill", "invoice"])
    customer_col = find_col(["customer", "user", "client"])

    if not date_col or not amount_col:
        st.error("Dataset must contain Date and Sales column")
        st.stop()

    # ---------------- STANDARDIZE ----------------
    df.rename(columns={
        date_col: "Date",
        amount_col: "Total Amount"
    }, inplace=True)

    # ✅ FIX 1: remove duplicate columns (SAFE ADD)
    df = df.loc[:, ~df.columns.duplicated()]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # ✅ FIX 2: handle multi-column issue (SAFE ADD)
    if isinstance(df["Total Amount"], pd.DataFrame):
        df["Total Amount"] = df["Total Amount"].iloc[:, 0]

    df["Total Amount"] = pd.to_numeric(df["Total Amount"], errors="coerce")

    df.dropna(inplace=True)

    # Safe defaults
    df["Product Category"] = df[category_col] if category_col else "General"
    df["Gender"] = df[gender_col] if gender_col else "Unknown"

    if transaction_col:
        df["Transaction ID"] = df[transaction_col]
    else:
        df["Transaction ID"] = range(len(df))

    if customer_col:
        df["Customer ID"] = df[customer_col]
    else:
        df["Customer ID"] = 0

    # ================= PREMIUM FILTERS =================
    st.sidebar.title("🎛 Smart Control Panel")

    search = st.sidebar.text_input("🔍 Search Category")
    categories = df["Product Category"].unique()

    if search:
        categories = [c for c in categories if search.lower() in str(c).lower()]

    category = st.sidebar.multiselect("📦 Category", categories)
    gender = st.sidebar.multiselect("👤 Gender", df["Gender"].unique())

    date_range = st.sidebar.date_input(
        "📅 Date Range",
        [df["Date"].min(), df["Date"].max()]
    )

    st.sidebar.markdown("### ⚡ Quick Filters")

    if st.sidebar.button("Last 7 Days"):
        df = df[df["Date"] >= df["Date"].max() - pd.Timedelta(days=7)]

    if st.sidebar.button("Top Category"):
        top = df.groupby("Product Category")["Total Amount"].sum().idxmax()
        df = df[df["Product Category"] == top]

    filtered_df = df.copy()

    if category:
        filtered_df = filtered_df[filtered_df["Product Category"].isin(category)]

    if gender:
        filtered_df = filtered_df[filtered_df["Gender"].isin(gender)]

    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["Date"] >= pd.to_datetime(date_range[0])) &
            (filtered_df["Date"] <= pd.to_datetime(date_range[1]))
        ]

    # ---------------- KPIs ----------------
    st.subheader("📊 Key Metrics")

    revenue = filtered_df['Total Amount'].sum()
    transactions = filtered_df['Transaction ID'].nunique()
    avg_order = revenue / max(transactions, 1)
    customers = filtered_df['Customer ID'].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'>💰 Revenue<br>${revenue:,.0f}</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'>🛒 Transactions<br>{transactions}</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'>📦 Avg Order<br>${avg_order:.2f}</div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'>👥 Customers<br>{customers}</div>", unsafe_allow_html=True)

    # ---------------- TABS ----------------
    tabs = st.tabs([
        "📊 Dashboard",
        "📂 Dataset Overview",
        "📈 Forecast",
        "🤖 ML Models",
        "⚖️ Comparison",
        "🔮 Predictor",
        "🚨 Anomaly Detection",
        "💡 Reinforcement Learning",
        "💡 Deep Reinforcement Learning"
    ])

    # ---------------- DASHBOARD ----------------
    with tabs[0]:
        st.subheader("📊 Category Analysis")

        col1, col2 = st.columns(2)

        cat_sales = filtered_df.groupby('Product Category')['Total Amount'].sum().sort_values(ascending=False)

        col1.plotly_chart(px.bar(cat_sales, x=cat_sales.index, y=cat_sales.values,
                                 title="Revenue by Category"), use_container_width=True)

        col2.plotly_chart(px.pie(values=cat_sales.values, names=cat_sales.index,
                                 hole=0.5, title="Category Share"), use_container_width=True)

        st.subheader("📈 Sales Trend")

        ts = filtered_df.groupby('Date')['Total Amount'].sum()
        rolling = ts.rolling(7).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Daily"))
        fig.add_trace(go.Scatter(x=rolling.index, y=rolling.values, name="7-Day Avg"))
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- NEW DATASET OVERVIEW TAB ----------------
    with tabs[1]:
        st.subheader("📂 Dataset Overview & EDA")

        st.write("### 📏 Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        st.write("### 🔍 Data Types")
        st.dataframe(df.dtypes)

        st.write("### ❗ Missing Values")
        missing = df.isnull().sum()
        st.dataframe(missing[missing > 0])

        st.write("### 📊 Statistical Summary")
        st.dataframe(df.describe())

        st.write("### 🚨 Outlier Detection (IQR Method)")
        numeric_cols = df.select_dtypes(include=np.number).columns
        outlier_summary = {}

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            outlier_summary[col] = len(outliers)

        st.dataframe(pd.DataFrame(outlier_summary.items(), columns=["Column","Outliers Count"]))

        st.write("### 📈 Distributions")
        for col in numeric_cols[:3]:
            fig = px.histogram(df, x=col, title=f"{col} Distribution")
            st.plotly_chart(fig, use_container_width=True)

    # ---------------- REST OF YOUR CODE (UNCHANGED) ----------------
    with tabs[2]:
        st.subheader("📈 30-Day Forecast")
        ts = filtered_df.groupby('Date')['Total Amount'].sum().asfreq('D').fillna(0)
        ts_smooth = ts.rolling(7).mean().bfill()

        if len(ts_smooth) > 20:
            model = ARIMA(ts_smooth, order=(2,1,2)).fit()
            forecast = model.get_forecast(30)
            pred = forecast.predicted_mean
            conf = forecast.conf_int()
        else:
            pred = pd.Series([ts_smooth.mean()] * 30)
            conf = pd.DataFrame()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_smooth.index, y=ts_smooth.values, name="Actual"))
        fig.add_trace(go.Scatter(x=pred.index, y=pred.values, name="Forecast"))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("🤖 Model Training")
        df_ml = filtered_df.copy()
        df_ml['day'] = df_ml['Date'].dt.day
        df_ml['month'] = df_ml['Date'].dt.month

        df_ml = df_ml.drop(['Transaction ID', 'Date'], axis=1, errors='ignore')
        df_ml = pd.get_dummies(df_ml, drop_first=True).fillna(0)

        X = df_ml.drop('Total Amount', axis=1)
        y = df_ml['Total Amount']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        models = {
            "Random Forest": RandomForestRegressor(),
            "XGBoost": xgb.XGBRegressor(objective='reg:squarederror'),
            "SVM": SVR()
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            results[name] = {
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                "R2": r2_score(y_test, pred)
            }

        res_df = pd.DataFrame(results).T.reset_index().rename(columns={'index':'Model'})
        st.dataframe(res_df)

        best_model_name = res_df.loc[res_df['R2'].idxmax(),'Model']
        st.success(f"Best Model: {best_model_name}")

        st.session_state["model"] = models[best_model_name]
        st.session_state["columns"] = X.columns
        
                # ---------------- AI INSIGHTS ----------------
        st.subheader("🧠 AI Insights")

        best_model = models[best_model_name]

        try:
            if hasattr(best_model, "feature_importances_"):
                importance = pd.Series(best_model.feature_importances_, index=X.columns)
                importance = importance.sort_values(ascending=False)[:10]

                fig = px.bar(importance, x=importance.values, y=importance.index,
                             orientation='h', title="Top Feature Importance")
                st.plotly_chart(fig, use_container_width=True)

                st.success("Top factors influencing sales identified!")
            else:
                st.info("Feature importance not available for this model.")
        except:
            st.warning("Could not compute feature importance.")

    with tabs[4]:
        st.subheader("⚖️ Model Comparison")
        fig = px.bar(res_df, x='Model', y=['MAE','RMSE','R2'], barmode='group')
        st.plotly_chart(fig, use_container_width=True)
                # ---------------- BEST MODEL BADGE ----------------
        st.subheader("🏆 Model Winner")
        st.success(f"🏆 {best_model_name} performs best based on R² score!")

        # ---------------- RADAR CHART ----------------
        st.subheader("📡 Model Performance Radar")

        try:
            radar_df = res_df.set_index("Model")
            fig = go.Figure()

            for model in radar_df.index:
                fig.add_trace(go.Scatterpolar(
                    r=radar_df.loc[model].values,
                    theta=radar_df.columns,
                    fill='toself',
                    name=model
                ))

            fig.update_layout(polar=dict(radialaxis=dict(visible=True)))
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("Radar chart not available.")

    with tabs[5]:
        st.subheader("🔮 Sales Predictor")
        day = st.slider("Day", 1, 31, 15)
        month = st.slider("Month", 1, 12, 6)

        if st.button("Predict"):
            sample = pd.DataFrame([[day, month]], columns=["day","month"])

            for col in st.session_state["columns"]:
                if col not in sample:
                    sample[col] = 0

            sample = sample[st.session_state["columns"]]
            pred = st.session_state["model"].predict(sample)[0]
            st.success(f"Predicted Sales: ${pred:.2f}")
                    # ---------------- WHAT-IF ANALYSIS ----------------
        st.subheader("🔍 What-If Analysis")

        if st.checkbox("Run Scenario Simulation"):
            scenarios = []
            for d in range(1, 8):
                temp = pd.DataFrame([[d, month]], columns=["day", "month"])
                for col in st.session_state["columns"]:
                    if col not in temp:
                        temp[col] = 0
                temp = temp[st.session_state["columns"]]

                val = st.session_state["model"].predict(temp)[0]
                scenarios.append(val)

            fig = px.line(x=list(range(1,8)), y=scenarios,
                          title="Next 7 Days Prediction Trend")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[6]:
        st.subheader("🚨 Theft Detection")

        ts = filtered_df.groupby('Date')['Total Amount'].sum()
        z = (ts - ts.mean()) / ts.std()
        anomalies = ts[abs(z) > 2]

        st.dataframe(anomalies)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Sales"))
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies.values,
                                 mode='markers', name="Anomaly"))
        st.plotly_chart(fig, use_container_width=True)
                # ---------------- ANOMALY SEVERITY ----------------
        st.subheader("⚠️ Anomaly Severity")

        if not anomalies.empty:
            severity = (abs(z[anomalies.index]) * 10).round(2)
            alert_df = pd.DataFrame({
                "Date": anomalies.index,
                "Sales": anomalies.values,
                "Severity Score": severity.values
            })

            st.dataframe(alert_df)

            high_risk = alert_df[alert_df["Severity Score"] > 20]

            if not high_risk.empty:
                st.error("🚨 High Risk Theft Detected!")
            else:
                st.warning("⚠️ Moderate anomalies detected.")
        else:
            st.success("✅ No anomalies detected!")

        # ---------------- HEATMAP VIEW ----------------
        st.subheader("🔥 Sales Heatmap")

        heatmap_data = filtered_df.copy()
        heatmap_data["day"] = heatmap_data["Date"].dt.day
        heatmap_data["month"] = heatmap_data["Date"].dt.month

        pivot = heatmap_data.pivot_table(values="Total Amount",
                                         index="day",
                                         columns="month",
                                         aggfunc="sum")

        fig = px.imshow(pivot, title="Sales Heatmap")
        st.plotly_chart(fig, use_container_width=True)
    with tabs[7]:
        st.subheader("💡 Reinforcement Learning: Dynamic Pricing")

        st.markdown("### 🎯 Goal: Maximize Revenue using Smart Pricing")

        # ---------------- PREPARE DATA ----------------
        rl_df = filtered_df.copy()

        daily_sales = rl_df.groupby("Date")["Total Amount"].sum()
        demand_levels = pd.qcut(daily_sales, q=3, labels=["Low", "Medium", "High"])

        states = demand_levels.values

        actions = ["Decrease Price", "Keep Same", "Increase Price"]

    # Encode states
        state_map = {"Low":0, "Medium":1, "High":2}
        n_states = 3
        n_actions = 3

        # ---------------- Q-TABLE ----------------
        Q = np.zeros((n_states, n_actions))

        alpha = 0.1   # learning rate
        gamma = 0.9   # discount
        epsilon = 0.2 # exploration

    # ---------------- TRAINING ----------------
        episodes = 200

        for _ in range(episodes):
          for i in range(len(states)-1):

            state = state_map[states[i]]

            # epsilon-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(Q[state])

            # Simulated reward
            price_factor = [0.9, 1.0, 1.1][action]
            reward = daily_sales.iloc[i] * price_factor

            next_state = state_map[states[i+1]]

            # Q update
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

    # ---------------- RESULTS ----------------
        st.write("### 📊 Learned Pricing Strategy (Q-Table)")
        q_df = pd.DataFrame(Q, index=["Low","Medium","High"], columns=actions)
        st.dataframe(q_df)

    # ---------------- BEST ACTION ----------------
        st.write("### 🧠 Optimal Strategy")

        best_actions = q_df.idxmax(axis=1)
        for state, action in best_actions.items():
          st.success(f"For {state} demand → {action}")

    # ---------------- VISUALIZATION ----------------
        fig = px.imshow(Q,
                    x=actions,
                    y=["Low","Medium","High"],
                    title="Q-Table Heatmap")

        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 RL Agent learns how to adjust pricing dynamically to maximize revenue.")
        
    with tabs[8]:
        st.subheader("🧠 Deep Reinforcement Learning (DQN for Dynamic Pricing)")
        
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import random
        from collections import deque

    # ---------------- DATA PREP ----------------
        rl_df = filtered_df.copy()
        daily_sales = rl_df.groupby("Date")["Total Amount"].sum().values
        st.write("Number of days in dataset:", len(daily_sales))

    # Normalize
        sales = (daily_sales - np.mean(daily_sales)) / (np.std(daily_sales) + 1e-5)


    # ---------------- ENVIRONMENT ----------------
        class PricingEnv:

           def __init__(self, sales):

             self.sales = sales
             self.t = 0
             self.n = len(sales)

             # Simulated inventory
             self.inventory = 1000

           def reset(self):

             self.t = 0
             self.inventory = 1000

             return np.array([
             self.sales[self.t],
             self.inventory / 1000
             ])

           def step(self, action):

             current_demand = self.sales[self.t]

        # ---------------- ACTIONS ----------------
        # 0 = decrease
        # 1 = same
        # 2 = increase

             price_change = [-0.10, 0.0, 0.10][action]

        # ---------------- DEMAND ELASTICITY ----------------
        # higher price -> lower demand
             elasticity = 1.5

             adjusted_demand = current_demand * (
             1 - price_change * elasticity
             )

             adjusted_demand = max(adjusted_demand, 0)

             # ---------------- REVENUE ----------------
             revenue = adjusted_demand * (1 + price_change)

             # ---------------- INVENTORY EFFECT ----------------
             sold_units = adjusted_demand * 10

             self.inventory -= sold_units
             self.inventory = max(self.inventory, 0)

             # ---------------- INVENTORY PENALTY ----------------
             inventory_penalty = 0

             if self.inventory < 100:
              inventory_penalty = -20

             # ---------------- FINAL REWARD ----------------
             reward = revenue + inventory_penalty

             # ---------------- NEXT STEP ----------------
             self.t += 1

             done = (
             self.t >= self.n - 1
             or self.inventory <= 0
             )

             next_state = (
             np.array([
             self.sales[self.t],
             self.inventory / 1000
             ])
             if not done else np.array([0, 0])
             )

             return next_state, reward, done
               
        if len(daily_sales) < 10:
             st.warning("Need at least 10 days of sales data for DQN training.")
             st.stop()       

        env = PricingEnv(sales)

    # ---------------- DQN MODEL ----------------
        class DQN(nn.Module):
           def __init__(self):
             super(DQN, self).__init__()
             self.net = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32,3)
            )

           def forward(self, x):
             return self.net(x)

        model = DQN()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

    # ---------------- MEMORY ----------------
        memory = deque(maxlen=2000)

        gamma = 0.95
        epsilon = 1.0
        epsilon_decay = 0.995
        epsilon_min = 0.01

    # ---------------- TRAIN ----------------
        episodes = st.slider("Training Episodes", 5, 20, 10)
        if st.button("Train DQN Model"):

          rewards_history = []

          for ep in range(episodes):
             state = env.reset()
             total_reward = 0

             while True:
                state_tensor = torch.FloatTensor(state)

                # epsilon-greedy
                if random.random() < epsilon:
                   action = random.randint(0,2)
                else:
                   with torch.no_grad():
                    action = torch.argmax(model(state_tensor)).item()

                next_state, reward, done = env.step(action)

                memory.append((state, action, reward, next_state, done))

                state = next_state
                total_reward += reward

                # Train from memory
                if len(memory) > 16:
                  batch = random.sample(memory, 16)

                  for s, a, r, ns, d in batch:
                    s = torch.FloatTensor(s)
                    ns = torch.FloatTensor(ns)

                    target = r
                    if not d:
                        target += gamma * torch.max(model(ns)).item()

                    target_f = model(s).clone().detach()
                    target_f[a] = target

                    loss = loss_fn(model(s), target_f.detach())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if done:
                  break

             rewards_history.append(total_reward)

             if epsilon > epsilon_min:
                epsilon *= epsilon_decay

    # ---------------- RESULTS ----------------
          st.write("### 📈 Training Reward Progress")

          fig = px.line(y=rewards_history, title="DQN Learning Curve")
          st.plotly_chart(fig, use_container_width=True)

          st.success("✅ Deep RL Agent trained successfully!")

    # ---------------- POLICY ----------------
          st.write("### 🧠 Learned Pricing Policy")

          actions_map = ["Decrease Price", "Keep Same", "Increase Price"]

# Ensure valid states
          test_demands = np.linspace( float(min(sales)), float(max(sales)), 10 ) 
          inventory_levels = [1.0, 0.7, 0.4] 
          policy = [] 
          for demand in test_demands: 
           for inv in inventory_levels: 
            state = torch.FloatTensor([ demand, inv ]) 
            with torch.no_grad(): 
                     action = torch.argmax( model(state) ).item() 
            policy.append({ 
                         "Demand": round(demand, 2), 
                         "Inventory": int(inv * 1000), 
                         "Recommended Action": actions_map[action] }) 
          policy_df = pd.DataFrame(policy) 
          st.dataframe(policy_df)

          st.info("💡 This model learns optimal pricing strategy dynamically using Deep Q-Learning.")
else:
    st.info("Upload a dataset to begin 🚀")