import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="✈️ Flight Delay Predictor",
    page_icon="✈️",
    layout="wide"
)

# ── Load assets ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    scaler    = joblib.load('scaler.pkl')
    lr_model  = joblib.load('logistic_model.pkl')
    dt_model  = joblib.load('decision_tree_model.pkl')
    features  = joblib.load('feature_columns.pkl')
    sample_df = pd.read_csv('sample_predictions.csv')
    return scaler, lr_model, dt_model, features, sample_df

scaler, lr_model, dt_model, features, sample_df = load_assets()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("✈️ Interactive Flight Delay Predictor")
st.markdown(
    "Adjust the sliders below and click **Predict** to see real-time delay probability. "
    "Charts update with every prediction."
)
st.divider()

# ── Sidebar: Model selector ────────────────────────────────────────────────
model_choice = st.sidebar.selectbox(
    "🧠 Choose Prediction Model",
    ["Logistic Regression", "Decision Tree"]
)
active_model = lr_model if model_choice == "Logistic Regression" else dt_model

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "Models trained on the **2019 Airline Delays** dataset. "
    "A delay is defined as ≥15 minutes late at departure."
)

# ── Sliders ───────────────────────────────────────────────────────────────────
st.subheader("🎛️ Flight Parameters")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**🕐 Time & Route**")
    month         = st.slider("Month",               1,  12,  6)
    day_of_week   = st.slider("Day of Week",          1,   7,  3)
    dep_time_blk  = st.slider("Departure Time Block", 0,  18,  8)
    distance_group= st.slider("Distance Group",       1,  11,  3)
    segment_number= st.slider("Segment Number",       1,   6,  1)

with c2:
    st.markdown("**✈️ Aircraft & Airline**")
    concurrent_flights = st.slider("Concurrent Flights",  1, 109, 25)
    number_of_seats    = st.slider("Number of Seats",    44, 337, 150)
    plane_age          = st.slider("Plane Age (years)",   0,  32,  10)
    carrier_name       = st.slider("Carrier Code",        0,  16,   5)

with c3:
    st.markdown("**🌦️ Weather**")
    prcp = st.slider("Precipitation (in)", 0.0, 12.0, 0.0, 0.1)
    snow = st.slider("Snow (in)",          0.0, 18.0, 0.0, 0.1)
    tmax = st.slider("Max Temp (°F)",    -10.0,115.0, 70.0)
    awnd = st.slider("Avg Wind Speed",    0.0, 35.0,  8.0)

# ── Build input row ───────────────────────────────────────────────────────────
# Must match the exact column order from feature_columns.pkl
input_dict = {
    'MONTH': month, 'DAY_OF_WEEK': day_of_week, 'DEP_TIME_BLK': dep_time_blk,
    'DISTANCE_GROUP': distance_group, 'SEGMENT_NUMBER': segment_number,
    'CONCURRENT_FLIGHTS': concurrent_flights, 'NUMBER_OF_SEATS': number_of_seats,
    'CARRIER_NAME': carrier_name, 'AIRPORT_FLIGHTS_MONTH': 12000,
    'AIRLINE_FLIGHTS_MONTH': 5000, 'AIRLINE_AIRPORT_FLIGHTS_MONTH': 1000,
    'AVG_MONTHLY_PASS_AIRPORT': 50000, 'AVG_MONTHLY_PASS_AIRLINE': 20000,
    'FLT_ATTENDANTS_PER_PASS': 0.0005, 'GROUND_SERV_PER_PASS': 0.0005,
    'PLANE_AGE': plane_age, 'DEPARTING_AIRPORT': 40, 'LATITUDE': 36.08,
    'LONGITUDE': -90.0, 'PREVIOUS_AIRPORT': 200, 'PRCP': prcp,
    'SNOW': snow, 'SNWD': 0.0, 'TMAX': tmax, 'AWND': awnd
}

input_df = pd.DataFrame([input_dict])[features]  # enforce column order

# ── Predict ───────────────────────────────────────────────────────────────────
st.divider()
predict_btn = st.button("🔍 Predict Delay Probability", use_container_width=True, type="primary")

if predict_btn:
    scaled = scaler.transform(input_df)
    prob   = active_model.predict_proba(scaled)[0][1]

    st.subheader("📊 Prediction Result")
    col_gauge, col_metric = st.columns([2, 1])

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            title={"text": "Delay Probability (%)"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "crimson" if prob > 0.5 else "steelblue"},
                "steps": [
                    {"range": [0, 33],  "color": "#d4edda"},
                    {"range": [33, 66], "color": "#fff3cd"},
                    {"range": [66, 100],"color": "#f8d7da"},
                ],
                "threshold": {"line": {"color": "black", "width": 4}, "value": 50}
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=40, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_metric:
        st.metric("Model Used", model_choice)
        st.metric("Delay Probability", f"{prob:.1%}")
        if prob > 0.5:
            st.error("🚨 **HIGH RISK** of delay (≥15 min)")
        else:
            st.success("✅ **LOW RISK** — likely on time")

    # ── Feature importance bars (Decision Tree only) ──────────────────────
    if model_choice == "Decision Tree":
        st.subheader("🔎 Feature Importances")
        fi = pd.DataFrame({
            "Feature": features,
            "Importance": dt_model.feature_importances_
        }).sort_values("Importance", ascending=False).head(10)
        fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale="Blues",
                        title="Top 10 Features (Decision Tree)")
        fig_fi.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_fi, use_container_width=True)

# ── Static charts from sample data ────────────────────────────────────────────
st.divider()
st.subheader("📈 Dataset Insights")

tab1, tab2, tab3, tab4 = st.tabs([
    "Delay by Month", "Delay by Day", "Delay vs Weather", "Model Comparison"
])

with tab1:
    monthly = sample_df.groupby('MONTH')['DEP_DEL15'].mean().reset_index()
    monthly.columns = ['Month', 'Delay Rate']
    fig1 = px.bar(monthly, x='Month', y='Delay Rate',
                  title="Average Delay Rate by Month",
                  color='Delay Rate', color_continuous_scale='RdYlGn_r',
                  labels={'Delay Rate': 'Delay Rate (fraction)'})
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    days = {1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat',7:'Sun'}
    daily = sample_df.groupby('DAY_OF_WEEK')['DEP_DEL15'].mean().reset_index()
    daily['Day'] = daily['DAY_OF_WEEK'].map(days)
    fig2 = px.line(daily, x='Day', y='DEP_DEL15', markers=True,
                   title="Average Delay Rate by Day of Week",
                   labels={'DEP_DEL15': 'Delay Rate'})
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = px.scatter(sample_df.sample(300), x='TMAX', y='LR_PROB',
                      color='DEP_DEL15', opacity=0.6,
                      title="Max Temperature vs Predicted Delay Probability",
                      labels={'TMAX':'Max Temp (°F)', 'LR_PROB':'Predicted Probability',
                              'DEP_DEL15':'Actually Delayed'})
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    fig4 = px.scatter(sample_df.sample(300), x='LR_PROB', y='DT_PROB',
                      color='DEP_DEL15', opacity=0.6,
                      title="Logistic Regression vs Decision Tree Predictions",
                      labels={'LR_PROB':'Logistic Regression Prob',
                              'DT_PROB':'Decision Tree Prob',
                              'DEP_DEL15':'Actually Delayed'})
    fig4.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                   line=dict(color="grey", dash="dash"))
    st.plotly_chart(fig4, use_container_width=True)