import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Energy Dashboard", layout="wide")

# Title
st.title("Energy Demand Dashboard")

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")  # adjust if needed
    df["settlement_date"] = pd.to_datetime(df["settlement_date"], errors='coerce')
    df.dropna(subset=["settlement_date"], inplace=True)

    # Drop columns that are completely empty
    df.dropna(axis=1, how='all', inplace=True)

    df = df.dropna(thresh=len(df) * 0.7, axis=1)

    # Drop irrelevant or constant-value columns manually (if known)
    drop_cols = [
        'non_bm_stor', 'pump_storage_pumping', 'ifa_flow', 'ifa2_flow',
        'britned_flow', 'moyle_flow', 'east_west_flow', 'nemo_flow',
        'nsl_flow', 'eleclink_flow', 'scottish_transfer', 'viking_flow',
        'greenlink_flow', 'embedded_solar_generation', 'embedded_solar_capacity'
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    df.fillna(method="ffill", inplace=True)

    df['year'] = df['settlement_date'].dt.year
    df['month'] = df['settlement_date'].dt.month_name()
    df['weekday'] = df['settlement_date'].dt.day_name()
    df['date'] = df['settlement_date'].dt.date

    df.fillna(method='ffill', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    
    # Remove zero or unrealistic demand
    df = df[df["england_wales_demand"] > 100]

    # Remove extreme outliers using IQR
    Q1 = df["england_wales_demand"].quantile(0.25)
    Q3 = df["england_wales_demand"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df["england_wales_demand"] >= lower_bound) & (df["england_wales_demand"] <= upper_bound)]

    return df

df = load_data()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
     # Year Range Filter
    min_year = int(df["year"].min())
    max_year = int(df["year"].max())
   # selected_year_range = st.slider("Select Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year))
    selected_year = st.selectbox("Select Year", ["All"] + sorted(df["year"].unique()))
    selected_month = st.selectbox("Select Month", ["All"] + sorted(df["month"].dropna().unique()))
    selected_weekday = st.selectbox("Select Weekday", ["All"] + sorted(df["weekday"].dropna().unique()))

    if selected_year != "All":
        df = df[df["year"] == selected_year]

    if selected_month != "All":
        df = df[df["month"] == selected_month]

    if selected_weekday != "All":
        df = df[df["weekday"] == selected_weekday]

# Electricity Demand Over Time
col1, col2 = st.columns(2)

with col1:
    st.subheader("Electricity Demand Over Time")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df["settlement_date"], df["england_wales_demand"], label="Demand", color="tab:blue")
    ax.set_ylabel("Demand (MW)")
    ax.set_xlabel("Date")
    ax.set_title("England & Wales Electricity Demand")
    ax.grid(True)
    st.pyplot(fig)

    # 🆕 Comparative Line Chart
    st.subheader("Comparative Trends: Demand vs Wind Generation/Capacity")
    fig4, ax4 = plt.subplots(figsize=(6, 3))
    ax4.plot(df["settlement_date"], df["england_wales_demand"], label="England & Wales Demand", color="purple")
    ax4.plot(df["settlement_date"], df["embedded_wind_generation"], label="Embedded Wind Generation", color="green")
    ax4.plot(df["settlement_date"], df["embedded_wind_capacity"], label="Embedded Wind Capacity", color="blue")
    ax4.set_ylabel("MW")
    ax4.set_xlabel("Date")
    ax4.set_title("Comparison: Demand vs Wind Generation and Capacity")
    ax4.legend()
    ax4.grid(True)
    st.pyplot(fig4)


with col2:

    # 📊 Average Demand by Weekday
    st.subheader("Average Demand by Weekday")
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_avg = df.groupby("weekday")["england_wales_demand"].mean().reindex(weekday_order)
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.barplot(x=weekday_avg.index, y=weekday_avg.values, palette="viridis", ax=ax2)
    ax2.set_ylabel("Average Demand (MW)")
    ax2.set_title("Electricity Demand by Weekday")
    st.pyplot(fig2)

    # Average Demand by Month
    st.subheader("Average Demand by Month")
    month_order = ["January", "February", "March", "April", "May", "June", 
                "July", "August", "September", "October", "November", "December"]
    month_avg = df.groupby("month")["england_wales_demand"].mean().reindex(month_order)
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    sns.barplot(x=month_avg.index, y=month_avg.values, palette="coolwarm", ax=ax3)
    ax3.set_ylabel("Average Demand (MW)")
    ax3.set_title("Electricity Demand by Month")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

#  Data Preview
#st.subheader(" Data Preview")
#st.dataframe(df.head(20))
