import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----------- SAMPLE MULTI-FEATURE DATA -----------
np.random.seed(42)
cities = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"]
years = list(range(2015, 2025))
data = []
for city in cities:
    for year in years:
        row = {
            "City": city,
            "Year": year,
            "PM2.5": np.random.randint(40, 200),
            "PM10": np.random.randint(80, 300),
            "NO2": np.random.uniform(15, 80),
            "SO2": np.random.uniform(5, 40),
            "CO": np.random.uniform(0.5, 3.0),
            "O3": np.random.uniform(10, 60),
            "NH3": np.random.uniform(5, 35),
            "Temperature": np.random.uniform(15, 35),
            "Humidity": np.random.uniform(30, 90),
            "AQI": np.random.randint(80, 400),
            "Rainfall": np.random.uniform(100, 3000),
            "WindSpeed": np.random.uniform(0.5, 10),
            "Population": np.random.randint(10_00_000, 2_50_00_000),
            "Vehicles": np.random.randint(1_00_000, 80_00_000),
            "IndustryCount": np.random.randint(100, 5000),
            "GreenCover": np.random.uniform(2, 35),
            "HospitalCount": np.random.randint(10, 1000),
            "SchoolCount": np.random.randint(50, 5000),
            "PowerPlants": np.random.randint(1, 30),
            "WaterQuality": np.random.uniform(40, 100),
            "NoiseLevel": np.random.uniform(50, 100),
            "SolarRadiation": np.random.uniform(2, 8),
            "UVIndex": np.random.uniform(3, 12),
            "ForestArea": np.random.uniform(1, 20),
            "RoadDensity": np.random.uniform(3, 20),
            "GDP": np.random.uniform(1, 100),
            "LiteracyRate": np.random.uniform(70, 99),
            "InternetUsers": np.random.randint(1_00_000, 2_00_00_000),
            "CrimeRate": np.random.uniform(0.5, 10),
            "WasteGenerated": np.random.uniform(100, 9000),
            "RecyclingRate": np.random.uniform(5, 90),
            "TreePlantation": np.random.randint(1000, 100000),
            "PublicTransport": np.random.randint(100, 10000),
            "BikeLanes": np.random.uniform(0.5, 200),
            "Parks": np.random.randint(10, 1000),
            "SportsFacilities": np.random.randint(5, 500),
            "FireStations": np.random.randint(1, 100),
            "PoliceStations": np.random.randint(5, 200),
            "FloodIncidents": np.random.randint(0, 10),
            "DroughtIncidents": np.random.randint(0, 5),
            "HeatwaveDays": np.random.randint(0, 30),
            "ColdwaveDays": np.random.randint(0, 20),
            "AirPurifiersInstalled": np.random.randint(0, 500),
            "SmartSensors": np.random.randint(0, 1000),
            "WeatherStations": np.random.randint(1, 50),
            "AirQualityStations": np.random.randint(1, 50),
        }
        data.append(row)
df_sample = pd.DataFrame(data)

# ----------- SIDEBAR NAVIGATION -----------
st.set_page_config(page_title="🌍 City Environment Mega Dashboard", layout="wide")
st.sidebar.title("🌍 City Environment Mega Dashboard")
menu = st.sidebar.radio("Menu", [
    "Home", "Upload Data", "Data Overview", "Visualization", "Correlation", "Prediction", "Download", "About"
])

# ----------- DATA HANDLING -----------
def get_data():
    uploaded = st.session_state.get("uploaded_file")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        df = df_sample.copy()
    return df

# ----------- HOME PAGE -----------
if menu == "Home":
    st.title("🌍 City Environment Mega Dashboard")
    st.markdown("""
    - Analyze, visualize, and predict city environmental and urban metrics
    - 40+ features: pollution, weather, demography, urban, health, infra, climate, etc.
    - Upload your own CSV or use rich sample data
    - Advanced analytics: Correlation, regression, multi-feature support
    - Download cleaned data and predictions
    """)
    st.success("Navigate using the sidebar. For best experience, use wide screen.")

# ----------- DATA UPLOAD -----------
elif menu == "Upload Data":
    st.header("Upload Your Data")
    uploaded_file = st.file_uploader("Upload your City Environmental Data CSV", type=["csv"])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.success("File uploaded! Now go to 'Data Overview'.")

# ----------- DATA OVERVIEW -----------
elif menu == "Data Overview":
    st.header("Data Overview & Cleaning")
    df = get_data()
    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader("Shape")
    st.write(df.shape)
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    st.subheader("Duplicates")
    st.write(f"Duplicate rows: {df.duplicated().sum()}")
    if st.button("Remove Duplicates"):
        df = df.drop_duplicates()
        st.success("Duplicates removed.")
    st.subheader("Summary Statistics")
    st.write(df.describe(include='all'))

# ----------- VISUALIZATION -----------
elif menu == "Visualization":
    st.header("Interactive Visualization")
    df = get_data()
    city = st.selectbox("Select City", sorted(df["City"].unique()))
    year_range = st.slider("Select Year Range", int(df["Year"].min()), int(df["Year"].max()), (int(df["Year"].min()), int(df["Year"].max())))
    dfc = df[(df["City"] == city) & (df["Year"].between(year_range[0], year_range[1]))]
    feature = st.selectbox("Select Feature", [col for col in dfc.columns if col not in ["City", "Year"]])
    st.plotly_chart(px.line(dfc, x="Year", y=feature, markers=True, title=f"{city} - {feature} Over Years"), use_container_width=True)
    st.subheader("Compare Multiple Features")
    features = st.multiselect(
        "Select Features (max 5)",
        [col for col in dfc.columns if col not in ["City", "Year"]],
        default=[col for col in ["PM2.5", "AQI"] if col in dfc.columns][:2],
        max_selections=5
    )
    if features:
        st.plotly_chart(px.line(dfc, x="Year", y=features, markers=True, title=f"{city} - Multiple Features Over Years"), use_container_width=True)
    st.subheader("Distribution (All Cities)")
    feature2 = st.selectbox("Select Feature for Distribution", [col for col in df.columns if col not in ["City", "Year"]], key="dist")
    fig, ax = plt.subplots()
    sns.histplot(df[feature2], kde=True, ax=ax)
    st.pyplot(fig)

# ----------- CORRELATION -----------
elif menu == "Correlation":
    st.header("Feature Correlation Matrix")
    df = get_data()
    city = st.selectbox("Select City", sorted(df["City"].unique()), key="corr_city")
    dfc = df[df["City"] == city]
    corr = dfc.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    st.info("Dark red/blue = strong correlation. Use this to find related features.")

# ----------- PREDICTION -----------
elif menu == "Prediction":
    st.header("Predict Any Feature (Regression)")
    df = get_data()
    city = st.selectbox("Select City", sorted(df["City"].unique()), key="pred_city")
    target = st.selectbox("Select Target Feature", [col for col in df.columns if col not in ["City", "Year"]], key="target")
    available_features = [col for col in df.columns if col not in ["City", "Year", target]]
    # Dynamically choose defaults only if present in available_features
    default_features = [col for col in ["PM2.5", "Temperature"] if col in available_features][:2]
    features = st.multiselect(
        "Select Input Features (max 5)",
        available_features,
        default=default_features,
        max_selections=5
    )
    year = st.number_input("Year for Prediction", min_value=int(df["Year"].min()), max_value=2100, value=int(df["Year"].max())+1)
    dfc = df[df["City"] == city]
    if len(features) > 0 and len(dfc) > 1:
        X = dfc[features]
        y = dfc[target]
        model = LinearRegression()
        model.fit(X, y)
        st.write(f"Model R2 Score: {model.score(X, y):.2f}")
        st.write(f"Model Mean Squared Error: {mean_squared_error(y, model.predict(X)):.2f}")
        st.subheader("Enter values for prediction")
        user_input = []
        for feat in features:
            user_input.append(st.number_input(f"{feat}", float(df[feat].min()), float(df[feat].max()), float(df[feat].mean())))
        if st.button("Predict"):
            pred = model.predict([user_input])[0]
            st.success(f"Predicted {target} for {city} in {year}: {pred:.2f}")
    else:
        st.warning("Please select at least one feature and ensure sufficient data.")

# ----------- DOWNLOAD DATA -----------
elif menu == "Download":
    st.header("Download Data")
    df = get_data()
    st.download_button("Download Current Data CSV", df.to_csv(index=False), file_name="city_environment_data.csv")

# ----------- ABOUT -----------
elif menu == "About":
    st.header("About")
    st.markdown("""
    **City Environment Mega Dashboard**
    - 40+ features, multi-city, multi-year, multi-variable analytics
    - Upload your own data or use rich sample data
    - Advanced visualization, correlation, regression prediction
    - Developed for smart city, urban, environment, and data science projects
    """)
    st.write("Developed by Perplexity AI | 2025")
