import streamlit as st
import pandas as pd
import joblib
import os

# ==========================
# PAGE CONFIGURATION
# ==========================
st.set_page_config(
    page_title="House Price Prediction System",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction System")
st.write("Enter the property information below to predict the house price (ETB).")

# ==========================
# LOAD MODEL
# ==========================

MODEL_PATH = "house_price_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("house_price_model.pkl was not found.")
    st.info("Train your model first to create house_price_model.pkl.")
    st.stop()

model = joblib.load(MODEL_PATH)

# ==========================
# INPUT SECTION
# ==========================

st.header("Property Information")

col1, col2 = st.columns(2)

with col1:
    Number_of_Rooms = st.number_input(
        "Number of Rooms",
        min_value=1,
        value=3
    )

    Site_Area_sqm = st.number_input(
        "Site Area (sqm)",
        min_value=1.0,
        value=200.0
    )

    Built_Area_sqm = st.number_input(
        "Built Area (sqm)",
        min_value=1.0,
        value=150.0
    )

    Property_Years = st.number_input(
        "Property Years",
        min_value=0,
        value=5
    )

    Construction_Materials = st.selectbox(
        "Construction Materials",
        [
            "Concrete",
            "Mud&Wood"
        ]
    )

    Housing_Typology = st.selectbox(
        "Housing Typology",
        [
            "Detached",
            "Semi-detached",
            "Condominium"
        ]
    )

with col2:

    Land_Value_Grading = st.selectbox(
        "Land Value Grading",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    Proximity_to_CBD_km = st.number_input(
        "Distance to CBD (km)",
        min_value=0.0,
        value=5.0
    )

    Proximity_to_Bus_Station_km = st.number_input(
        "Distance to Bus Station (km)",
        min_value=0.0,
        value=1.0
    )

    Type_of_Nearest_Road = st.selectbox(
        "Nearest Road Type",
        [
            "Asphalt",
            "Gravel"
        ]
    )

    Proximity_to_Schools_km = st.number_input(
        "Distance to Schools (km)",
        min_value=0.0,
        value=1.0
    )

# ==========================
# PREDICTION
# ==========================

if st.button("Predict House Price"):

    input_df = pd.DataFrame({

        "Number_of_Rooms":[Number_of_Rooms],

        "Site_Area_sqm":[Site_Area_sqm],

        "Built_Area_sqm":[Built_Area_sqm],

        "Property_Years":[Property_Years],

        "Construction_Materials":[Construction_Materials],

        "Housing_Typology":[Housing_Typology],

        "Land_Value_Grading":[Land_Value_Grading],

        "Proximity_to_CBD_km":[Proximity_to_CBD_km],

        "Proximity_to_Bus_Station_km":[Proximity_to_Bus_Station_km],

        "Type_of_Nearest_Road":[Type_of_Nearest_Road],

        "Proximity_to_Schools_km":[Proximity_to_Schools_km]

    })

    try:

        prediction = model.predict(input_df)

        st.success(
            f"Predicted House Price: {prediction[0]:,.2f} ETB"
        )

    except Exception as e:

        st.error("Prediction failed.")

        st.exception(e)

# ==========================
# FOOTER
# ==========================

st.markdown("---")
st.caption("Machine Learning House Price Prediction System")