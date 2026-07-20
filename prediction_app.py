import streamlit as st
import pandas as pd
import joblib


# ==============================
# Load trained best model
# ==============================

model = joblib.load(
    "house_price_model.pkl"
)



# ==============================
# Website Title
# ==============================

st.title("🏠 House Price Prediction System")

st.write(
    "Enter house information to predict price in ETB"
)



# ==============================
# INPUT FEATURES
# Change these according to your columns
# ==============================


bedrooms = st.number_input(
    "Bedrooms",
    min_value=0
)


bathrooms = st.number_input(
    "Bathrooms",
    min_value=0
)


area = st.number_input(
    "Area",
    min_value=0
)


location = st.text_input(
    "Location"
)


house_type = st.selectbox(

    "House Type",

    [
        "Apartment",
        "Villa",
        "House"
    ]

)


year = st.number_input(
    "Year Built",
    min_value=1900,
    max_value=2026
)


parking = st.number_input(
    "Parking Spaces",
    min_value=0
)



# ==============================
# Prediction
# ==============================


if st.button("Predict Price"):


    input_data = pd.DataFrame(

        {

        "bedrooms":[bedrooms],

        "bathrooms":[bathrooms],

        "area":[area],

        "location":[location],

        "house_type":[house_type],

        "year":[year],

        "parking":[parking]

        }

    )


    prediction = model.predict(
        input_data
    )


    st.success(

        f"Estimated Price: {prediction[0]:,.2f} ETB"

    )