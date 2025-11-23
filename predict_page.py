import streamlit as st
import joblib
import numpy as np
import pandas as pd
from custom_transformers import RareCategoryGrouper


def load_model():
    model = joblib.load('emission_model_pipeline.pkl')
    return model


def show_predict_page():
    st.title("Vehicle CO2 Emissions Prediction")

    st.write("""### Enter the vehicle details to predict CO2 emissions""")

    make_names = (
        'Acura', 'Alfa Romeo', 'Aston Martin', 'Audi', 'Bentley', 'BMW', 'Bugatti', 'Buick', 'Cadillac', \
        'Chevrolet', 'Chrysler', 'Dodge', 'FIAT', 'Ford', 'Genesis', 'GMC', 'Honda', 'Hyundai', 'Infiniti', \
        'Jaguar', 'Jeep', 'Kia', 'Lamborghini', 'Lexus', 'Lincoln', 'Maserati', 'Mazda', 'Mercedes-Benz', 'MINI', \
        'Mitsubishi', 'Nissan', 'Porsche', 'Ram', 'Rolls-Royce', 'Subaru', 'Toyota', 'Volkswagen', 'Volvo'
    )
    vehicle_names = (
        'Compact', 'Two-seater', 'SUV: Small', 'Mid-size', 'Minicompact', 'SUV: Standard', 'Station wagon: Small', \
        'Subcompact', 'Station wagon: Mid-size', 'Full-size', 'Pickup truck: Small', 'Pickup truck: Standard', \
        'Minivan', 'Van: Passenger', 'Special purpose vehicle'
    )
    transmission_names = (
        'AM8', 'AM9', 'AS10', 'A8', 'A9', 'M7', 'AM7', 'AS8', 'M6', 'AS6', 'AV', 'AS9', 'A10', 'A6', 'M5', 'AV7', \
        'AV1', 'AM6', 'AS7', 'AV8', 'AV6', 'AV10', 'AS5'
    )

    # Example input features - adjust based on actual model features
    make = st.selectbox("Make", options=make_names)
    vehicle_class = st.selectbox("Vehicle Class", options=vehicle_names)
    transmission = st.selectbox("Transmission", options=transmission_names)
    engine_size = st.slider("Engine Size (L)", min_value=1.0, max_value=8.0, value=2.0)
    cylinders = st.slider("Cylinders", min_value=1, max_value=16, value=4)
    fuel_comsum_in_city = st.slider("Fuel Consumption in City (L/100 km)", min_value=1.0, max_value=30.0, value=12.0)
    fuel_comsum_in_city_hwy = st.slider("Fuel Consumption in City Hwy (L/100 km)", min_value=1.0, max_value=30.0, value=8.0)
    fuel_comsum_comb = st.slider("Fuel Consumption comb (L/100 km)", min_value=1.0, max_value=30.0, value=10.0)
    smog_level = st.slider("Smog Level", min_value=1, max_value=7, value=5)

    ok = st.button("Predict CO2 Emissions")
    if ok:
        model = load_model()
        input_data = np.array([[make, vehicle_class, engine_size, cylinders, transmission,
                                fuel_comsum_in_city, fuel_comsum_in_city_hwy,
                                fuel_comsum_comb, smog_level]])
        input_df = pd.DataFrame(input_data, columns=[
            'Make', 'Vehicle_Class', 'Engine_Size', 'Cylinders', 'Transmission',
            'Fuel_Consumption_in_City(L/100 km)', 'Fuel_Consumption_in_City_Hwy(L/100 km)',
            'Fuel_Consumption_comb(L/100km)', 'Smog_Level'
        ])
        pred_emission = model.predict(input_df)
        st.subheader("The Predicted CO2 Emissions: {:.2f} g/km".format(pred_emission[0]))


