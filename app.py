import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page config
st.set_page_config(page_title="AQI Prediction App", layout="wide")

# Load the model
# @st.cache_resource
def load_model():
    with open("Decision_Tree.pkl", 'rb') as file:
        model = pickle.load(file)
    return model

# Load the dataset
# @st.cache_data
def load_data():
    df = pd.read_csv("AQI and Lat Long of Countries cleaned dataset.csv")  # Corrected file name
    return df

# Main function
def main():
    st.title("Air Quality Index (AQI) Prediction")
    
    try:
        # Load model and data
        model = load_model()
        df = load_data()

        # Country dropdown
        countries = sorted(df['Country'].unique())
        country = st.selectbox('Select Country', countries)

        # Filter cities based on selected country
        cities = sorted(df[df['Country'] == country]['City'].unique())
        city = st.selectbox('Select City', cities)

        # Latitude and Longitude inputs
        lat = st.number_input('Latitude', value=df[(df['Country'] == country) & (df['City'] == city)]['lat'].iloc[0])
        lng = st.number_input('Longitude', value=df[(df['Country'] == country) & (df['City'] == city)]['lng'].iloc[0])

        # AQI inputs
        co_aqi = st.number_input('CO AQI Value', min_value=0.0)
        ozone_aqi = st.number_input('Ozone AQI Value', min_value=0.0)
        no2_aqi = st.number_input('NO2 AQI Value', min_value=0.0)
        pm25_aqi = st.number_input('PM2.5 AQI Value', min_value=0.0)

        # Predict button
        if st.button('Predict AQI'):
            # Prepare input data
            input_data = np.array([[co_aqi, ozone_aqi, no2_aqi, pm25_aqi, lat, lng]])

            # Make prediction
            aqi_value, aqi_category = model.predict(input_data)[0]

            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted AQI", f"{aqi_value:.2f}")
            with col2:
                st.metric("AQI Category", aqi_category)

            # Display location details
            st.markdown("---")
            st.markdown(f"### Location Details:\n- Country: {country}\n- City: {city}\n- Coordinates: ({lat:.4f}, {lng:.4f})")

            # Display input values
            st.markdown("### Input Parameters:")
            st.markdown(f"- CO AQI: {co_aqi}\n- Ozone AQI: {ozone_aqi}\n- NO2 AQI: {no2_aqi}\n- PM2.5 AQI: {pm25_aqi}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please ensure the model file and dataset are correctly uploaded.")

if __name__ == "__main__":
    main()
