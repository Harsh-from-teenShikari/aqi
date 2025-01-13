import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page config
st.set_page_config(page_title="AQI Prediction App", layout="wide")

# Load the model
@st.cache_resource
def load_model():
    with open('best_aqi_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_data
def load_data():
    # Load your dataset here
    df = pd.read_csv('best_aqi_model.pkl')  # Assuming you have a CSV with the same name
    return df

def main():
    st.title("Air Quality Index (AQI) Prediction")
    
    try:
        # Load model and data
        model = load_model()
        df = load_data()
        
        # Create two columns for inputs
        col1, col2 = st.columns(2)
        
        with col1:
            # Country dropdown
            countries = sorted(df['Country'].unique())
            country = st.selectbox('Select Country', countries)
            
            # Filter cities based on selected country
            cities = sorted(df[df['Country'] == country]['City'].unique())
            city = st.selectbox('Select City', cities)
            
            # Latitude and Longitude
            lat = st.number_input('Latitude', value=df[(df['Country'] == country) & (df['City'] == city)]['lat'].iloc[0] if len(cities) > 0 else 0.0)
            lng = st.number_input('Longitude', value=df[(df['Country'] == country) & (df['City'] == city)]['lng'].iloc[0] if len(cities) > 0 else 0.0)
        
        with col2:
            # AQI inputs
            co_aqi = st.number_input('CO AQI Value', min_value=0.0)
            ozone_aqi = st.number_input('Ozone AQI Value', min_value=0.0)
            no2_aqi = st.number_input('NO2 AQI Value', min_value=0.0)
            pm25_aqi = st.number_input('PM2.5 AQI Value', min_value=0.0)
        
        # Predict button
        if st.button('Predict AQI'):
            # Prepare input data
            input_data = np.array([[
                co_aqi, ozone_aqi, no2_aqi, pm25_aqi, lat, lng
            ]])
            
            # Make prediction - assuming model returns both AQI value and category
            aqi_value, aqi_category = model.predict(input_data)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            # Create two columns for displaying results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted AQI", f"{aqi_value:.2f}")
            
            with col2:
                st.metric("AQI Category", aqi_category)
            
            # Display location details
            st.markdown("---")
            st.markdown(f"""
            ### Location Details:
            - Country: {country}
            - City: {city}
            - Coordinates: ({lat:.4f}, {lng:.4f})
            """)
            
            # Input values summary
            st.markdown("### Input Parameters:")
            st.markdown(f"""
            - CO AQI: {co_aqi}
            - Ozone AQI: {ozone_aqi}
            - NO2 AQI: {no2_aqi}
            - PM2.5 AQI: {pm25_aqi}
            """)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure the model file and dataset are in the correct location.")

if __name__ == "__main__":
    main()
