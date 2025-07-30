import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64

#load the trained model
model = joblib.load('model.pkl')

#list of features
FEATURE_COLS = [
    'floor_area_sqm', 'lease_commence_date',
    # flat_model dummies
    'flat_model_2-room', 'flat_model_3Gen', 'flat_model_Adjoined flat',
    'flat_model_Apartment', 'flat_model_DBSS', 'flat_model_Improved',
    'flat_model_Improved-Maisonette', 'flat_model_Maisonette',
    'flat_model_Model A', 'flat_model_Model A-Maisonette', 'flat_model_Model A2',
    'flat_model_Multi Generation', 'flat_model_New Generation',
    'flat_model_Premium Apartment', 'flat_model_Premium Apartment Loft',
    'flat_model_Premium Maisonette', 'flat_model_Simplified',
    'flat_model_Standard', 'flat_model_Terrace', 'flat_model_Type S1',
    'flat_model_Type S2',
    # flat_type dummies
    'flat_type_1 ROOM', 'flat_type_2 ROOM', 'flat_type_3 ROOM',
    'flat_type_4 ROOM', 'flat_type_5 ROOM', 'flat_type_EXECUTIVE',
    'flat_type_MULTI-GENERATION',
    # town dummies
    'town_ANG MO KIO', 'town_BEDOK', 'town_BISHAN', 'town_BUKIT BATOK', 'town_BUKIT MERAH',
    'town_BUKIT PANJANG', 'town_BUKIT TIMAH', 'town_CENTRAL AREA', 'town_CHOA CHU KANG',
    'town_CLEMENTI', 'town_GEYLANG', 'town_HOUGANG', 'town_JURONG EAST', 'town_JURONG WEST',
    'town_KALLANG/WHAMPOA', 'town_MARINE PARADE', 'town_PASIR RIS', 'town_PUNGGOL',
    'town_QUEENSTOWN', 'town_SEMBAWANG', 'town_SENGKANG', 'town_SERANGOON', 'town_TAMPINES',
    'town_TOA PAYOH', 'town_WOODLANDS', 'town_YISHUN',
    #other numeric/features
    'storey_avg', 'year', 'month_num', 'flat_age', 'area_times_age',
    #region dummies
    'region_Central', 'region_East', 'region_North', 'region_North-East', 'region_West'
]

# Town to region mapping for region one-hot
TOWN_TO_REGION = {
    'ANG MO KIO': 'region_North',
    'BEDOK': 'region_East',
    'BISHAN': 'region_Central',
    'BUKIT BATOK': 'region_West',
    'BUKIT MERAH': 'region_Central',
    'BUKIT PANJANG': 'region_North',
    'BUKIT TIMAH': 'region_Central',
    'CENTRAL AREA': 'region_Central',
    'CHOA CHU KANG': 'region_West',
    'CLEMENTI': 'region_West',
    'GEYLANG': 'region_East',
    'HOUGANG': 'region_North-East',
    'JURONG EAST': 'region_West',
    'JURONG WEST': 'region_West',
    'KALLANG/WHAMPOA': 'region_Central',
    'MARINE PARADE': 'region_East',
    'PASIR RIS': 'region_East',
    'PUNGGOL': 'region_North-East',
    'QUEENSTOWN': 'region_Central',
    'SEMBAWANG': 'region_North',
    'SENGKANG': 'region_North-East',
    'SERANGOON': 'region_North',
    'TAMPINES': 'region_East',
    'TOA PAYOH': 'region_Central',
    'WOODLANDS': 'region_North',
    'YISHUN': 'region_North',
}

# Flat models list from feature names (extracted)
FLAT_MODELS = [
    '2-room', '3Gen', 'Adjoined flat', 'Apartment', 'DBSS', 'Improved',
    'Improved-Maisonette', 'Maisonette', 'Model A', 'Model A-Maisonette',
    'Model A2', 'Multi Generation', 'New Generation', 'Premium Apartment',
    'Premium Apartment Loft', 'Premium Maisonette', 'Simplified', 'Standard',
    'Terrace', 'Type S1', 'Type S2'
]

# Flat types list from feature names
FLAT_TYPES = [
    '1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'
]

# Town list (keys of TOWN_TO_REGION)
TOWN_LIST = list(TOWN_TO_REGION.keys())

st.set_page_config(page_title='HDB Resale Price Predictor', layout='centered')

def bgImg(image_url, blur_px=6):

    st.markdown(
        f"""
        <style>
        body {{
            background-image: url("{image_url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            background-attachment: fixed;
        }}

        .stApp {{
            background-color: rgba(0, 20, 0, 0.45);
            backdrop-filter: blur({blur_px}px);
            padding: 2rem;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

#to call before ui appears
bgImg("https://www.hdb.gov.sg/-/media/HDBContent/Images/SCEG/dw2101_a3_13_PatLaw.jpg")

st.title("HDB Resale Price Predictor")
st.markdown("Estimate your flat's resale price using a machine learning model trained on historical data.")

with st.form("prediction_form"):
    st.header("Enter Flat Details")
    
    lease_commence_date = st.number_input("Lease Commence Year (e.g., 1990)", min_value=1950, max_value=2025, value=1990)
    storey_avg = st.number_input("Average Storey", min_value=1, max_value=50, value=5)
    year = st.number_input("Year of Transaction", min_value=2000, max_value=2030, value=2023)
    month_num = st.number_input("Month of Transaction (1-12)", min_value=1, max_value=12, value=1)
    flat_age = st.number_input("Flat Age (years)", min_value=0, max_value=100, value=30)
    area_times_age = st.number_input("Floor Area x Flat Age", min_value=0.0, max_value=50000.0, value=2700.0)
    floor_area_sqm = st.slider("Floor Area (sqm)", min_value=10.0, max_value=300.0, value=90.0, step=10.0)

    
    st.subheader("Select Flat Model (choose one)")
    selected_flat_model = st.selectbox("Flat Model", FLAT_MODELS)
    
    st.subheader("Select Flat Type (choose one)")
    selected_flat_type = st.selectbox("Flat Type", FLAT_TYPES)
    
    selected_town = st.selectbox("Town", TOWN_LIST)
    
    submitted = st.form_submit_button("Predict Resale Price")

def preprocess_input(
    floor_area_sqm, lease_commence_date, storey_avg, year, month_num, flat_age, area_times_age,
    selected_flat_model, selected_flat_type, selected_town
):
    # Initialize all features to 0
    input_dict = {feat: 0 for feat in FEATURE_COLS}
    
    # Numeric features
    input_dict['floor_area_sqm'] = floor_area_sqm
    input_dict['lease_commence_date'] = lease_commence_date
    input_dict['storey_avg'] = storey_avg
    input_dict['year'] = year
    input_dict['month_num'] = month_num
    input_dict['flat_age'] = flat_age
    input_dict['area_times_age'] = area_times_age
    
    # Set flat model one-hot
    flat_model_col = f'flat_model_{selected_flat_model}'
    if flat_model_col in input_dict:
        input_dict[flat_model_col] = 1
    else:
        st.warning(f"Warning: Unknown flat model selected: {selected_flat_model}")
    
    # Set flat type one-hot
    flat_type_col = f'flat_type_{selected_flat_type}'
    if flat_type_col in input_dict:
        input_dict[flat_type_col] = 1
    else:
        st.warning(f"Warning: Unknown flat type selected: {selected_flat_type}")
       
    # Set town one-hot
    town_col = f'town_{selected_town}'
    if town_col in input_dict:
        input_dict[town_col] = 1
    else:
        st.warning(f"Warning: Unknown town selected: {selected_town}")
    
    # Infer region from town
    region_col = TOWN_TO_REGION.get(selected_town)
    if region_col and region_col in input_dict:
        input_dict[region_col] = 1
    else:
        st.warning(f"Warning: No valid region mapping for town {selected_town}")
    
    # Convert to dataframe
    df_input = pd.DataFrame([input_dict])
    return df_input

if submitted:
    input_df = preprocess_input(
        floor_area_sqm, lease_commence_date, storey_avg, year, month_num,
        flat_age, area_times_age, selected_flat_model, selected_flat_type, selected_town
    )
    try:
        pred = model.predict(input_df)[0]
        st.success(f"**Estimated Resale Price:** SGD ${pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.markdown("*(This is a data-driven estimate and may not reflect final market value.)*")

st.write(
    "This app was built as part of a machine learning project for predicting HDB resale prices. "
    "All data and predictions are for educational purposes only."
)
