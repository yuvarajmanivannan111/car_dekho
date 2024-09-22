import streamlit as st
import pandas as pd
import joblib

# --- Load the model and encoders (for later use) ---
loaded_model = joblib.load(r'C:\Users\yuvanelson\Downloads\Car-Dheko---Used-Car-Price-Prediction-main\Car-Dheko---Used-Car-Price-Prediction-main\Car_Dheko_Project\Notebooks\rf_trainedmodel.pkl')
loaded_encoders = joblib.load(r'C:\Users\yuvanelson\Downloads\Car-Dheko---Used-Car-Price-Prediction-main\Car-Dheko---Used-Car-Price-Prediction-main\Car_Dheko_Project\Notebooks\label_encoders.pkl')
loaded_scaler = joblib.load(r'C:\Users\yuvanelson\Downloads\Car-Dheko---Used-Car-Price-Prediction-main\Car-Dheko---Used-Car-Price-Prediction-main\Car_Dheko_Project\Notebooks\min.pkl')
ridge_model = joblib.load(r'C:\Users\yuvanelson\Downloads\Car-Dheko---Used-Car-Price-Prediction-main\Car-Dheko---Used-Car-Price-Prediction-main\Car_Dheko_Project\Notebooks\ridge_model.pkl')

categorical_columns = [
    "Body_type",
    "Transmission",
    "Original_equipment_manufacturer",
    "Model",
    "Variant_name",
    "Insurance_validity",
    "Fuel_type",
    "Colour",
    "Location",
]

# --- Function to take custom inputs and predict price ---
@st.cache_resource
def predict_price(
    mileage,
    engine_displacement,
    year_of_manufacture,
    transmission,
    fuel_type,
    owner_no,
    model_year,
    location,
    kilometer_driven,
    body_type,
):

    # Create a DataFrame from the input values
    input_data = pd.DataFrame(
        {
            "Mileage": [mileage],
            "Engine_displacement": [engine_displacement],
            "Year_of_manufacture": [year_of_manufacture],
            "Transmission": [transmission],
            "Fuel_type": [fuel_type],
            "Owner_No.": [owner_no],
            "Model_year": [model_year],
            "Location": [location],
            "Kilometer_Driven": [kilometer_driven],
            "Body_type": [body_type],
        }
    )

    # Encode categorical features using the loaded LabelEncoders
    for col in categorical_columns:
        if col not in {
            "Original_equipment_manufacturer",
            "Model",
            "Variant_name",
            "Insurance_validity",
            "Colour",
        }:
            le = loaded_encoders[col]
            input_data[col] = le.transform(input_data[col].astype(str))

    # Make the prediction
    predicted_price = loaded_model.predict(input_data)
    predicted_price_norm = loaded_scaler.inverse_transform([[predicted_price[0]]])[0][0]
    return predicted_price_norm


# Streamlit UI
st.set_page_config(page_title="Used Car Price Prediction", page_icon=":red_car:", layout="wide")
st.title(":red_car: Used Car Price Prediction")
st.markdown(
    "<style>div.block-container{padding-top:2rem;}</style", unsafe_allow_html=True
)

# URL of the image
image_url = 'https://dcassetcdn.com/design_img/6348/18475/18475_151725_6348_image.jpg'

# Display the image
st.image(image_url, use_column_width=True)

st.sidebar.header("Input Features")

# Input fields with dropdowns for Location and Body Type
mileage = st.sidebar.number_input("Mileage", min_value=0.0, format="%.1f")
engine_displacement = st.sidebar.number_input("Engine Displacement", min_value=0)
year_of_manufacture = st.sidebar.number_input(
    "Year of Manufacture", min_value=1900, max_value=2024
)
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel"])
owner_no = st.sidebar.number_input("Owner No.", min_value=0)
model_year = st.sidebar.number_input("Model Year", min_value=1900, max_value=2024)

# Dropdown for Location
location = st.sidebar.selectbox(
    "Location",
    ["Chennai", "Bangalore", "Delhi", "Kolkata", "Jaipur", "Hyderabad"]
)

# Dropdown for Body Type
body_type = st.sidebar.selectbox(
    "Body Type",
    ["Hatchback", "SUV", "Sedan", "MUV", "Minivans", "Coupe", "Pickup Trucks", "Convertibles", "Hybrids", "Wagon"]
)

kilometer_driven = st.sidebar.number_input("Kilometer Driven", min_value=0)

if st.sidebar.button("Estimate Used Car Price"):
    try:
        predicted_price = predict_price(
            mileage,
            engine_displacement,
            year_of_manufacture,
            transmission,
            fuel_type,
            owner_no,
            model_year,
            location,
            kilometer_driven,
            body_type,
        )
        st.write(f"Predicted Price: ₹{predicted_price:,.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
