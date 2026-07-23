
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Define the feature list as used during training
features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
            'total_bedrooms', 'population', 'households', 'median_income']

# Load the saved components
try:
    scaler = joblib.load('scaler.pkl')
    poly_features = joblib.load('poly_features_3.pkl')
    model = joblib.load('poly_reg_3_scaled_model.pkl')
except FileNotFoundError:
    st.error("Error: Model components not found. Make sure 'scaler.pkl', 'poly_features_3.pkl', and 'poly_reg_3_scaled_model.pkl' are in the same directory.")
    st.stop()

st.set_page_config(page_title="California Housing Price Predictor", layout="centered")

st.title("🏡 California Housing Price Predictor")
st.markdown("Enter the housing characteristics to predict its median house value.")

st.subheader("Property Characteristics")

# Input widgets for features
with st.sidebar:
    st.header("Input Features")
    longitude = st.slider('Longitude', -124.3, -114.0, -122.25, 0.01)
    latitude = st.slider('Latitude', 32.5, 42.0, 37.85, 0.01)
    housing_median_age = st.slider('Housing Median Age', 1, 52, 30, 1)
    total_rooms = st.number_input('Total Rooms', 1, 39320, 2000)
    total_bedrooms = st.number_input('Total Bedrooms', 1, 6445, 400)
    population = st.number_input('Population', 3, 35682, 1000)
    households = st.number_input('Households', 1, 6082, 350)
    median_income = st.slider('Median Income', 0.5, 15.0, 4.5, 0.1)

# Create a dictionary from the input values
raw_data = {
    'longitude': longitude,
    'latitude': latitude,
    'housing_median_age': housing_median_age,
    'total_rooms': total_rooms,
    'total_bedrooms': total_bedrooms,
    'population': population,
    'households': households,
    'median_income': median_income
}

# Convert raw input data to a DataFrame
input_df = pd.DataFrame([raw_data])

# Preprocess the input data
# Ensure the input DataFrame has the same feature order as training data
input_df = input_df[features]

# Scale the input features
scaled_input = scaler.transform(input_df)

# Apply polynomial features transformation
poly_input = poly_features.transform(scaled_input)

# Make prediction
if st.button('Predict Median House Value'):
    prediction = model.predict(poly_input)[0]
    st.success(f"The predicted median house value is: **${prediction:,.2f}**")
    
st.markdown("---")
st.info("Adjust the feature values in the sidebar to get a new prediction.")

st.markdown("## Data Visualizations")
st.write("Explore some key distributions and relationships in the housing data.")

# Placeholder for the full DataFrame for visualizations (can be a subset for performance if needed)
# In a real deployment, you might load the full dataset here or use aggregated data.
# For this example, we'll assume a representative sample or the original df is conceptually available
# from the training context for visualization purposes, or loaded separately if needed.

# --- Distribution of Median House Value ---
st.subheader("Distribution of Median House Value")
# To display the distribution of median_house_value, we need access to the original 'df' or 'df_cleaned'
# Since 'df_cleaned' was created in the notebook by dropping NaNs, we'll use that as the source for visualization if available.
# If not, you'd need to load a sample or the full dataset here.

# In a real scenario, you'd load df_cleaned or a representative sample here if not already available.
# For demonstration, let's create a dummy dataframe if df_cleaned is not directly accessible here.
# In this environment, 'df_cleaned' is available from kernel state.

# To avoid issues with data scope in Streamlit app, for demonstration purposes we will load the data again 
# or instruct the user to ensure the data is loaded.
# For now, let's assume `df_cleaned` represents the data to visualize.

# For the Streamlit app, we need to load the data again or have it accessible.
# Let's assume we load the full dataset for visualization within Streamlit for simplicity.
# NOTE: In a production app, you might only visualize a sample or pre-aggregate if the dataset is very large.

@st.cache_data # Cache data loading for performance
def load_data():
    # Load the dataset again, ensuring it's available for plotting
    data = pd.read_csv('/content/housing.csv')
    data_cleaned = data.dropna()
    return data_cleaned

df_for_viz = load_data()

fig_hist = plt.figure(figsize=(10, 6))
sns.histplot(df_for_viz['median_house_value'], kde=True, bins=50)
plt.title('Distribution of Median House Value')
plt.xlabel('Median House Value')
plt.ylabel('Count')
st.pyplot(fig_hist)

# --- Median Income vs. Median House Value ---
st.subheader("Median Income vs. Median House Value")
fig_scatter = plt.figure(figsize=(10, 6))
sns.scatterplot(x='median_income', y='median_house_value', data=df_for_viz, alpha=0.6)
plt.title('Median Income vs. Median House Value')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
st.pyplot(fig_scatter)

st.markdown("---")
st.write("**Note:** To see the updated visualizations, you need to re-run the `!streamlit run app.py` command after this cell has executed.")
