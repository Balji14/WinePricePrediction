import streamlit as st
import pandas as pd
import joblib

# Load model and training columns
model, model_columns = joblib.load("wine_model.pkl")

# Streamlit page setup
st.set_page_config(page_title="Wine Price Prediction", page_icon="🍷", layout="centered")

st.title("🍷 Wine Price Prediction App")
st.markdown("""
Enter the wine details below and click **Predict Price**  
to estimate its value using the trained Random Forest model.
""")

# Input fields
col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Wine Name", "Pomerol 2011")
    country = st.text_input("Country", "France")
    region = st.text_input("Region", "Pomerol")
    winery = st.text_input("Winery", "Château La Providence")

with col2:
    rating = st.number_input("Rating", min_value=0.0, max_value=5.0, value=4.2, step=0.1)
    num_ratings = st.number_input("Number of Ratings", min_value=0, value=100, step=1)
    year = st.number_input("Year", min_value=1900, max_value=2025, value=2011, step=1)

st.markdown("---")

# Predict button
if st.button("🔮 Predict Price"):
    try:
        # Prepare input
        input_df = pd.DataFrame({
            "Name": [name],
            "Country": [country],
            "Region": [region],
            "Winery": [winery],
            "Rating": [rating],
            "NumberOfRatings": [num_ratings],
            "Year": [year]
        })

        # Convert categorical columns into dummies
        input_df = pd.get_dummies(input_df)

        # Align columns with training data
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_df)[0]

        st.success(f"💰 Predicted Wine Price: **${prediction:.2f}**")

    except Exception as e:
        st.error(f"⚠️ Error while predicting: {e}")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit and Scikit-learn")
