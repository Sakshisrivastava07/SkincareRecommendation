import streamlit as st
import pandas as pd
import joblib
# streamlit run app.py
# Exact feature order from training

import numpy as np





import streamlit as st
import pandas as pd
import joblib

# Load trained model and MultiLabelBinarizer
model = joblib.load("xgb_model.pkl")
mlb = joblib.load("mlb.pkl")

st.title("🧴 Skincare Product Recommender")

st.write("Fill out your skin concerns below to get product suggestions.")


# Force column order to match training
# training_columns = [
#     'wrinkles_severity', 'acne_severity', 'dark_circle_severity', 'pigmentation',
#     'redness', 'dark_spots', '_combination', '_dry', '_normal', '_oily',
#     '_sensitive', '_brown', '_dark', '_fair', '_medium', '_olive'
# ]


# input_df = input_df[training_columns]


# # Add missing columns if any (fill with 0)
# for col in training_columns:
#     if col not in input_df.columns:
#         input_df[col] = 0

# Drop extra columns if any
# input_df = input_df[training_columns]


# Input sliders for skin conditions
wrinkles = st.slider("Wrinkles Severity", 0, 10, 5)
acne = st.slider("Acne Severity", 0.0, 10.0, 5.0)
dark_circles = st.slider("Dark Circle Severity", 0.0, 10.0, 5.0)
pigmentation = st.slider("Pigmentation", 0, 10, 5)
redness = st.slider("Redness", 0.0, 10.0, 5.0)
dark_spots = st.selectbox("Do you have dark spots?", [0, 1])

# Skin type (one-hot encoding)
skin_type = st.selectbox("Skin Type", ['_dry', '_oily', '_combination', '_normal', '_sensitive'])

# Skin tone (one-hot encoding)
skin_tone = st.selectbox("Skin Tone", ['_fair', '_medium', '_olive', '_dark', '_brown'])

# Create input DataFrame
input_data = {
    'wrinkles_severity': wrinkles,
    'acne_severity': acne,
    'dark_circle_severity': dark_circles,
    'pigmentation': pigmentation,
    'redness': redness,
    'dark_spots': dark_spots,
    '_dry': 1 if skin_type == '_dry' else 0,
    '_oily': 1 if skin_type == '_oily' else 0,
    '_combination': 1 if skin_type == '_combination' else 0,
    '_normal': 1 if skin_type == '_normal' else 0,
    '_sensitive': 1 if skin_type == '_sensitive' else 0,
    '_fair': 1 if skin_tone == '_fair' else 0,
    '_medium': 1 if skin_tone == '_medium' else 0,
    '_olive': 1 if skin_tone == '_olive' else 0,
    '_dark': 1 if skin_tone == '_dark' else 0,
    '_brown': 1 if skin_tone == '_brown' else 0,
}

input_df = pd.DataFrame([input_data])

training_columns = [
    'wrinkles_severity', 'acne_severity', 'dark_circle_severity', 'pigmentation',
    'redness', 'dark_spots', '_combination', '_dry', '_normal', '_oily',
    '_sensitive', '_brown', '_dark', '_fair', '_medium', '_olive'
]


input_df = input_df[training_columns]


# Add missing columns if any (fill with 0)
for col in training_columns:
    if col not in input_df.columns:
        input_df[col] = 0


# Predict button
if st.button("Get Recommendations"):
    probs = model.predict_proba(input_df)

    # Apply custom threshold
    threshold = 0.3
    custom_pred = [[1 if prob[1] >= threshold else 0 for prob in col] for col in zip(*probs)]


    





    custom_pred = np.array(custom_pred).reshape(1, -1)  # ensure 2D array
    predicted_products = mlb.inverse_transform(custom_pred)

    # Decode product labels
    predicted_products = mlb.inverse_transform(custom_pred)

    # Show results
    if predicted_products[0]:
        st.success("Recommended Products:")
        for product in predicted_products[0]:
            st.write(f"- {product}")
    else:
        st.warning("No product recommended based on your current inputs.")

