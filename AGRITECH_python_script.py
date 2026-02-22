# ===============================================
# AgriTech Crop Yield Prediction Backend Script
# ===============================================

import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Load saved objects
# -----------------------------

with open("AGRITECH_Decision_Tree.pkl", "rb") as file:
    tree_reg = pickle.load(file)

with open("AGRITECH_scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("AGRITECH_LabelEncoder.pkl", "rb") as file:
    le = pickle.load(file)

# -----------------------------
# Define prediction function
# -----------------------------

def predict_crop_yield(input_data):
    """
    Predicts crop yield for a given input dictionary.

    Parameters:
    -----------
    input_data : dict
        Input features with keys:
        Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp

    Returns:
    --------
    float
        Predicted crop yield (hg/ha)
    """

    # -----------------------------
    # Encode categorical Item (only column left)
    # -----------------------------
    item_encoded = le.transform([input_data["Item"]])[0]

    # -----------------------------
    # Create DataFrame with exact feature order
    # -----------------------------
    feature_df = pd.DataFrame([{
        "Item": item_encoded,
        "Year": input_data["Year"],
        "average_rain_fall_mm_per_year": input_data["average_rain_fall_mm_per_year"],
        "pesticides_tonnes": input_data["pesticides_tonnes"],
        "avg_temp": input_data["avg_temp"]
    }])

    # -----------------------------
    # Scale features
    # -----------------------------
    features_scaled = scaler.transform(feature_df)

    # -----------------------------
    # Predict
    # -----------------------------
    predicted_yield = tree_reg.predict(features_scaled)

    return float(predicted_yield[0])