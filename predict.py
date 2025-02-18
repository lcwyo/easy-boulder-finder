import torch
import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
import math
import pandas as pd  # Import pandas explicitly

from utils import load_data_from_db, preprocess_data, get_features, SimpleMLPRegression, initialize_database_if_needed

MODEL_PATH = "model.pth"
N_TOP_DIFFERENCES = 20
MIN_ASCENTS = 1

def load_and_prepare_model(model_path):
    # ... (no changes needed)

def preprocess_and_extract_features():
    """Load, preprocess data, and extract features."""
    print("Loading data")
    holes_df, routes_df, routes_grade_df, grade_comparision_df = load_data_from_db()
    
    # Check if the dataframes are empty after the database call
    if holes_df.empty or routes_df.empty or routes_grade_df.empty or grade_comparision_df.empty:
        print("Error: One or more dataframes are empty after loading from the database. Check the database connection and data.")
        sys.exit(1)

    grade_comparision = grade_comparision_df.set_index('difficulty')['boulder_name'].to_dict()
    print("Data Loaded")

    # Only include boulders that are publicly listed in app
    print(f"Excluding boulders that are not publicly listed, size before: {len(routes_df)}")
    routes_df = routes_df[routes_df['is_listed'] == 1]
    print(f"Size after: {len(routes_df)}")

    print("Preprocessing data")
    routes_l1 = preprocess_data(routes_df, routes_grade_df, min_ascents=MIN_ASCENTS) # Corrected parameter name
    print(f"Data preprocessed, {len(routes_l1)} boulder problems")

    print("Extracting features")
    routes = get_features(routes_l1, holes_df)
    print(f"Features extracted. Got {routes.shape} features")

    return routes, routes_l1, routes_df, grade_comparision

def prepare_features_for_model(routes, routes_l1):
    # ... (no changes needed)

def make_predictions(model, features, actual_values):
    # ... (no changes needed)

def display_top_differences(differences, uuids, actual_values, predictions, routes_df, grade_comparision, N=N_TOP_DIFFERENCES):
    # ... (no changes needed)

def main():
    initialize_database_if_needed()

    model = load_and_prepare_model(MODEL_PATH)
    routes, routes_l1, routes_df, grade_comparision = preprocess_and_extract_features()

    # Check if routes or routes_l1 are empty after feature extraction
    if routes is None or routes_l1.empty:
        print("Error: No routes or labels available after feature extraction. Check the data processing steps.")
        sys.exit(1)

    features, uuids = prepare_features_for_model(routes, routes_l1)
    actual_values = torch.tensor(routes_l1['difficulty_average'].values, dtype=torch.float)

    # Check if features or actual_values have the correct size
    if features.shape[0] != actual_values.shape[0]:
        print("Error: Number of features and labels do not match.")
        print(f"Features shape: {features.shape}, Actual values shape: {actual_values.shape}")
        sys.exit(1)

    predictions, differences = make_predictions(model, features, actual_values)
    display_top_differences(differences, uuids, actual_values, predictions, routes_df, grade_comparision)

if __name__ == "__main__":
    main()
