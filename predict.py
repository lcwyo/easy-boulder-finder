import torch
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import (
    load_data_from_db, 
    preprocess_data, 
    get_features, 
    SimpleMLPRegression, 
    ClimbingModelConfig,
    initialize_database_if_needed
)

MODEL_PATH = "./my_climbing_model.pth"
N_TOP_DIFFERENCES = 20
MIN_ASCENTS = 1

def load_and_prepare_model(model_path):
    """
    Load the pretrained model from the given path.
    
    Args:
        model_path (str): Path to the saved model file
    
    Returns:
        SimpleMLPRegression: Loaded and prepared model
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file at {model_path} not found.")
        sys.exit(1)

    # Load the data to determine input size
    holes_df, routes_df, routes_grade_df, _ = load_data_from_db()
    routes_l1 = preprocess_data(routes_df, routes_grade_df, min_ascents=MIN_ASCENTS)
    routes = get_features(routes_l1, holes_df)
    
    # Create configuration with correct input size
    config = ClimbingModelConfig(input_size=routes.shape[1])
    
    # Initialize and load model
    model = SimpleMLPRegression(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print(f"Model loaded successfully. Input size: {routes.shape[1]}")
    return model, routes.shape[1]

def prepare_features_for_model(routes, routes_l1):
    """
    Prepare and scale features for model input.
    
    Args:
        routes (np.ndarray): Feature matrix
        routes_l1 (pd.DataFrame): Routes metadata
    
    Returns:
        tuple: Scaled features, actual values, UUIDs
    """
    # Flatten features
    features = routes.reshape(routes.shape[0], -1).astype(np.float32)
    
    # Prepare labels and metadata
    actual_values = routes_l1['difficulty_average'].values
    uuids = routes_l1['uuid'].values
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, actual_values, uuids

def make_predictions(model, features, actual_values, batch_size=1024):
    """
    Make predictions using the model and compare with actual values in batches.
    
    Args:
        model (SimpleMLPRegression): Trained model
        features (np.ndarray): Scaled input features
        actual_values (np.ndarray): Actual difficulty values
        batch_size (int, optional): Number of samples to process in each batch
    
    Returns:
        tuple: Predictions, differences
    """
    # Preallocate arrays for predictions and differences
    predictions = np.zeros_like(actual_values, dtype=np.float32)
    differences = np.zeros_like(actual_values, dtype=np.float32)
    
    # Process in batches to manage memory
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        
        # Extract batch
        batch_features = features[start:end]
        batch_actual_values = actual_values[start:end]
        
        # Convert to torch tensors
        features_tensor = torch.tensor(batch_features, dtype=torch.float32)
        
        # Disable gradient computation
        with torch.no_grad():
            model_output = model(input_ids=features_tensor)
            
            # Extract logits from the model output dictionary
            batch_predictions = model_output.get('logits', model_output).squeeze().numpy()
        
        # Store results
        predictions[start:end] = batch_predictions
        differences[start:end] = np.abs(batch_predictions - batch_actual_values)
    
    return predictions, differences

def display_top_differences(differences, uuids, actual_values, predictions, routes_df, grade_comparison, N=20):
    """
    Display top N differences between actual and predicted values.
    
    Args:
        differences (np.ndarray): Absolute differences
        uuids (np.ndarray): Route UUIDs
        actual_values (np.ndarray): Actual difficulty values
        predictions (np.ndarray): Predicted difficulty values
        routes_df (pd.DataFrame): Routes dataframe
        grade_comparison (dict): Grade comparison dictionary
        N (int, optional): Number of top differences to display. Defaults to 20.
    """
    # Sort indices by differences in descending order
    top_indices = np.argsort(differences)[-N:][::-1]
    
    print("\nTop {} Routes with Largest Prediction Differences:".format(N))
    print("{:<10} {:<20} {:<15} {:<15} {:<15}".format(
        "Rank", "UUID", "Actual Difficulty", "Predicted", "Difference"
    ))
    
    for rank, idx in enumerate(top_indices, 1):
        uuid = uuids[idx]
        actual = actual_values[idx]
        predicted = predictions[idx]
        diff = differences[idx]
        
        # Try to get route name from routes_df if possible
        route_name = routes_df[routes_df['uuid'] == uuid]['name'].values[0] if not routes_df[routes_df['uuid'] == uuid].empty else "Unknown"
        
        print("{:<10} {:<20} {:<15.2f} {:<15.2f} {:<15.2f}".format(
            rank, uuid, actual, predicted, diff
        ))

def main():
    """Main prediction workflow."""
    # Initialize database if needed
    initialize_database_if_needed()
    
    # Load model
    model, input_size = load_and_prepare_model(MODEL_PATH)
    
    # Load and preprocess data
    holes_df, routes_df, routes_grade_df, grade_comparision_df = load_data_from_db()
    grade_comparison = grade_comparision_df.set_index('difficulty')['boulder_name'].to_dict()
    
    # Preprocess routes
    routes_l1 = preprocess_data(routes_df, routes_grade_df, min_ascents=MIN_ASCENTS)
    routes = get_features(routes_l1, holes_df)
    
    # Prepare features
    features, actual_values, uuids = prepare_features_for_model(routes, routes_l1)
    
    # Make predictions
    predictions, differences = make_predictions(model, features, actual_values)
    
    # Display results
    display_top_differences(
        differences, 
        uuids, 
        actual_values, 
        predictions, 
        routes_df, 
        grade_comparison, 
        N=N_TOP_DIFFERENCES
    )
    
    # Calculate overall performance metrics
    mse = np.mean((predictions - actual_values)**2)
    mae = np.mean(np.abs(predictions - actual_values))
    
    print("\nOverall Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

if __name__ == "__main__":
    main()
