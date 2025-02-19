import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import SimpleMLPRegression, ClimbingModelConfig, load_data_from_db, preprocess_data, get_features

def prepare_features(routes):
    """
    Prepare and scale features for model input
    
    Args:
        routes (np.ndarray): Raw route features
    
    Returns:
        torch.Tensor: Scaled and prepared input tensor
    """
    # Scale features
    scaler = StandardScaler()
    scaled_routes = scaler.fit_transform(routes)
    
    # Convert to float32 tensor
    return torch.tensor(scaled_routes, dtype=torch.float32)

def analyze_route_misclassification(results_df):
    """
    Analyze routes that might be misclassified
    
    Args:
        results_df (pd.DataFrame): DataFrame with route predictions
    
    Returns:
        pd.DataFrame: Routes sorted by potential misclassification
    """
    # Calculate the discrepancy between actual and predicted difficulty
    results_df['difficulty_discrepancy'] = np.abs(results_df['Actual Difficulty'] - results_df['Predicted Difficulty'])
    
    # Calculate a misclassification score
    # Higher score indicates more likely misclassification
    results_df['misclassification_score'] = (
        results_df['difficulty_discrepancy'] / 
        (1 + np.log1p(results_df['Ascent Count']))
    )
    
    # Sort by misclassification score in descending order
    misclassified_routes = results_df.sort_values('misclassification_score', ascending=False)
    
    return misclassified_routes

def main():
    try:
        # Load data to get feature representation
        holes_df, routes_df, routes_grade_df, _ = load_data_from_db()
        routes_l1 = preprocess_data(routes_df, routes_grade_df, min_ascents=1)
        routes, routes_data = get_features(routes_l1, holes_df, routes_grade_df)
        
        # Diagnostic information
        print("\n--- Feature Diagnostics ---")
        print(f"Routes shape: {routes.shape}")
        print(f"Routes dtype: {routes.dtype}")
        print(f"Routes sample (first 5 rows, first 10 columns):\n{routes[:5, :10]}")
        
        # Create configuration
        config = ClimbingModelConfig(input_size=routes.shape[1])
        
        # Prepare input tensor
        input_tensor = prepare_features(routes[:10])
        labels = torch.tensor(routes_data['difficulty_average'][:10].values, dtype=torch.float32)
        print(f"Labels: {labels}")
        print("\n--- Input Tensor Diagnostics ---")
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Input tensor dtype: {input_tensor.dtype}")
        print(f"Input tensor sample (first 5 rows, first 10 columns):\n{input_tensor[:5, :10]}")
        
        # Initialize model
        print("\nInitializing model...")
        model = SimpleMLPRegression(config)
        
        # Make predictions
        print("\nMaking predictions...")
        with torch.no_grad():
            predictions = model(input_ids=input_tensor)
        
        # Print results
        print("\nPrediction Results:")
        results_df = pd.DataFrame({
            'UUID': routes_data['uuid'][:10],
            'Actual Difficulty': routes_data['difficulty_average'][:10],
            'Predicted Difficulty': predictions['logits'].numpy().flatten(),
            'Adjusted Difficulty': routes_data['adjusted_difficulty'][:10],
            'Ascent Count': routes_data['ascensionist_count'][:10]
        })
        results_df['Difference'] = abs(results_df['Actual Difficulty'] - results_df['Predicted Difficulty'])
        
        print(results_df.to_string(index=False))
        
        # Analyze potential route misclassifications
        print("\n--- Route Misclassification Analysis ---")
        misclassified_routes = analyze_route_misclassification(results_df)
        print(misclassified_routes.to_string(index=False))
        
        # Additional model diagnostics
        print("\n--- Model Architecture Diagnostics ---")
        print(model)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
