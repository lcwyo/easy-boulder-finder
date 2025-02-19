import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any

from utils import (
    SimpleMLPRegression, 
    ClimbingModelConfig, 
    load_data_from_db, 
    preprocess_data, 
    get_features
)

class ModelVersionManager:
    def __init__(self, 
                 model_dir='model_versions', 
                 performance_threshold=0.1,
                 retrain_interval_days=90):
        """
        Initialize Model Version Manager
        
        Args:
            model_dir (str): Directory to store model versions
            performance_threshold (float): Performance drop threshold for retraining
            retrain_interval_days (int): Days between mandatory retrains
        """
        self.model_dir = model_dir
        self.performance_threshold = performance_threshold
        self.retrain_interval_days = retrain_interval_days
        
        os.makedirs(model_dir, exist_ok=True)
    
    def _get_model_metadata_path(self, version):
        return os.path.join(self.model_dir, f'model_v{version}_metadata.json')
    
    def _get_model_path(self, version):
        return os.path.join(self.model_dir, f'model_v{version}.pth')
    
    def _calculate_performance_metrics(self, predictions, actual):
        """
        Calculate model performance metrics
        
        Args:
            predictions (np.ndarray): Model predictions
            actual (np.ndarray): Actual route difficulties
        
        Returns:
            Dict[str, float]: Performance metrics
        """
        mae = np.mean(np.abs(predictions - actual))
        mse = np.mean((predictions - actual)**2)
        r2 = 1 - (np.sum((actual - predictions)**2) / np.sum((actual - np.mean(actual))**2))
        
        return {
            'mean_absolute_error': float(mae),
            'mean_squared_error': float(mse),
            'r2_score': float(r2)
        }
    
    def train_and_save_model(self, routes, routes_data):
        """
        Train a new model version and save it
        
        Args:
            routes (np.ndarray): Route features
            routes_data (pd.DataFrame): Route metadata
        
        Returns:
            Dict[str, Any]: Model version metadata
        """
        # Determine next version number
        existing_versions = [
            int(f.split('_v')[1].split('_')[0]) 
            for f in os.listdir(self.model_dir) 
            if f.startswith('model_v') and f.endswith('_metadata.json')
        ]
        version = max(existing_versions) + 1 if existing_versions else 1
        
        # Prepare data
        X = routes
        y = routes_data['difficulty_average'].values
        
        # Create configuration
        config = ClimbingModelConfig(input_size=X.shape[1])
        model = SimpleMLPRegression(config)
        
        # Training logic would go here (simplified for this example)
        # In a real scenario, you'd split data, use proper training loop, etc.
        
        # Make predictions for performance metrics
        with torch.no_grad():
            input_tensor = torch.tensor(X, dtype=torch.float32)
            predictions = model(input_ids=input_tensor)['logits'].numpy().flatten()
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(predictions, y)
        
        # Prepare metadata
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'total_routes': len(routes),
            'performance': performance,
            'input_features': X.shape[1],
            'difficulty_range': {
                'min': float(routes_data['difficulty_average'].min()),
                'max': float(routes_data['difficulty_average'].max()),
                'mean': float(routes_data['difficulty_average'].mean())
            }
        }
        
        # Save model
        model_path = self._get_model_path(version)
        torch.save(model.state_dict(), model_path)
        
        # Save metadata
        metadata_path = self._get_model_metadata_path(version)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return metadata
    
    def should_retrain(self):
        """
        Determine if model should be retrained
        
        Returns:
            bool: Whether retraining is recommended
        """
        if not os.listdir(self.model_dir):
            return True
        
        # Get latest model metadata
        latest_metadata_file = max(
            [f for f in os.listdir(self.model_dir) if f.endswith('_metadata.json')],
            key=lambda x: os.path.getctime(os.path.join(self.model_dir, x))
        )
        
        with open(os.path.join(self.model_dir, latest_metadata_file), 'r') as f:
            latest_metadata = json.load(f)
        
        # Check time since last training
        last_train_time = datetime.fromisoformat(latest_metadata['timestamp'])
        days_since_train = (datetime.now() - last_train_time).days
        
        # Check performance threshold
        performance = latest_metadata['performance']
        
        return (
            days_since_train >= self.retrain_interval_days or
            performance['mean_absolute_error'] > self.performance_threshold
        )
    
    def get_latest_model(self):
        """
        Retrieve the latest trained model
        
        Returns:
            SimpleMLPRegression: Latest model
        """
        if not os.listdir(self.model_dir):
            raise ValueError("No models have been trained yet")
        
        # Get latest model file
        latest_model_file = max(
            [f for f in os.listdir(self.model_dir) if f.endswith('.pth')],
            key=lambda x: os.path.getctime(os.path.join(self.model_dir, x))
        )
        
        # Load model configuration and state
        config = ClimbingModelConfig()
        model = SimpleMLPRegression(config)
        model.load_state_dict(torch.load(os.path.join(self.model_dir, latest_model_file)))
        
        return model

def main():
    # Load data
    holes_df, routes_df, routes_grade_df, _ = load_data_from_db()
    routes_l1 = preprocess_data(routes_df, routes_grade_df, min_ascents=1)
    routes, routes_data = get_features(routes_l1, holes_df, routes_grade_df)
    
    # Initialize version manager
    version_manager = ModelVersionManager()
    
    # Check if retraining is needed
    if version_manager.should_retrain():
        print("Retraining model...")
        version_metadata = version_manager.train_and_save_model(routes, routes_data)
        print(f"New model version {version_metadata['version']} trained.")
    else:
        print("No retraining needed at this time.")
        latest_model = version_manager.get_latest_model()
        print(f"""Using latest model: {latest_model}""")

if __name__ == "__main__":
    main()
