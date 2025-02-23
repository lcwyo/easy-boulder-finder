import torch
import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
import math
from typing import Dict, List, Tuple
import pandas as pd
from dataclasses import dataclass

from utils import (
    load_data_from_db,
    preprocess_data,
    get_features,
    SimpleMLPRegression,
    initialize_database_if_needed,
)

MODEL_PATH = "model.pth"
N_TOP_DIFFERENCES = 20
MIN_ASCENTS = 5

@dataclass
class PredictionDifference:
    """Data class to store prediction difference information."""
    rank: int
    index: int
    uuid: str
    actual_value: float
    predicted_value: float
    difference: float
    actual_grade: str
    predicted_grade: str
    setter_info: Dict[str, str]

class PredictionAnalyzer:
    """Analyzes differences between predicted and actual values."""
    
    def __init__(
        self,
        grade_comparison: Dict[int, str],
        routes_df: pd.DataFrame,
        n_top_differences: int = 20
    ):
        self.grade_comparison = grade_comparison
        self.routes_df = routes_df
        self.n_top_differences = n_top_differences
    
    def _get_grade_name(self, value: float) -> str:
        """Get grade name from numerical value."""
        try:
            return self.grade_comparison[math.floor(value)]
        except KeyError:
            return "Unknown Grade"
    
    def _get_route_info(self, uuid: str) -> Dict[str, str]:
        """Get route information from DataFrame."""
        try:
            route = self.routes_df[
                self.routes_df['uuid'] == uuid
            ][['setter_username', 'name', 'angle']].iloc[0]
            
            return {
                'setter': route['setter_username'],
                'name': route['name'],
                'angle': str(route['angle'])
            }
        except (IndexError, KeyError):
            return {
                'setter': 'Unknown',
                'name': 'Unknown',
                'angle': 'Unknown'
            }
    
    def analyze_differences(
        self,
        differences: torch.Tensor,
        uuids: List[str],
        actual_values: torch.Tensor,
        predictions: torch.Tensor
    ) -> List[PredictionDifference]:
        """Analyze differences between predicted and actual values."""
        differences_np = differences.cpu().numpy()
        top_indices = np.argsort(np.abs(differences_np))[-self.n_top_differences:]
        
        results = []
        for rank, index in enumerate(reversed(top_indices), start=1):
            uuid = uuids[index]
            actual = actual_values[index].item()
            predicted = predictions[index].item()
            diff = differences[index].item()
            
            result = PredictionDifference(
                rank=rank,
                index=index,
                uuid=uuid,
                actual_value=actual,
                predicted_value=predicted,
                difference=diff,
                actual_grade=self._get_grade_name(actual),
                predicted_grade=self._get_grade_name(predicted),
                setter_info=self._get_route_info(uuid)
            )
            results.append(result)
            
        return results
    
    def display_differences(self, differences: List[PredictionDifference]):
        """Display analyzed differences in a formatted way."""
        print(f"\nTop {self.n_top_differences} datapoints with the biggest prediction differences:")
        print("-" * 80)
        
        for diff in differences:
            print(f"\nRank {diff.rank} | Index {diff.index} | UUID: {diff.uuid}")
            print(f"Actual: {diff.actual_value:.2f} ({diff.actual_grade})")
            print(f"Predicted: {diff.predicted_value:.2f} ({diff.predicted_grade})")
            print(f"Difference: {diff.difference:.2f}")
            print(f"Route Info:")
            print(f"  - Setter: {diff.setter_info['setter']}")
            print(f"  - Name: {diff.setter_info['name']}")
            print(f"  - Angle: {diff.setter_info['angle']}")
            print("-" * 40)


def load_and_prepare_model(model_path):
    """Load the pretrained model from the given path."""
    if not os.path.exists(model_path):
        print(f"Error: Model file at {model_path} not found.")
        sys.exit(1)  # Exit the program with an error status.

    model = SimpleMLPRegression(
        507
    )  # Assuming 'SimpleMLPRegression' is defined in 'utils'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully.")
    return model


def preprocess_and_extract_features():
    """Load, preprocess data, and extract features."""
    print("Loading data")
    holes_df, routes_df, routes_grade_df, grade_comparision_df = load_data_from_db()
    grade_comparision = grade_comparision_df.set_index("difficulty")[
        "boulder_name"
    ].to_dict()
    print("Data Loaded")

    # Only include boulders that are publicly listed in app
    print(
        f"Excluding boulders that are not publicly listed, size before: {len(routes_df)}"
    )
    routes_df = routes_df[routes_df["is_listed"] == 1]
    print(f"Size after: {len(routes_df)}")

    print("Preprocessing data")
    routes_l1 = preprocess_data(routes_df, routes_grade_df, min_ascents=MIN_ASCENTS)
    print(f"Data preprocessed, {len(routes_l1)} boulder problems")

    print("Extracting features")
    routes = get_features(routes_l1, holes_df)
    print(f"Features extracted. Got {routes.shape} features")

    return routes, routes_l1, routes_df, grade_comparision


def prepare_features_for_model(routes, routes_l1):
    """Prepare and scale features for model input."""
    print("Preparing features")
    N, _, _ = routes.shape
    features = np.hstack(
        [routes.reshape(N, -1), routes_l1["angle"].values.reshape(-1, 1)]
    )
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return torch.tensor(features, dtype=torch.float), routes_l1["uuid"]


def make_predictions(model, features, actual_values):
    """Make predictions using the model and compare with actual values."""
    print("Making Predictions")
    with torch.no_grad():  # Disable gradient computation
        predictions = model(features).squeeze()
    differences = actual_values - predictions
    return predictions, differences


def display_top_differences(
    differences,
    uuids,
    actual_values,
    predictions,
    routes_df,
    grade_comparision,
    N=N_TOP_DIFFERENCES,
):
    """Display top N differences between actual and predicted values."""
    differences_np = differences.cpu().numpy()
    top_indices = np.argsort(differences_np)[-N:]
    print(grade_comparision, predictions[637])
    print(f"Top {N} datapoints with the biggest prediction differences:")
    for rank, index in enumerate(reversed(top_indices), start=1):
        uuid = uuids[index]
        # print(routes_df[routes_df['uuid'] == uuid].columns)
        # break
        print(f"\nRank {rank}, Datapoint {index}, UUID: {uuid}")
        print(
            f"Actual: {actual_values[index].item():.2f} ({grade_comparision[math.floor(actual_values[index].item())]}), Predicted: {predictions[index].item():.2f} ({grade_comparision[math.floor(predictions[index].item())]}), Difference: {differences[index].item():.2f}"
        )
        subset = routes_df[routes_df["uuid"] == uuid][
            ["setter_username", "name", "angle"]
        ]
        print(subset.to_string(index=False))


def main():
    initialize_database_if_needed()

    model = load_and_prepare_model(MODEL_PATH)
    routes, routes_l1, routes_df, grade_comparision = preprocess_and_extract_features()
    features, uuids = prepare_features_for_model(routes, routes_l1)
    actual_values = torch.tensor(
        routes_l1["difficulty_average"].values, dtype=torch.float
    )
    predictions, differences = make_predictions(model, features, actual_values)

    # Initialize analyzer
    analyzer = PredictionAnalyzer(
        grade_comparison=grade_comparision,
        routes_df=routes_df,
        n_top_differences=N_TOP_DIFFERENCES
    )
    
    # Analyze differences
    analysis_results = analyzer.analyze_differences(
        differences=differences,
        uuids=uuids,
        actual_values=actual_values,
        predictions=predictions
    )
    
    # Display results
    analyzer.display_differences(analysis_results)


if __name__ == "__main__":
    main()
