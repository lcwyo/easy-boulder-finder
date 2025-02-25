import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import math
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
import sys

# Import utility functions from a separate module
from utils import (
    load_data_from_db,
    preprocess_data,
    get_features,
    RoutesDataset,
    SimpleMLPRegression,
    initialize_database_if_needed,
)


#logger.remove()  # Remove the default handler.
#logger.add(sys.stderr, colorize=True, backtrace=True, diagnose=True)  # Force ANSI sequences in output.

logger.add("file.log", level="INFO", rotation="500 MB")  # Also log to a file, rotating every 500 MB.

# Constants
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
MIN_ASCENTS = 5  # needs this many ascents to boulder to be included


def load_and_preprocess_data():
    """Load and preprocess data."""
    logger.info("Loading data")
    holes_df, routes_df, routes_grade_df, _ = load_data_from_db()
    logger.info("Data Loaded")

    logger.info("Preprocessing data")
    routes_l1 = preprocess_data(routes_df, routes_grade_df, min_ascents=MIN_ASCENTS)
    logger.info(f"Data preprocessed, {len(routes_l1)} boulder problems")

    logger.info("Extracting features")
    routes = get_features(routes_l1, holes_df)
    logger.info(f"Features extracted. Got {routes.shape} features")

    return routes, routes_l1


def create_datasets_and_loaders(routes, routes_l1):
    """Create datasets and dataloaders."""
    logger.info("Creating dataset")
    full_dataset = RoutesDataset(
        routes, routes_l1["difficulty_average"].values, routes_l1["angle"].values
    )

    # Splitting the dataset
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    logger.info("Creating dataloaders")
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def train_and_evaluate_model(train_loader, test_loader):
    """Train and evaluate the model."""

    data_iterator = iter(train_loader)
    images, _ = next(data_iterator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        SimpleMLPRegression(images.shape[-1]).to(device).double()
    )  # Convert model to double precision

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    test_losses, train_losses = [], []

    logger.info("Starting training")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, device
        )
        test_loss = evaluate(model, test_loader, criterion, device)

        test_losses.append(test_loss)
        train_losses.append(train_loss)

        logger.info(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
        )

    save_model(model, "model.pth")
    plot_losses(train_losses, test_losses)


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device):
    """Train the model for one epoch."""
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
    )

    for _, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        progress_bar.set_postfix({"Train Loss": loss.item()})

    train_loss /= len(train_loader.dataset)
    return train_loss


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
    test_loss /= len(test_loader.dataset)

    return test_loss


def save_model(model, model_path):
    """Save the trained model to a file."""
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


def plot_losses(train_losses, test_losses):
    """Plot training and testing losses over epochs."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, test_losses, color="red", label="Testing Loss")

    plt.title("Training and Testing Losses Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


@dataclass(frozen=True)  # Make the dataclass immutable
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

    def __post_init__(self):
        """Validate the data after initialization."""
        if self.rank <= 0:
            raise ValueError("Rank must be positive")
        if not self.uuid:
            raise ValueError("UUID cannot be empty")


class PredictionAnalyzer:
    """Analyzes differences between predicted and actual values."""
    
    def __init__(
        self,
        grade_comparison: Dict[int, str],
        routes_df: pd.DataFrame,
        n_top_differences: int = 20
    ) -> None:
        """Initialize the PredictionAnalyzer.
        
        Args:
            grade_comparison: Mapping of difficulty values to grade names
            routes_df: DataFrame containing route information
            n_top_differences: Number of top differences to analyze
        """
        if n_top_differences <= 0:
            raise ValueError("n_top_differences must be positive")
            
        self.grade_comparison = grade_comparison
        self.routes_df = routes_df
        self.n_top_differences = n_top_differences
    
    def _get_grade_name(self, value: float) -> str:
        """Get grade name from numerical value.
        
        Args:
            value: Numerical grade value
            
        Returns:
            String representation of the grade
        """
        try:
            return self.grade_comparison[math.floor(value)]
        except KeyError:
            logger.warning(f"Unknown grade value: {value}")
            return "Unknown Grade"
    
    def _get_route_info(self, uuid: str) -> Dict[str, str]:
        """Get route information from DataFrame.
        
        Args:
            uuid: Route UUID
            
        Returns:
            Dictionary containing route information
        """
        default_info = {
            'setter': 'Unknown',
            'name': 'Unknown',
            'angle': 'Unknown'
        }
        
        try:
            route = self.routes_df.loc[
                self.routes_df['uuid'] == uuid,
                ['setter_username', 'name', 'angle']
            ].iloc[0]
            
            return {
                'setter': route['setter_username'],
                'name': route['name'],
                'angle': str(route['angle'])
            }
        except (IndexError, KeyError) as e:
            logger.warning(f"Failed to get route info for UUID {uuid}: {str(e)}")
            return default_info
    
    def analyze_differences(
        self,
        differences: torch.Tensor,
        uuids: List[str],
        actual_values: torch.Tensor,
        predictions: torch.Tensor
    ) -> List[PredictionDifference]:
        """Analyze differences between predicted and actual values.
        
        Args:
            differences: Tensor of differences between actual and predicted values
            uuids: List of route UUIDs
            actual_values: Tensor of actual values
            predictions: Tensor of predicted values
            
        Returns:
            List of PredictionDifference objects
        """
        if len(uuids) != len(actual_values) or len(uuids) != len(predictions):
            raise ValueError("Length mismatch between inputs")
            
        differences_np = differences.cpu().numpy()
        top_indices = np.argsort(np.abs(differences_np))[-self.n_top_differences:]
        
        results = []
        for rank, index in enumerate(reversed(top_indices), start=1):
            try:
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
            except Exception as e:
                logger.error(f"Error processing index {index}: {str(e)}")
                continue
            
        return results
    
    def display_differences(self, differences: List[PredictionDifference]) -> None:
        """Display analyzed differences in a formatted way.
        
        Args:
            differences: List of PredictionDifference objects to display
        """
        if not differences:
            logger.warning("No differences to display")
            return
            
        print(f"\nTop {self.n_top_differences} datapoints with the biggest prediction differences:")
        print("-" * 80)
        
        for diff in differences:
            try:
                self._display_single_difference(diff)
            except Exception as e:
                logger.error(f"Error displaying difference: {str(e)}")
                continue
    
    def _display_single_difference(self, diff: PredictionDifference) -> None:
        """Display a single prediction difference.
        
        Args:
            diff: PredictionDifference object to display
        """
        print(f"\nRank {diff.rank} | Index {diff.index} | UUID: {diff.uuid}")
        print(f"Actual: {diff.actual_value:.2f} ({diff.actual_grade})")
        print(f"Predicted: {diff.predicted_value:.2f} ({diff.predicted_grade})")
        print(f"Difference: {diff.difference:.2f}")
        print(f"Route Info:")
        print(f"  - Setter: {diff.setter_info['setter']}")
        print(f"  - Name: {diff.setter_info['name']}")
        print(f"  - Angle: {diff.setter_info['angle']}")
        print("-" * 40)


if __name__ == "__main__":
    initialize_database_if_needed()

    routes, routes_l1 = load_and_preprocess_data()
    train_loader, test_loader = create_datasets_and_loaders(routes, routes_l1)
    train_and_evaluate_model(train_loader, test_loader)
