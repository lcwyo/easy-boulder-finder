import os
import subprocess
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from logging import getLogger
from datetime import datetime

# Constants
DB_PATH = "kilter_data.db"
LAYOUT_ID = 1
MIN_ASCENTS = 5
SIZE = (177, 185)

logger = getLogger(__name__)

@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 128
    num_epochs: int = 10
    learning_rate: float = 0.001
    train_split: float = 0.8
    min_ascents: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir: Path = Path("models")
    log_dir: Path = Path("logs")

class TrainingMetrics(NamedTuple):
    """Stores training and validation metrics."""
    train_loss: float
    val_loss: float
    epoch: int

class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration.
        
        Args:
            config: Training configuration parameters
        """
        self.config = config
        self.device = torch.device(config.device)
        self._setup_directories()
        
    def _setup_directories(self) -> None:
        """Create necessary directories for models and logs."""
        self.config.model_dir.mkdir(exist_ok=True)
        self.config.log_dir.mkdir(exist_ok=True)
        
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int
    ) -> float:
        """Train model for one epoch.
        
        Args:
            model: Neural network model
            train_loader: DataLoader for training data
            criterion: Loss function
            optimizer: Optimization algorithm
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{self.config.num_epochs}"
        )
        
        for _, (features, targets) in progress_bar:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * features.size(0)
            progress_bar.set_postfix({"Train Loss": loss.item()})
            
        return total_loss / len(train_loader.dataset)
    
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """Evaluate model on validation set.
        
        Args:
            model: Neural network model
            val_loader: DataLoader for validation data
            criterion: Loss function
            
        Returns:
            Average validation loss
        """
        model.eval()
        total_loss = 0.0
        
        for features, targets in val_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            outputs = model(features).squeeze()
            loss = criterion(outputs, targets)
            total_loss += loss.item() * features.size(0)
            
        return total_loss / len(val_loader.dataset)
    
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> List[TrainingMetrics]:
        """Train model for specified number of epochs.
        
        Args:
            model: Neural network model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            
        Returns:
            List of training metrics for each epoch
        """
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        metrics: List[TrainingMetrics] = []
        best_val_loss = float('inf')
        
        logger.info("Starting training...")
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            val_loss = self.evaluate(model, val_loader, criterion)
            
            metrics.append(TrainingMetrics(train_loss, val_loss, epoch))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(model, "best_model.pt")
            
            logger.info(
                f"Epoch [{epoch + 1}/{self.config.num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
        return metrics
    
    def save_model(self, model: nn.Module, filename: str) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save
            filename: Name of the checkpoint file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.config.model_dir / f"{timestamp}_{filename}"
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': model.config if hasattr(model, 'config') else None,
            'timestamp': timestamp
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def plot_training_history(
        self,
        metrics: List[TrainingMetrics],
        save: bool = True
    ) -> None:
        """Plot training and validation loss history.
        
        Args:
            metrics: List of training metrics
            save: Whether to save the plot to file
        """
        epochs = range(1, len(metrics) + 1)
        train_losses = [m.train_loss for m in metrics]
        val_losses = [m.val_loss for m in metrics]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.config.log_dir / f"training_history_{timestamp}.png"
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()

def initialize_database_if_needed(db_path=DB_PATH):
    """Check if the database exists and initialize it if it doesn't."""
    if not os.path.exists(db_path):
        print(f"Database at {db_path} not found. Initializing database...")
        # Run the command to initialize the database
        command = f"python3 -m boardlib database kilter {db_path}"
        subprocess.run(command, shell=True, check=True)
        print("Database initialized.")


def load_data_from_db():
    """Load and merge data from SQLite database."""
    sql = """
    SELECT placements.id, holes.x, holes.y
    FROM holes
    INNER JOIN placements ON placements.hole_id = holes.id
    WHERE placements.layout_id = ?
    """

    with sqlite3.connect(DB_PATH) as conn:
        holes_df = pd.read_sql_query(sql, conn, params=(LAYOUT_ID,))
        routes_df = pd.read_sql_query("SELECT * FROM climbs", conn)
        routes_grade_df = pd.read_sql_query("SELECT * FROM climb_stats", conn)
        grade_comparision_df = pd.read_sql_query(
            "SELECT * FROM difficulty_grades", conn
        )

    return holes_df, routes_df, routes_grade_df, grade_comparision_df


def preprocess_data(routes_df, routes_grade_df, min_ascents=MIN_ASCENTS):
    """Filter and merge data based on ascents and layout."""
    routes_grade_df = routes_grade_df[
        routes_grade_df["ascensionist_count"] >= min_ascents
    ]
    routes_l1 = routes_df[(routes_df["layout_id"] == 1) & (routes_df["hsm"] == 1)][
        ["uuid", "frames"]
    ]
    # routes_l1 = routes_df[routes_df['layout_id'] == 1][['uuid', 'frames']]
    diffs = routes_grade_df[["climb_uuid", "angle", "difficulty_average"]]
    diffs = diffs.copy()
    diffs.rename(columns={"climb_uuid": "uuid"}, inplace=True)

    routes_l1 = routes_l1[~routes_l1["frames"].str.contains("x", na=False)]
    routes_l1 = routes_l1.merge(diffs, on="uuid", how="inner")

    return routes_l1


def get_features(routes_l1, holes_df, size=(177, 185)):
    """Generate feature matrix from route frames and hole positions."""
    # Initialize matrices to store route features and occupancy grid
    num_routes = len(routes_l1["frames"])
    routes_matrix = np.zeros((num_routes, *size))
    routes_occupied = np.zeros(size)

    # Preprocess hole positions for efficient lookup
    id_to_position = {row.id: (row.x + 20, row.y) for _, row in holes_df.iterrows()}

    for i, route_str in enumerate(routes_l1["frames"]):
        # Clean and split route frame string into individual holds
        route_holds = route_str.replace(",", "").replace('"', "").split("p")
        route_holds = [hold for hold in route_holds if hold]  # Remove empty strings

        for hold_str in route_holds:
            try:
                # Split hold string into position and type, convert to integers
                pos, hold_type = map(int, hold_str.split("r"))
                if pos in id_to_position:  # Check if position is valid
                    x, y = id_to_position[pos]
                    # Map hold type to route matrix, adjust for indexing from bottom left
                    routes_matrix[i, size[0] - y, x] = hold_type
                    routes_occupied[size[0] - y, x] = 1
            except ValueError as e:
                print(f"Error parsing hold string '{hold_str}' in route {i}: {e}")

    # Identify non-empty rows and columns to reduce matrix size
    non_empty_rows = np.any(routes_occupied, axis=1)
    non_empty_cols = np.any(routes_occupied, axis=0)

    # Create a placeholder matrix for non-empty regions
    placeholder_matrix = np.zeros(
        (num_routes, non_empty_rows.sum(), non_empty_cols.sum())
    )

    # Fill placeholder matrix with non-empty route data
    for i, route in enumerate(routes_matrix):
        placeholder_matrix[i] = route[non_empty_rows][:, non_empty_cols]

    return placeholder_matrix


class SimpleMLPRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 516)
        self.fc2 = nn.Linear(516, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class RoutesDataset(Dataset):
    def __init__(self, features, labels, angle):
        self.features = features.reshape(features.shape[0], -1)
        self.labels = labels
        self.angle = angle.reshape(-1, 1)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(
            np.hstack([self.features, self.angle])
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return torch.tensor(self.features[index], dtype=torch.double), torch.tensor(
            self.labels[index], dtype=torch.double
        )

def main():
    """Main training function."""
    config = TrainingConfig()
    trainer = ModelTrainer(config)
    
    # Load and prepare data
    train_loader, val_loader = prepare_data(config)  # This function needs to be implemented
    
    # Initialize model
    model = initialize_model()  # This function needs to be implemented
    
    # Train model
    metrics = trainer.train(model, train_loader, val_loader)
    
    # Plot training history
    trainer.plot_training_history(metrics)

if __name__ == "__main__":
    main()
