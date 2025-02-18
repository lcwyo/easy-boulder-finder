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
from transformers import PreTrainedModel, PretrainedConfig

# Constants
DB_PATH = 'kilter_data.db'
LAYOUT_ID = 1
MIN_ASCENTS = 1
SIZE = (177, 185)

def initialize_database_if_needed(db_path=DB_PATH):
    """Check if the database exists and initialize it if it doesn't."""
    if not os.path.exists(db_path):
        print(f"Database at {db_path} not found. Initializing database...")
        # Run the command to initialize the database
        command = f"python3 -m boardlib database kilter {db_path}"
        subprocess.run(command, shell=True, check=True)
        print("Database initialized.")


def load_data_from_db(db_path=DB_PATH):
    """
    Load and merge data from SQLite database with comprehensive error handling.
    
    Args:
        db_path (str, optional): Path to the SQLite database. Defaults to DB_PATH.
    
    Returns:
        tuple: DataFrames for holes, routes, route grades, and grade comparison
    """
    # Comprehensive database initialization and validation
    def validate_database(path):
        """Validate database file and connection."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Database file not found at {path}")

        try:
            with sqlite3.connect(path) as conn:
                cursor = conn.cursor()

                # Check database file integrity
                try:
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                except sqlite3.DatabaseError:
                    raise ValueError("Database appears to be corrupted or unreadable")

                # Print existing tables for diagnostics
                print("Existing tables:", [table[0] for table in tables])

                # Check specific required tables
                required_tables = ['holes', 'placements', 'climbs', 'climb_stats', 'difficulty_grades']
                missing_tables = [table for table in required_tables if table not in [t[0] for t in tables]]

                if missing_tables:
                    raise ValueError(f"Missing required tables: {missing_tables}")

                return tables
        except sqlite3.Error as e:
            raise RuntimeError(f"SQLite connection error: {e}")

    # Ensure database is initialized
    initialize_database_if_needed(db_path)

    # Validate database before proceeding
    validate_database(db_path)

    # Detailed SQL queries with error handling
    try:
        with sqlite3.connect(db_path) as conn:
            # Diagnostic queries with error handling
            queries = {
                'holes': """
                    SELECT placements.id, holes.x, holes.y
                    FROM holes
                    INNER JOIN placements ON placements.hole_id = holes.id
                    WHERE placements.layout_id = ?
                """,
                'climbs': "SELECT * FROM climbs",
                'climb_stats': "SELECT * FROM climb_stats",
                'difficulty_grades': "SELECT * FROM difficulty_grades"
            }

            results = {}
            for name, query in queries.items():
                try:
                    # Use different parameter handling based on query
                    if name == 'holes':
                        df = pd.read_sql_query(query, conn, params=(LAYOUT_ID,))
                    else:
                        df = pd.read_sql_query(query, conn)

                    results[name] = df
                    print(f"{name.capitalize()} query returned {len(df)} rows")

                except Exception as query_err:
                    print(f"Error in {name} query: {query_err}")
                    raise

    except sqlite3.Error as conn_err:
        print(f"Database connection error: {conn_err}")
        raise

    # Unpack results with error checking
    holes_df = results.get('holes')
    routes_df = results.get('climbs')
    routes_grade_df = results.get('climb_stats')
    grade_comparision_df = results.get('difficulty_grades')

    if any(df is None or df.empty for df in [holes_df, routes_df, routes_grade_df, grade_comparision_df]):
        raise ValueError("Failed to retrieve all required dataframes or dataframes are empty")

    return holes_df, routes_df, routes_grade_df, grade_comparision_df


def preprocess_data(routes_df, routes_grade_df, min_ascents = MIN_ASCENTS):
    """Filter and merge data based on ascents and layout."""
    routes_grade_df = routes_grade_df[routes_grade_df['ascensionist_count'] >= min_ascents]
    routes_l1 = routes_df[routes_df['layout_id'] == 1][['uuid', 'frames']]
    diffs = routes_grade_df[['climb_uuid', 'angle', 'difficulty_average']]
    diffs = diffs.copy()
    diffs.rename(columns={"climb_uuid": "uuid"}, inplace=True)

    routes_l1 = routes_l1[~routes_l1['frames'].str.contains('x', na=False)]
    routes_l1 = routes_l1.merge(diffs, on='uuid', how='inner')

    return routes_l1


def get_features(routes_l1, holes_df, max_routes=50000):
    """
    Extract features for boulder routes with memory-efficient processing.
    
    Args:
        routes_l1 (pd.DataFrame): Preprocessed routes dataframe
        holes_df (pd.DataFrame): Holes dataframe
        max_routes (int, optional): Maximum number of routes to process
    
    Returns:
        np.ndarray: Feature matrix for routes
    """
    print("get_features() diagnostic:")
    
    # Limit number of routes if needed
    if len(routes_l1) > max_routes:
        print(f"Warning: Too many routes ({len(routes_l1)}). Limiting to {max_routes}")
        routes_l1 = routes_l1.sample(n=max_routes, random_state=42)
    
    # Preallocate memory for routes matrix
    routes_matrix = np.zeros((len(routes_l1), 177, 185), dtype=np.float32)
    
    # Process routes in batches to manage memory
    for idx, row in routes_l1.iterrows():
        # Get route UUID
        route_uuid = row['uuid']
        
        # Find hole placements for this route
        route_holes = holes_df[holes_df['id'] == route_uuid]
        
        # Create route matrix
        for _, hole in route_holes.iterrows():
            x, y = int(hole['x']), int(hole['y'])
            if 0 <= x < 185 and 0 <= y < 177:
                routes_matrix[idx, y, x] = 1.0
    
    print(f"Original routes_matrix shape: {routes_matrix.shape}")
    
    # Flatten routes matrix
    routes = routes_matrix.reshape(routes_matrix.shape[0], -1)
    print(f"Flattened routes shape: {routes.shape}")
    
    return routes


class ClimbingModelConfig(PretrainedConfig):
    """Configuration class for the Climbing Regression Model."""
    model_type = "climbing_regression"

    def __init__(
        self,
        input_size=38,  # Default input size, can be overridden
        hidden_size1=516,
        hidden_size2=256,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2


class SimpleMLPRegression(PreTrainedModel):
    """A simple MLP Regression model for climbing route difficulty prediction."""
    config_class = ClimbingModelConfig

    def __init__(self, config):
        super().__init__(config)

        # Model architecture
        self.fc1 = nn.Linear(config.input_size, config.hidden_size1)
        self.fc2 = nn.Linear(config.hidden_size1, config.hidden_size2)
        self.fc3 = nn.Linear(config.hidden_size2, 1)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids=None,
        labels=None,
        **kwargs
    ):
        # If input_ids is None, try to get x from kwargs
        x = input_ids if input_ids is not None else kwargs.get('x')

        if x is None:
            raise ValueError("No input tensor provided")

        # Forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        predictions = self.fc3(x)

        # Loss computation
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions.squeeze(), labels)

        return {
            "loss": loss,
            "logits": predictions
        }

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a pre-trained model from a directory or Hugging Face Hub.
        
        Args:
            pretrained_model_name_or_path (str): Path to pretrained model or model identifier from huggingface.co/models
        
        Returns:
            SimpleMLPRegression: Loaded model
        """
        config = ClimbingModelConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)

        # Load state dict
        state_dict = torch.load(
            os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"),
            map_location="cpu"
        )
        model.load_state_dict(state_dict)

        return model


class RoutesDataset(Dataset):
    def __init__(self, features, labels, angle):
        print("RoutesDataset initialization diagnostic:")
        print("Features shape:", features.shape)
        print("Labels shape:", labels.shape)
        print("Angle shape:", angle.shape)

        # Ensure features are 2D
        self.features = features.reshape(features.shape[0], -1).astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.angle = angle.reshape(-1, 1).astype(np.float32)

        print("After reshaping:")
        print("Features shape:", self.features.shape)

        self.scaler = StandardScaler()

        # Combine features with angle before scaling
        combined_data = np.hstack([self.features, self.angle])
        scaled_data = self.scaler.fit_transform(combined_data)

        # Separate scaled features and angle
        self.features = scaled_data[:, :-1]
        self.scaled_angle = scaled_data[:, -1:]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        # Return features and label as a dictionary
        return {
            "input_ids": torch.tensor(self.features[index], dtype=torch.float),
            "labels": torch.tensor(self.labels[index], dtype=torch.float)
        }
