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
from transformers import (
    PreTrainedModel, 
    PretrainedConfig
)

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


import numpy as np
import scipy.stats as stats

def calculate_route_complexity(routes_matrix, route_stats):
    """
    Calculate route complexity based on geometric and statistical features.
    
    Args:
        routes_matrix (np.ndarray): 2D route matrix with hole placements
        route_stats (np.ndarray): Statistical features about the route
    
    Returns:
        np.ndarray: Complexity score for each route
    """
    # Spatial randomness of hole placements
    def spatial_randomness(route_matrix):
        # Calculate the standard deviation of hole coordinates
        nonzero_coords = np.argwhere(route_matrix > 0)
        if len(nonzero_coords) == 0:
            return 0
        
        x_std = np.std(nonzero_coords[:, 1])
        y_std = np.std(nonzero_coords[:, 0])
        
        # Higher standard deviation indicates more random placement
        return np.mean([x_std, y_std]) / route_matrix.shape[1]
    
    # Hole distribution entropy
    def hole_distribution_entropy(route_matrix):
        # Flatten the matrix and calculate entropy of hole placements
        flattened = route_matrix.flatten()
        unique, counts = np.unique(flattened, return_counts=True)
        probabilities = counts / len(flattened)
        return stats.entropy(probabilities)
    
    # Complexity calculation for each route
    complexities = np.zeros(len(routes_matrix))
    for i, route_matrix in enumerate(routes_matrix):
        spatial_random = spatial_randomness(route_matrix)
        entropy = hole_distribution_entropy(route_matrix)
        
        # Combine complexity metrics
        complexities[i] = np.mean([spatial_random, entropy])
    
    return complexities

def calculate_adjusted_difficulty(original_difficulty, ascent_count, route_complexity):
    """
    Calculate an adjusted difficulty that accounts for ascent frequency and route complexity.
    
    Args:
        original_difficulty (float): Initial route difficulty
        ascent_count (int): Number of times the route has been climbed
        route_complexity (float): Complexity score of the route
    
    Returns:
        float: Adjusted difficulty score
    """
    # Normalize ascent count (log transform to reduce extreme values)
    normalized_ascents = np.log1p(ascent_count)
    
    # Adjusted difficulty reduces original difficulty based on ascent frequency
    # More ascents → lower adjusted difficulty
    # Route complexity increases the difficulty
    adjusted_difficulty = original_difficulty * (1 / (1 + normalized_ascents)) * (1 + route_complexity)
    
    return adjusted_difficulty

def stratified_route_sampling(routes_l1, max_routes=50000, n_strata=10):
    """
    Perform stratified sampling of routes based on difficulty
    
    Args:
        routes_l1 (pd.DataFrame): Routes dataframe
        max_routes (int): Maximum total number of routes to sample
        n_strata (int): Number of difficulty strata to create
    
    Returns:
        pd.DataFrame: Stratified sample of routes
    """
    # Create difficulty strata
    routes_l1['difficulty_stratum'] = pd.qcut(
        routes_l1['difficulty_average'], 
        q=n_strata, 
        labels=False
    )
    
    # Calculate routes per stratum
    routes_per_stratum = max_routes // n_strata
    
    # Stratified sampling
    sampled_routes = routes_l1.groupby('difficulty_stratum').apply(
        lambda x: x.sample(
            n=min(routes_per_stratum, len(x)), 
            random_state=42
        )
    ).reset_index(drop=True)
    
    # If we haven't reached max_routes, sample additional routes
    if len(sampled_routes) < max_routes:
        remaining_routes = routes_l1[~routes_l1.index.isin(sampled_routes.index)]
        additional_routes = remaining_routes.sample(
            n=max_routes - len(sampled_routes), 
            random_state=42
        )
        sampled_routes = pd.concat([sampled_routes, additional_routes]).reset_index(drop=True)
    
    return sampled_routes

def get_features(routes_l1, holes_df, routes_grade_df, max_routes=50000):
    """
    Extract features for boulder routes with memory-efficient processing and stratified sampling.
    
    Args:
        routes_l1 (pd.DataFrame): Preprocessed routes dataframe
        holes_df (pd.DataFrame): Holes dataframe
        routes_grade_df (pd.DataFrame): Route grade statistics dataframe
        max_routes (int, optional): Maximum number of routes to process
    
    Returns:
        tuple: Feature matrix and route data
    """
    print("get_features() diagnostic:")
    
    # Print total routes before sampling
    print(f"Total routes before sampling: {len(routes_l1)}")
    
    # Merge ascent count information
    routes_l1 = routes_l1.merge(
        routes_grade_df[['climb_uuid', 'ascensionist_count']], 
        left_on='uuid', 
        right_on='climb_uuid', 
        how='left'
    )
    
    # Perform stratified sampling
    routes_l1 = stratified_route_sampling(routes_l1, max_routes)
    
    print(f"Routes after stratified sampling: {len(routes_l1)}")
    
    # Rest of the function remains the same as in the previous implementation
    # Preallocate memory for routes matrix
    routes_matrix = np.zeros((len(routes_l1), 177, 185), dtype=np.float32)
    
    # Process routes in batches to manage memory
    for idx, row in routes_l1.iterrows():
        # Get route UUID and angle
        route_uuid = row['uuid']
        route_angle = row['angle']
        
        # Find hole placements for this route
        route_holes = holes_df[holes_df['id'] == route_uuid]
        
        # Create route matrix with additional features
        for _, hole in route_holes.iterrows():
            x, y = int(hole['x']), int(hole['y'])
            if 0 <= x < 185 and 0 <= y < 177:
                # Encode hole position with angle information
                routes_matrix[idx, y, x] = route_angle / 90.0  # Normalize angle
    
    print(f"Original routes_matrix shape: {routes_matrix.shape}")
    
    # Calculate route complexity
    route_complexity = calculate_route_complexity(routes_matrix, routes_l1)
    
    # Calculate adjusted difficulty
    routes_l1['adjusted_difficulty'] = calculate_adjusted_difficulty(
        routes_l1['difficulty_average'].values, 
        routes_l1['ascensionist_count'].fillna(1).values,
        route_complexity
    )
    
    # Flatten routes matrix
    routes = routes_matrix.reshape(routes_matrix.shape[0], -1)
    print(f"Flattened routes shape: {routes.shape}")
    
    # Add statistical features
    route_stats = np.zeros((len(routes_l1), 7), dtype=np.float32)
    route_stats[:, 0] = routes_l1['angle'].values / 90.0  # Normalized angle
    route_stats[:, 1] = routes_l1['difficulty_average'].values / 50.0  # Normalized difficulty
    route_stats[:, 2] = routes_l1['adjusted_difficulty'].values / 50.0  # Normalized adjusted difficulty
    route_stats[:, 3] = np.log1p(routes_l1['ascensionist_count'].fillna(1).values)  # Log-transformed ascent count
    route_stats[:, 4] = np.sum(routes_matrix[..., :10], axis=(1, 2))  # Hole count in first 10 columns
    route_stats[:, 5] = np.sum(routes_matrix[..., -10:], axis=(1, 2))  # Hole count in last 10 columns
    route_stats[:, 6] = route_complexity  # Route complexity score
    
    # Combine flattened routes with statistical features
    combined_routes = np.hstack([routes, route_stats])
    
    print(f"Combined routes shape: {combined_routes.shape}")
    
    return combined_routes, routes_l1


class ClimbingModelConfig(PretrainedConfig):
    """
    Configuration class for the Climbing Route Difficulty Prediction model.
    
    Attributes:
        input_size (int): Size of the input features
        hidden_layers (list): Number and size of hidden layers
        dropout_rate (float): Dropout rate for regularization
    """
    model_type = "climbing_mlp_regression"
    
    def __init__(
        self, 
        input_size=32763,  # Updated to match new feature size 
        hidden_layers=[256, 128, 64], 
        dropout_rate=0.3, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate

class SimpleMLPRegression(PreTrainedModel):
    """
    Multi-Layer Perceptron Regression Model for Climbing Route Difficulty Prediction.
    
    Fully compatible with Hugging Face's PreTrainedModel and from_pretrained() method.
    """
    config_class = ClimbingModelConfig
    base_model_prefix = "climbing_mlp"
    
    def __init__(self, config):
        """
        Initialize the model with a configuration.
        
        Args:
            config (ClimbingModelConfig): Model configuration
        """
        super().__init__(config)
        
        # Input layer
        layers = [nn.Linear(config.input_size, config.hidden_layers[0])]
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config.dropout_rate))
        
        # Hidden layers
        for i in range(1, len(config.hidden_layers)):
            layers.append(nn.Linear(config.hidden_layers[i-1], config.hidden_layers[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(config.hidden_layers[-1], 1))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self.init_weights()
    
    def forward(
        self, 
        input_ids=None, 
        labels=None, 
        **kwargs
    ):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input features
            labels (torch.Tensor, optional): Ground truth labels
        
        Returns:
            dict: Model outputs including logits and optional loss
        """
        # Use input_ids or fallback to x in kwargs
        x = input_ids if input_ids is not None else kwargs.get('x')
        
        # Predict
        logits = self.model(x).squeeze()
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)
        
        return {
            'logits': logits,
            'loss': loss
        }
    
    def init_weights(self):
        """Initialize model weights."""
        for module in self.model:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load a pre-trained model from Hugging Face Hub or local directory.
        
        Args:
            pretrained_model_name_or_path (str): Path to pretrained model or model identifier
            *model_args: Additional positional arguments
            **kwargs: Additional keyword arguments
        
        Returns:
            SimpleMLPRegression: Loaded and initialized model
        """
        # Load configuration
        config = ClimbingModelConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Initialize model with loaded configuration
        model = cls(config)
        
        # Try to load state dictionary
        try:
            # Attempt to load from the specified path
            state_dict = torch.load(
                os.path.join(pretrained_model_name_or_path, "pytorch_model.bin"),
                map_location="cpu"
            )
            model.load_state_dict(state_dict)
        except (FileNotFoundError, IOError):
            print(f"Could not find model weights at {pretrained_model_name_or_path}. Using randomly initialized weights.")
        
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
