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

# Constants
DB_PATH = 'kilter_data.db'
LAYOUT_ID = 1
MIN_ASCENTS = 1
SIZE = (177, 185)

def initialize_database_if_needed(db_path=DB_PATH):
    # ... (no changes needed)

def load_data_from_db():
    # ... (no changes needed)

def preprocess_data(routes_df, routes_grade_df, min_ascents=MIN_ASCENTS):
    # ... (no changes needed)

def get_features(routes_l1, holes_df, size=(177, 185)):
    """Generate feature matrix from route frames and hole positions."""
    num_routes = len(routes_l1['frames'])
    
    # Initialize an empty list to store the route matrices
    route_matrices = []

    id_to_position = {row.id: (row.x + 20, row.y) for _, row in holes_df.iterrows()}

    for i, route_str in enumerate(routes_l1['frames']):
        route_matrix = np.zeros(size) # Initialize matrix for CURRENT route
        route_holds = route_str.replace(',', '').replace('"', '').split('p')
        route_holds = [hold for hold in route_holds if hold]

        for hold_str in route_holds:
            try:
                pos, hold_type = map(int, hold_str.split('r'))
                if pos in id_to_position:
                    x, y = id_to_position[pos]
                    route_matrix[size[0] - y, x] = hold_type # Fill the matrix for this route
            except ValueError as e:
                print(f"Error parsing hold string '{hold_str}' in route {i}: {e}")

        route_matrices.append(route_matrix) # Add the route matrix to the list

    # Convert the list of matrices to a NumPy array
    routes_matrix = np.array(route_matrices)

    # Identify non-empty rows and columns (now done on the whole dataset)
    routes_occupied = np.sum(routes_matrix, axis=0) # Sum across all routes
    non_empty_rows = np.any(routes_occupied, axis=1)
    non_empty_cols = np.any(routes_occupied, axis=0)

    # Crop each route matrix based on occupied holes
    cropped_routes = []
    for route in routes_matrix:
        cropped_route = route[non_empty_rows][:, non_empty_cols]
        cropped_routes.append(cropped_route)
    
    # Convert cropped routes to numpy array
    placeholder_matrix = np.array(cropped_routes)


    return placeholder_matrix


class SimpleMLPRegression(nn.Module):
    # ... (no changes needed)

class RoutesDataset(Dataset):
    def __init__(self, features, labels, angle):

        self.features = features.reshape(features.shape[0], -1) # Flatten each image/route
        self.labels = labels
        self.angle = angle.reshape(-1, 1)
        self.scaler = StandardScaler()

        # Concatenate features and angle before scaling
        combined_data = np.hstack([self.features, self.angle])
        self.features = self.scaler.fit_transform(combined_data)[:, :-1] # Scale, then remove the angle column

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return torch.tensor(self.features[index], dtype=torch.double), torch.tensor(self.labels[index], dtype=torch.double)
