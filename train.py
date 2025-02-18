import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas explicitly

# Import utility functions from a separate module (ensure these are well-defined)
from utils import load_data_from_db, preprocess_data, get_features, RoutesDataset, SimpleMLPRegression, initialize_database_if_needed

# Constants (make these configurable via command-line arguments or config file)
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
MIN_ASCENTS = 2
RANDOM_SEED = 42  # Add a random seed for reproducibility

# Hugging Face related imports (add these)
from transformers import Trainer, TrainingArguments

def load_and_preprocess_data():
    """Load and preprocess data."""
    print("Loading data")
    holes_df, routes_df, routes_grade_df, _ = load_data_from_db()  # Assuming this returns pandas DataFrames
    print("Data Loaded")

    print("Preprocessing data")
    routes_l1 = preprocess_data(routes_df, routes_grade_df, min_ascents=MIN_ASCENTS)
    print(f"Data preprocessed, {len(routes_l1)} boulder problems")

    print("Extracting features")
    routes = get_features(routes_l1, holes_df)  # Assuming this returns a NumPy array or PyTorch tensor
    print(f"Features extracted. Got {routes.shape} features")

    return routes, routes_l1

def create_datasets_and_loaders(routes, routes_l1):
    """Create datasets and dataloaders."""
    print("Creating dataset")
    full_dataset = RoutesDataset(routes, routes_l1['difficulty_average'].values, routes_l1['angle'].values)

    # Splitting the dataset with a random seed
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(RANDOM_SEED))

    print("Creating dataloaders")
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset # Return the datasets as well

def train_and_evaluate_model(train_loader, test_loader, train_dataset, test_dataset, routes): # Added datasets and routes
    """Train and evaluate the model using Hugging Face Trainer."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLPRegression(routes.shape[1]).to(device).double() # Use routes shape here

    # Training arguments for Hugging Face Trainer
    training_args = TrainingArguments(
        output_dir="./results",          # Output directory for checkpoints and logs
        num_train_epochs=NUM_EPOCHS,      # Number of training epochs
        per_device_train_batch_size=BATCH_SIZE,  # Batch size per device during training
        per_device_eval_batch_size=BATCH_SIZE,   # Batch size for evaluation
        learning_rate=LEARNING_RATE,      # Learning rate
        weight_decay=0.01,                # Weight decay (L2 regularization) - optional
        evaluation_strategy="epoch",      # Evaluation strategy
        save_strategy="epoch",            # Save checkpoints every epoch
        load_best_model_at_end=True,      # Load the best model after training
        metric_for_best_model="eval_loss", # Metric to determine best model
        seed=RANDOM_SEED,                 # Set random seed for reproducibility
        # Add more arguments as needed (e.g., logging, reporting)

    )

    # Define a custom compute_metrics function (highly recommended)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Calculate metrics using scikit-learn or other libraries
        # Example using MSE:
        mse = ((predictions - labels)**2).mean()
        return {"mse": mse} # return a dictionary of metrics

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics, # Pass the compute_metrics function
    )

    # Train the model
    trainer.train()

    # Save the best model (already done by the Trainer)
    # trainer.save_model("best_model")  # No need to call save_model manually

    # Evaluate the model (can also be done with trainer.evaluate())
    metrics = trainer.evaluate()
    print(metrics)

    # Plot losses (you might need to extract them from the trainer logs if you want to plot them manually)
    # trainer.plot_losses() will plot the losses if you have configured logging correctly.

    return trainer # return the trainer object


# ... (save_model and plot_losses can be removed or adapted if you use Trainer's built-in functionalities)

if __name__ == "__main__":
    initialize_database_if_needed()

    routes, routes_l1 = load_and_preprocess_data()
    train_loader, test_loader, train_dataset, test_dataset = create_datasets_and_loaders(routes, routes_l1)
    trainer = train_and_evaluate_model(train_loader, test_loader, train_dataset, test_dataset, routes)

    # Example of how to access the best model
    best_model = trainer.model
    # You can now use best_model for inference or save it in a different format (e.g., ONNX)

    # Example: Save the model in Hugging Face format
    best_model.save_pretrained("./my_climbing_model") # Saves the model and tokenizer (if applicable)

    # You can now push your model to the Hugging Face Hub:
    # from huggingface_hub import push_to_hub
    # best_model.push_to_hub("my_climbing_model") # Requires you to be logged in to Hugging Face
