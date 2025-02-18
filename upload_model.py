import os
import sys
import torch
from huggingface_hub import HfApi, ModelHubMixin

from utils import SimpleMLPRegression, ClimbingModelConfig, load_data_from_db, preprocess_data, get_features

def upload_model_to_hub(model_path='./my_climbing_model.pth', 
                        repo_name='lcwyo/easy-boulder-finder', 
                        commit_message='Initial model upload',
                        hf_token=None):
    """
    Upload the trained climbing route difficulty prediction model to Hugging Face Hub.
    
    Args:
        model_path (str): Path to the saved model state dict
        repo_name (str): Hugging Face repository name
        commit_message (str): Commit message for the upload
        hf_token (str, optional): Hugging Face authentication token
    """
    # Check for Hugging Face token
    if not hf_token:
        hf_token = os.environ.get('HF_TOKEN')
    
    if not hf_token:
        print("Error: No Hugging Face token provided. Please set HF_TOKEN environment variable or pass token as argument.")
        sys.exit(1)

    # Load data to get input size
    holes_df, routes_df, routes_grade_df, _ = load_data_from_db()
    routes_l1 = preprocess_data(routes_df, routes_grade_df, min_ascents=1)
    routes = get_features(routes_l1, holes_df)
    
    # Create configuration
    config = ClimbingModelConfig(input_size=routes.shape[1])
    
    # Initialize model
    model = SimpleMLPRegression(config)
    
    # Load state dict
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    
    # Initialize Hugging Face API
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        api.create_repo(repo_id=repo_name, token=hf_token, exist_ok=True)
    except Exception as e:
        print(f"Error creating repository: {e}")
        sys.exit(1)
    
    # Save model and config
    model.save_pretrained(repo_name, token=hf_token, push_to_hub=True, 
                           commit_message=commit_message)
    config.save_pretrained(repo_name, token=hf_token, push_to_hub=True)
    
    print(f"Model successfully uploaded to {repo_name}")
    print("You can view your model at: https://huggingface.co/" + repo_name)

if __name__ == "__main__":
    # Allow token to be passed as command-line argument
    import argparse
    
    parser = argparse.ArgumentParser(description='Upload model to Hugging Face Hub')
    parser.add_argument('--token', type=str, help='Hugging Face authentication token')
    args = parser.parse_args()
    
    upload_model_to_hub(hf_token=args.token)
