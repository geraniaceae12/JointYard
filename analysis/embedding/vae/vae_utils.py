import torch
import numpy as np
import json
from torch import nn, optim
from .vae_model import DeepVAE, VanillaVAE, DeepVAE2, DeepVAE3
from .vae_visualize import timeorder_visualize_latent_space_and_save

def check_gpu():
    # GPU가 사용 가능한지 확인
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("...Using GPU...")
    else:
        device = torch.device("cpu")
        print("...Using CPU...")
    return device

def load_data(file_path):
    return torch.from_numpy(np.load(file_path)).float()

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def load_hyperparameters(file_path):
    with open(file_path, 'r') as f:
        params = json.load(f)
        print()
    return params

def load_savedmodel(model_type, model_path, hyparam_path, feature_dim, device = None):
    # Load hyperparameters
    hyperparameters = load_hyperparameters(hyparam_path)
    
    latent_dim = hyperparameters.get('latent_dim', 4)
    hidden_dim = hyperparameters.get('hidden_dim', 472)
    
    if model_type == 'deepvae':
        model = DeepVAE(latent_dim=latent_dim, feature_dim=feature_dim, hidden_dim=hidden_dim)  # Initialize with the loaded parameters
    elif model_type == 'deepvae2':
        model = DeepVAE2(latent_dim=latent_dim, feature_dim=feature_dim, hidden_dim=hidden_dim)  # Initialize with the loaded parameters
    elif model_type == 'deepvae3':
        model = DeepVAE3(latent_dim=latent_dim, feature_dim=feature_dim, hidden_dim=hidden_dim)  # Initialize with the loaded parameters
    elif model_type == 'vanillavae':
        model = VanillaVAE(latent_dim=latent_dim, feature_dim=feature_dim, hidden_dim=hidden_dim)  # Initialize with the loaded parameters
    else:
        raise ValueError("Invalid model type")

    if device is None:
        device = check_gpu()
        
    #map_location = None if device.type == 'cuda' else torch.device('cpu')
    map_location = torch.device('cpu') if device.type == 'cpu' else None
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    model.to(device)
    model.eval()
    return model, device

def numpy_to_tensor(data):
    data = torch.from_numpy(data).float()
    data = data.T
    return data

def get_optimizer(parameters, learning_rate, optimizer_type='adam'):
    if optimizer_type == 'adam':
        return torch.optim.Adam(parameters, lr=learning_rate)
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(parameters, lr=learning_rate)
    else:
        raise ValueError("Unsupported optimizer type")

def evaluate_vae(model, data_path, log_dir, vae_config):
    # Load and preprocess the data
    device = check_gpu()
    data = torch.from_numpy(np.load(data_path)).float()
    data = data.T.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        reconstructed_data, mean, logvar = model(data)
    
    # Visualization
    time_data = np.arange(len(data))  # Example time data
    video_path = vae_config['video_path']
    timeorder_visualize_latent_space_and_save(mean.cpu().numpy(), time_data, vae_config['latent_dim'], video_path)
