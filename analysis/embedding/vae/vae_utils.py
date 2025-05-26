import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
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

def find_best_k_fixed(data, k_range=range(2, 30), save_dir = None):
    # Define path for saving/loading k_fixed value
    if save_dir is None:
        save_dir = './'
    os.makedirs(save_dir, exist_ok=True)

    k_file_path = os.path.join(save_dir, 'k_fixed.json')
    elbow_plot_path = os.path.join(save_dir, 'elbow_plot.png')

    # If result already exists, load and return
    if os.path.exists(k_file_path):
        with open(k_file_path, 'r') as f:
            k_fixed = json.load(f)['k_fixed']
        print(f"[INFO: before VAE] Loaded existing k_fixed: {k_fixed}")
        return k_fixed

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # Subsample if data too large
    if data.shape[0] > 20000:
        np.random.seed(42)
        idx = np.random.choice(data.shape[0], size=20000, replace=False)
        data_subset = data[idx]
    else:
        data_subset = data

    # Run elbow method
    inertia = []
    for k in k_range:
        kmeans = KMeans(init='k-means++', n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(data_subset)
        inertia.append(kmeans.inertia_)

    # Use the elbow method: select k where the second derivative is minimized
    deltas = np.diff(inertia)
    second_deltas = np.diff(deltas)
    k_fixed = k_range[np.argmin(second_deltas) + 1]

    # Plot and save
    plt.figure()
    plt.plot(list(k_range), inertia, marker='o')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title(f'K Elbow Method: {k_fixed}')
    plt.savefig(elbow_plot_path)
    plt.close()

    # Save k_fixed value
    with open(k_file_path, 'w') as f:
        json.dump({'k_fixed': int(k_fixed)}, f)

    print(f"[INFO: before VAE] Saved k_fixed: {k_fixed} to {k_file_path}")
    return k_fixed