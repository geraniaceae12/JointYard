# analysis/vae/train.py
import torch
import torch.optim as optim
import gc
import numpy as np
import datetime
import os
import optuna

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .vae_model import DeepVAE, VanillaVAE
from .vae_utils import load_config, load_data, check_gpu, get_optimizer
from .vae_train import train_vae

def vae_run(config_path, data = None):
    """
    Function to run the Variational Autoencoder (VAE) model with hyperparameter optimization using Optuna.
    The input data can be provided in one of two ways:
        1) Specify the 'data_path' in the configuration file (config_path) 
        or 
        2) Pass the data directly as a NumPy array.
    
    Args:
        config_path (str): Path to the configuration file.
        data (numpy.ndarray, optional): Pre-loaded data in NumPy array format. 
                                        If None, data will be loaded from config file.
    
    Returns:
        trained_model: The final trained VAE model.
        device: Device (GPU/CPU) used for training.
    """
    # Load configuration
    config = load_config(config_path)
    vae_config = config["vae"]
    
    model_type = vae_config['vaemodel_type']  # 'deepvae' or 'vanillavae'
    data_path = vae_config.get('data_path', None)
    patience = vae_config['patience']
    optuna_checkpoint = vae_config['optuna_checkpoint']
    
    model_save_dir = os.path.join(config['info']['save_dir'],model_type,"logs/") + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(model_save_dir, exist_ok=True) 

    # Check device
    device = check_gpu()

    # Load and preprocess the data
    if data_path:
        data = load_data(data_path)
    elif data is not None:
        data = torch.from_numpy(data).float()
    else:
        print("no data is uploaded before starting the vae")

    if vae_config["scaler"]:
        data = data.T 
        scaler = StandardScaler()  # 또는 MinMaxScaler()
        data = scaler.fit_transform(data)  # 스케일링 적용, NumPy 배열 형태
        data = torch.from_numpy(data).float().to(device)
        print("Scaling before VAE...")
    else:
        data = data.T.to(device) # (17546*5200)

    train_data, validation_data = train_test_split(data, test_size = vae_config.get('test_split', 0.2) , random_state=42)
    
    # Define the Optuna study with a Pruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,   # Perform at least 5 trials before considering pruning
        n_warmup_steps=50,   # Allow the first 100 steps (epochs) of each trial before pruning
        interval_steps=5      # Check for pruning every 5 steps (epochs)
    )
    study = optuna.create_study(direction='minimize', pruner=pruner)

    # Define the objective function for Optuna
    def objective(trial):
        nonlocal optuna_checkpoint

        # Trial-specific log directory
        trial_log_dir = os.path.join(model_save_dir, f"trial_{trial.number}") # trial number starts with '0'
        os.makedirs(trial_log_dir, exist_ok=True)

        if optuna_checkpoint and os.path.exists(optuna_checkpoint):
            # Load optuna checkpoint
            checkpoint = torch.load(optuna_checkpoint)

            # 모델 초기화 (체크포인트 하이퍼파라미터 기반)
            if model_type == 'deepvae':
                model = DeepVAE(latent_dim=checkpoint['latent_dim'], feature_dim=train_data.shape[1], hidden_dim=checkpoint['hidden_dim']).to(device)
            elif model_type == 'vanillavae':
                model = VanillaVAE(latent_dim=checkpoint['latent_dim'], feature_dim=train_data.shape[1], hidden_dim=checkpoint['latent_dim']).to(device)
            else:
                raise ValueError("Invalid VAE type in config")
            
            optimizer = get_optimizer(model.parameters(), checkpoint['learning_rate'])
            
            # 체크포인트에서 모델 및 옵티마이저 상태 복원
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            epoch = checkpoint['epoch']         
            print(f"Resumed training from optuna checkpoint: epoch {epoch} with latent_dim={checkpoint['latent_dim']} and hidden_dim={checkpoint['hidden_dim']}")
            
            optuna_checkpoint = None
            
            epochs = trial.suggest_int('epochs', vae_config['epochs_range'][0], vae_config['epochs_range'][1])
            # Train model and evaluate
            try:
                trained_model, validation_loss = train_vae(
                    model, train_data, validation_data, optimizer, epochs, trial_log_dir, device, 
                    batch_size=checkpoint['batch_size'], patience=patience, latent_dim=checkpoint['latent_dim'], hidden_dim=checkpoint['hidden_dim'],
                    learning_rate=checkpoint['learning_rate'], trial=trial, 
                    start_epoch= checkpoint['epoch'], best_loss = checkpoint['best_loss'], no_improvement_count = checkpoint['no_improvement_count']
                )
            except optuna.exceptions.TrialPruned:
                raise  # Let Optuna handle the pruned trial
            
        else:    
            hidden_dim = trial.suggest_int('hidden_dim', vae_config['hidden_dim_range'][0], vae_config['hidden_dim_range'][1])
            latent_dim = trial.suggest_int('latent_dim', vae_config['latent_dim_range'][0], vae_config['latent_dim_range'][1])
            batch_size = trial.suggest_categorical('batch_size', vae_config['batch_size_options'])
            learning_rate = trial.suggest_float('learning_rate', vae_config['learning_rate_range'][0], vae_config['learning_rate_range'][1], log=True)
            epochs = trial.suggest_int('epochs', vae_config['epochs_range'][0], vae_config['epochs_range'][1])

            # Initialize Model
            if model_type == 'deepvae':
                model = DeepVAE(latent_dim=latent_dim, feature_dim=train_data.shape[1], hidden_dim=hidden_dim).to(device)
            elif model_type == 'vanillavae':
                model = VanillaVAE(latent_dim=latent_dim, feature_dim=train_data.shape[1], hidden_dim=hidden_dim).to(device)
            else:
                raise ValueError("Invalid VAE type in config")

            # Initialize Optimizer( type: adam or sgd )
            optimizer = get_optimizer(model.parameters(), learning_rate)
            
            # Train model and evaluate
            try:
                trained_model, validation_loss = train_vae(
                    model, train_data, validation_data, optimizer, epochs, trial_log_dir, device, batch_size=batch_size,
                    patience=patience, latent_dim=latent_dim, hidden_dim=hidden_dim, learning_rate=learning_rate, trial=trial
                )
            except optuna.exceptions.TrialPruned:
                raise  # Let Optuna handle the pruned trial

        # GPU memory release in the end of each trial
        del model, optimizer, trained_model
        gc.collect()
        torch.cuda.empty_cache()

        # Return validation loss for Optuna to minimize
        return validation_loss

    # Run Optuna optimization
    study.optimize(objective, n_trials=vae_config['optuna_n_trials'])
    
    # Print best parameters
    best_params = study.best_params
    print(f"\nStart with Best params: {best_params}")

    # Save the best model
    best_save_dir = os.path.join(model_save_dir,"bestcombi")
    os.makedirs(best_save_dir, exist_ok = True)
    
    # Initialize model with best hyperparameters
    if model_type == 'deepvae':
        model = DeepVAE(latent_dim=best_params['latent_dim'], feature_dim=train_data.shape[1], hidden_dim=best_params['hidden_dim']).to(device)
    elif model_type == 'vanillavae':
        model = VanillaVAE(latent_dim=best_params['latent_dim'], feature_dim=train_data.shape[1]).to(device)

    # Initialize optimizer with the best learning rate
    optimizer = get_optimizer(model.parameters(), best_params['learning_rate'])

    # Train with best hyperparameters
    trained_model, val_loss = train_vae(model, train_data, validation_data, optimizer, 
                              best_params['epochs'], best_save_dir, device, 
                              batch_size=best_params['batch_size'], patience=patience, 
                              latent_dim=best_params['latent_dim'], hidden_dim=best_params['hidden_dim']) # 마지막에 trial안넣나?
    
    # Save the final trained model
    model_path = os.path.join(best_save_dir, "final_model.pt")
    torch.save(trained_model.state_dict(), model_path)

    print(f"Save complete of final Best model: validation loss is {val_loss}")

    # GPU memory release after training
    del model, optimizer, train_data, validation_data
    gc.collect()
    torch.cuda.empty_cache()

    return trained_model, device


