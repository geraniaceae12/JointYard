import torch
import torch.nn as nn
import numpy as np
import os
import json
import optuna
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from .vae_model import DeepVAE, VanillaVAE
from .vae_utils import load_config, load_data, check_gpu, get_optimizer
from .vae_visualize import visualize_latent_space_and_save

def train_vae(model, train_data, validation_data, optimizer, epochs, model_save_dir, device, 
              patience=50, batch_size=32, beta = 1.0, latent_dim=2, hidden_dim=128, learning_rate=0.001, trial=None,
              start_epoch = None, best_loss = None, no_improvement_count = None):
    criterion = torch.nn.MSELoss()
    best_loss = float('inf') if best_loss is None else best_loss
    no_improvement_count = 0 if not no_improvement_count else no_improvement_count
    scaler = GradScaler() # Mixed Precision Traning을 위한 GradScaler 초기화

    writer = SummaryWriter(log_dir=model_save_dir)

    start_epoch = 0 if not start_epoch else start_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size].to(device)
            optimizer.zero_grad()

            # Mixed Precision Traning 시작
            with autocast():
                reconstructed, mean, logvar = model(batch)
                reconstruction_loss = criterion(reconstructed, batch)
                kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
                total_loss = reconstruction_loss + beta * kl_loss

            # Scale loss values and preceed with backpropagation and optimizer updates     
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item() * batch.size(0)

        train_loss /= len(train_data)
        writer.add_scalar('Loss/train', train_loss, epoch)

        if validation_data is not None:
            val_loss, reconstruction_loss, kl_loss = compute_validation_loss(model, validation_data, criterion, batch_size, device)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Loss/reconstruction', reconstruction_loss, epoch)
            writer.add_scalar('Loss/kl', kl_loss, epoch)

            # Prune the trial if necessary
            if trial is not None and trial.should_prune():
                raise optuna.exceptions.TrialPruned()  # Raise exception to stop the current trial

            if val_loss < best_loss:
                best_loss = val_loss

                model_info_dir = os.path.join(model_save_dir, "model_info")
                os.makedirs(model_info_dir, exist_ok=True)
                best_model_path = os.path.join(model_info_dir, f"best_model_latest.pt")
                #best_model_path = os.path.join(model_info_dir, f"best_model_epoch_{epoch+1}_loss_{best_loss:.4f}.pt")
                torch.save(model.state_dict(), best_model_path)

                # 하이퍼파라미터 저장
                best_hyperparams = {
                    'latent_dim': latent_dim,
                    'hidden_dim': hidden_dim,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'beta': beta
                }
                best_hyperparams_path = os.path.join(model_info_dir, "best_hyperparams.json")
                with open(best_hyperparams_path, 'w') as f:
                    json.dump(best_hyperparams, f)
                
                # 훈련 상태 저장 (Optimizer, Epoch 등)
                checkpoint_path = os.path.join(model_save_dir, f"optuna_checkpoint_latest.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'epochs': epochs,
                    'trial_number':trial.number,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'no_improvement_count': no_improvement_count,
                    'latent_dim': latent_dim,  
                    'hidden_dim': hidden_dim,
                    'batch_size': batch_size,
                    'learning_rate':learning_rate,
                    'beta': beta
                }, checkpoint_path)

                # 잠재 공간 저장 및 시각화
                model.eval()
                with torch.no_grad():
                    latent_train_list, latent_val_list = [], []
                    for i in range(0, len(train_data), batch_size):
                        batch = train_data[i:i+batch_size].to(device)
                        _, mean, _ = model(batch)
                        latent_train_list.append(mean.cpu().numpy())
                    for i in range(0, len(validation_data), batch_size):
                        batch = validation_data[i:i+batch_size].to(device)
                        _, mean, _ = model(batch)
                        latent_val_list.append(mean.cpu().numpy())
                    latent_space_train = np.vstack(latent_train_list)
                    latent_space_val = np.vstack(latent_val_list)

                embedding_info_dir = os.path.join(model_save_dir, "embedding_info")
                os.makedirs(embedding_info_dir, exist_ok=True)
                visualize_latent_space_and_save(latent_space_train, latent_space_val, latent_dim, embedding_info_dir, best_loss, model_file=os.path.basename(best_model_path))
                model.train() 
                # Early stopping logic based on improvement
                # min_delta = 1e-4  # Minimum loss change to reset counter
                # if best_loss - val_loss < min_delta:
                #     no_improvement_count += 1
                # else:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Early stopping
            if no_improvement_count >= patience:
                print(f"Early stopping at epoch {epoch+1} with no improvement for {patience} epochs.")
                break

    writer.close()
    return model, best_loss

def compute_validation_loss(model, validation_data, criterion, batch_size, device): # recon loss & kl loss도 전체에 대한 평균으로 교체
    model.eval()
    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0

    with torch.no_grad():
        for i in range(0, len(validation_data), batch_size):
            batch = validation_data[i:i+batch_size].to(device)
            reconstructed, mean, logvar = model(batch)

            reconstruction_loss = criterion(reconstructed, batch)
            kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            loss = reconstruction_loss + beta * kl_loss

            batch_size_actual = batch.size(0)  # 마지막 배치 처리용

            total_loss += loss.item() * batch_size_actual
            total_reconstruction_loss += reconstruction_loss.item() * batch_size_actual
            total_kl_loss += kl_loss.item() * batch_size_actual

    dataset_size = len(validation_data)
    avg_total_loss = total_loss / dataset_size
    avg_reconstruction_loss = total_reconstruction_loss / dataset_size
    avg_kl_loss = total_kl_loss / dataset_size

    return avg_total_loss, avg_reconstruction_loss, avg_kl_loss

