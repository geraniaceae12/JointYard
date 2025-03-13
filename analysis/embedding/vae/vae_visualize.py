import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import cv2
import os
from .vae_model import DeepVAE, VanillaVAE


def load_model(model_type, model_path, device=None):
    if model_type == 'deepvae':
        model = DeepVAE()  # Initialize with the appropriate parameters if needed.
    elif model_type == 'vanillavae':
        model = VanillaVAE()  # Initialize with the appropriate parameters if needed.
    else:
        raise ValueError("Invalid model type")

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def extract_latent_space(model, data, device = None, batch_size=32):
    latent_space = np.empty((0, model.latent_dim))
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].to(device)
            _, mean, _ = model(batch)
            latent_space = np.vstack([latent_space, mean.cpu().numpy()])
    return latent_space

def visualize_latent_space_and_save(latent_space_train, latent_space_val, latent_dim, log_dir, val_loss, model_file=None):
    combined_latent_space = np.vstack([latent_space_train, latent_space_val])

    pca = PCA()
    pca.fit(combined_latent_space)

    cumsum_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    num_components_90 = np.argmax(cumsum_variance_ratio >= 0.9) + 1

    pca = PCA(n_components=num_components_90)
    pca_latent_space = pca.fit_transform(combined_latent_space)

    if num_components_90 == 1:
        pass
    elif num_components_90 == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_latent_space[:len(latent_space_train), 0], pca_latent_space[:len(latent_space_train), 1], c='b', alpha=0.2, label='Train Data', s=1)
        plt.scatter(pca_latent_space[len(latent_space_train):, 0], pca_latent_space[len(latent_space_train):, 1], c='r', alpha=0.2, label='Validation Data', s=1)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'Latent Space Visualization (2D), latent_dim: {latent_dim}\nval_loss: {val_loss:.2f}, PC:{100 * cumsum_variance_ratio[num_components_90-1]:.2f}%')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, f'latent_space_{model_file}.png'))
        plt.close()
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pca_latent_space[:len(latent_space_train), 0], pca_latent_space[:len(latent_space_train), 1], pca_latent_space[:len(latent_space_train), 2], c='b', alpha=0.2, label='Train Data', s=1)
        ax.scatter(pca_latent_space[len(latent_space_train):, 0], pca_latent_space[len(latent_space_train):, 1], pca_latent_space[len(latent_space_train):, 2], c='r', alpha=0.2, label='Validation Data', s=1)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title(f'Latent Space Visualization (3D)_latentDim:{latent_dim}\nval_loss: {val_loss:.2f}, PC:{100 * cumsum_variance_ratio[num_components_90-1]:.2f}%')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, f'latent_space_{model_file}.png'))
        plt.close()

def timeorder_visualize_latent_space_and_save(latent_space, time_data, latent_dim, log_dir, video_path=None, model_file=None):
    pca = PCA()
    pca.fit(latent_space)

    cumsum_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    num_components_90 = np.argmax(cumsum_variance_ratio >= 0.9) + 1

    pca = PCA(n_components=num_components_90)
    pca_latent_space = pca.fit_transform(latent_space)

    cmap = plt.get_cmap('viridis')

    if num_components_90 == 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(pca_latent_space[:, 0], pca_latent_space[:, 1], c=time_data, cmap=cmap, alpha=0.4, s=1)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'Latent Space Visualization (2D), latent_dim: {latent_dim}\n PC:{100 * cumsum_variance_ratio[num_components_90-1]:.2f}%')
        plt.colorbar(scatter, label='Time')
        plt.grid(True)
        
        if video_path:
            def on_hover(event):
                if event.inaxes == plt.gca():
                    x, y = event.xdata, event.ydata
                    # Find closest point in latent space
                    distances = np.sqrt((pca_latent_space[:, 0] - x) ** 2 + (pca_latent_space[:, 1] - y) ** 2)
                    index = np.argmin(distances)
                    frame_idx = index
                    frame = extract_frame_from_video(video_path, frame_idx)
                    if frame is not None:
                        plt.imshow(frame)
                        plt.title(f'Frame {frame_idx}')
                        plt.draw()

            plt.gcf().canvas.mpl_connect('motion_notify_event', on_hover)

        plt.savefig(os.path.join(log_dir, f'latent_space_{model_file}.png'))
        plt.close()
    elif num_components_90 == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(pca_latent_space[:, 0], pca_latent_space[:, 1], pca_latent_space[:, 2], c=time_data, cmap=cmap, alpha=0.4, s=1)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title(f'Latent Space Visualization (3D), latent_dim: {latent_dim}\n PC:{100 * cumsum_variance_ratio[num_components_90-1]:.2f}%')
        plt.colorbar(scatter, label='Time')
        plt.grid(True)

        if video_path:
            def on_hover(event):
                if event.inaxes == ax:
                    x, y, z = event.xdata, event.ydata, event.zdata
                    # Find closest point in latent space
                    distances = np.sqrt((pca_latent_space[:, 0] - x) ** 2 + (pca_latent_space[:, 1] - y) ** 2 + (pca_latent_space[:, 2] - z) ** 2)
                    index = np.argmin(distances)
                    frame_idx = index
                    frame = extract_frame_from_video(video_path, frame_idx)
                    if frame is not None:
                        ax.imshow(frame)
                        ax.set_title(f'Frame {frame_idx}')
                        plt.draw()

            fig.canvas.mpl_connect('motion_notify_event', on_hover)

        plt.savefig(os.path.join(log_dir, f'latent_space_{model_file}.png'))
        plt.close()

    reconstructed_data = pca.inverse_transform(pca_latent_space)
    return pca_latent_space, reconstructed_data

def extract_frame_from_video(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        return None

def main(model_path, data_path, config_path, visualize_mode='normal', video_path=None):
    from .vae_utils import load_config, check_gpu
    config = load_config(config_path)
    vae_config = config["vae"]
    model_type = vae_config['type']
    device = check_gpu()
    
    data = torch.from_numpy(np.load(data_path)).float()
    data = data.T.to(device)
    
    model = load_model(model_type, model_path, device)
    
    log_dir = os.path.dirname(model_path)

    if visualize_mode == 'normal':
        train_data, validation_data = train_test_split(data, test_size=vae_config.get('test_split', 0.2), random_state=42)
        latent_space_train = extract_latent_space(model, train_data, device)
        latent_space_val = extract_latent_space(model, validation_data, device)
        visualize_latent_space_and_save(latent_space_train, latent_space_val, model.latent_dim, log_dir, val_loss=0)  # val_loss는 실제 검증 손실을 사용해야 합니다.
    elif visualize_mode == 'timeorder' and video_path:
        latent_space = extract_latent_space(model, data, device)
        time_data = np.arange(len(latent_space))
        timeorder_visualize_latent_space_and_save(latent_space, time_data, model.latent_dim, log_dir, video_path=video_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='VAE Latent Space Visualization')
    parser.add_argument('model_path', type=str, help='Path to the trained VAE model')
    parser.add_argument('data_path', type=str, help='Path to the data file (numpy format)')
    parser.add_argument('config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--visualize_mode', type=str, choices=['normal', 'timeorder'], default='normal', help='Visualization mode: normal or timeorder')
    parser.add_argument('--video_path', type=str, help='Path to the video file (required for timeorder mode)')
    args = parser.parse_args()
    
    main(args.model_path, args.data_path, args.config_path, args.visualize_mode, args.video_path)
