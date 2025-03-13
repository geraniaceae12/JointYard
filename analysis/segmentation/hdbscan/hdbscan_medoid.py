import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import math

from scipy.spatial.distance import cdist
from analysis.segmentation.hdbscan.hdbscan_confirm import load_video_frames
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import ticker

def calculate_cluster_medoids(data, labels, frames_to_show_per_medoid=6, selection_mode='closest', 
                              random_seed=42, batch_size=10000, sampling_frames = 50):
    """
    Calculate the medoids for clusters without storing the full distance matrix.

    Parameters:
    - data: np.ndarray
        The dataset, where each row represents a data point.
    - labels: np.ndarray
        Cluster labels for each data point (-1 indicates noise).
    - frames_to_show_per_medoid: int
        The number of points to select per medoid (including the medoid itself).
    - selection_mode: str
        Mode for selecting points. Options are:
        - 'closest': Select points closest to the medoid.
        - 'random': Randomly select points within the cluster.
    - random_seed: int
        Seed for reproducibility in random selection.
    - batch_size: int
        The number of points to process at a time during pairwise distance computation.

    Returns:
    - medoids: list
        A list of medoid coordinates for each cluster.
    - medoid_indices: list
        A list of indices corresponding to the medoids in the original dataset.
    - select_points_indices: list
        A list of lists, where each sublist contains the indices of selected points
        (close to or randomly chosen) within the corresponding cluster.
    """
    np.random.seed(random_seed)
    unique_labels = np.unique(labels[labels >= 0])

    medoids = []
    medoid_indices = []
    select_points_indices = []
    cluster_sampling_indices = [] # for sampling frames 

    for label in unique_labels:
        cluster_points_indices = np.where(labels == label)[0]
        cluster_points = data[cluster_points_indices]

        n = cluster_points.shape[0]

        # Incremental distance computation
        sum_distances = np.zeros(n, dtype=np.float32)  # Store sum of distances for each point

        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            for j in range(0, n, batch_size):
                end_j = min(j + batch_size, n)
                batch_distances = cdist(cluster_points[i:end_i], cluster_points[j:end_j], metric='euclidean')
                sum_distances[i:end_i] += batch_distances.sum(axis=1)  # Sum distances for each point in the batch

        # Find the medoid: point with the smallest sum of distances
        medoid_index_local = np.argmin(sum_distances)
        medoid_index_global = cluster_points_indices[medoid_index_local]

        medoids.append(cluster_points[medoid_index_local])
        medoid_indices.append(medoid_index_global)

        # Calculate Distances in each cluster
        distances_to_medoid = np.zeros(n, dtype=np.float32)
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            batch_distances = cdist(cluster_points[i:end_i], cluster_points[medoid_index_local:medoid_index_local + 1], metric='euclidean')
            distances_to_medoid[i:end_i] = batch_distances.flatten()

        # Sampling closest frames of each cluster
        sorted_indices_localC = np.argsort(distances_to_medoid)[:sampling_frames]
        sorted_indices_globalC = cluster_points_indices[sorted_indices_localC]
        cluster_sampling_indices.append(sorted_indices_globalC)

        if selection_mode == 'closest':
            # Select points closest to the medoid (compute distances in smaller chunks)
            sorted_indices_local = np.argsort(distances_to_medoid)[:frames_to_show_per_medoid]
            sorted_indices_global = cluster_points_indices[sorted_indices_local]
            select_points_indices.append(sorted_indices_global)
        elif selection_mode == 'random':
            num_points_to_select = min(frames_to_show_per_medoid, len(cluster_points_indices))
            random_indices_local = np.random.choice(
                len(cluster_points_indices), size=num_points_to_select, replace=False
            )
            random_indices_global = cluster_points_indices[random_indices_local]
            select_points_indices.append(random_indices_global)
        else:
            raise ValueError("\nInvalid selection_mode. Choose 'closest' or 'random'.")

    print("\nIncremental medoids calculation complete.")

    return medoids, medoid_indices, select_points_indices, cluster_sampling_indices

def plot_latent_space_with_medoids(reduced_data, cluster_labels, medoids, select_points_indices, index_info_df, 
                                   save_dir, filename, frames_to_show_per_medoid, selection_mode='closest',
                                   plot_3d=False, plot_show=False, save_ani=False):
    """
    Plots the latent space with medoids and creates an animation to display frames closest to or randomly selected from medoids.
    
    Parameters:
    - reduced_data: np.ndarray
        The 2D or 3D reduced latent representation of the data.
    - cluster_labels: np.ndarray
        Cluster labels for each data point (-1 for noise).
    - medoids: list
        Coordinates of the medoids for each cluster.
    - select_points_indices: list
        Indices of selected points (closest or random) for each medoid.
    - index_info_df: pd.DataFrame
        A DataFrame containing video paths and frame indices corresponding to the data points.
    - save_dir: str
        Directory to save the animation.
    - filename: str
        Filename for the animation.
    - frames_to_show_per_medoid: int
        Number of frames to display per medoid (including the medoid itself).
    - selection_mode: str
        Frame selection mode ('closest' or 'random').
    - plot_3d: bool
        Whether to plot in 3D.
    - plot_show: bool
        Whether to display the plot.
    - save_ani: bool
        Whether to save the animation as an MP4 file.

    Returns:
    - None
    """
    global fig, ax_latent, ax_frames, scatter, medoid_mark, colorbar
    colorbar = None

    def update_frame(frame):
        global ax_latent, ax_frames, scatter, medoid_mark, colorbar

        # Clear the latent space plot
        ax_latent.clear()

        # Medoid and selected points for the current frame
        medoid = medoids[frame]
        closest_indices = select_points_indices[frame]

        # Adjust the number of rows and columns in the frames grid based on frames_to_show_per_medoid
        rows = math.ceil(frames_to_show_per_medoid / 2)  # At least 2 columns per row
        cols = 2 if frames_to_show_per_medoid > 1 else 1  # If 1 frame, use 1 column

        # Update frames grid dynamically
        frames_gs = gs[0, 1].subgridspec(rows, cols)
        ax_frames = np.array([fig.add_subplot(frames_gs[i // cols, i % cols]) for i in range(frames_to_show_per_medoid)])

        for ax in ax_frames.flatten():
            ax.axis('off')

        # Filter data to exclude -1 labels
        valid_indices = cluster_labels != -1
        filtered_labels = cluster_labels[valid_indices]
        filtered_data = reduced_data[valid_indices]

        # Generate a colormap with a fixed number of colors (one for each cluster label)
        unique_labels = np.unique(filtered_labels)
        num_clusters = len(unique_labels)

        # Colormap과 BoundaryNorm 설정
        cmap = plt.cm.get_cmap('Spectral',num_clusters)  # viridis는 연속적이고 다양한 색상을 제공
        bounds = np.arange(-0.5, num_clusters + 0.5, 1)
        norm = BoundaryNorm(bounds, cmap.N)

        # Plot latent space
        if plot_3d:
            scatter = ax_latent.scatter(filtered_data[:, 0], filtered_data[:, 1], filtered_data[:, 2],
                                        c=filtered_labels, cmap=cmap, norm=norm, alpha=0.6, s=2)
            medoid_mark = ax_latent.scatter(medoid[0], medoid[1], medoid[2], c='black', marker='X', s=100, label='Medoid', zorder=5)
            ax_latent.scatter(reduced_data[closest_indices, 0], reduced_data[closest_indices, 1], reduced_data[closest_indices, 2],
                              c='red', marker='o', s=50, label='Selected Points')
            ax_latent.set_title('Latent Space with Clusters and Medoids (3D)', fontsize=14, pad=0.04)
            ax_latent.set_xlabel('X')
            ax_latent.set_ylabel('Y')
            ax_latent.set_zlabel('Z')
            ax_latent.grid(False)
        else:
            scatter = ax_latent.scatter(filtered_data[:, 0], filtered_data[:, 1],
                                        c=filtered_labels, cmap=cmap, norm=norm, alpha=0.6, s=2)
            medoid_mark = ax_latent.scatter(medoid[0], medoid[1], c='black', marker='X', s=100, label='Medoid', zorder=5)
            ax_latent.scatter(reduced_data[closest_indices, 0], reduced_data[closest_indices, 1],
                              c='red', marker='o', s=50, label='Selected Points')
            ax_latent.set_title('Latent Space with Clusters and Medoids (2D)', fontsize=14, pad=0.04)
            ax_latent.set_xlabel('X')
            ax_latent.set_ylabel('Y')
            ax_latent.grid(False)

        # Add colorbar (Cluster Labels)
        if colorbar:
            colorbar.remove()
        colorbar = fig.colorbar(scatter, ax=ax_latent, label='Cluster Label', orientation='horizontal', fraction=0.046, pad=0.04, aspect=50)

        # Ticker를 클러스터 라벨로 설정 (간격 조정)
        tick_positions = unique_labels[::5] if num_clusters > 20 else unique_labels[::2] # 고유 라벨만 표시
        colorbar.set_ticks(tick_positions)
        colorbar.ax.set_xticklabels([str(int(label)) for label in tick_positions])

        # Clear and plot video frames
        if isinstance(ax_frames, np.ndarray):
            for ax in ax_frames.flatten():
                ax.clear()
                ax.axis('off')
        else:
            ax_frames.clear()
            ax_frames.axis('off')

        # Retrieve video frames based on MultiIndex
        frames = []
        for idx in closest_indices[:frames_to_show_per_medoid]:
            video_path = index_info_df.loc[idx, 'VideoPath']
            frame_index = index_info_df.loc[idx, 'RowIndex']
            frame = load_video_frames(video_path, frame_index)
            frames.append(frame)
            
        for idx, frame_img in enumerate(frames):
            ax_frames[idx].imshow(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))

        # Add plot title
        cluster_label = cluster_labels[closest_indices[0]]
        fig.suptitle(f'Frames for Cluster {cluster_label} Medoid', fontsize=14)

    # Create figure and grid spec
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 4])
    ax_latent = fig.add_subplot(gs[0, 0], projection='3d' if plot_3d else None)

    # Create animation
    ani = FuncAnimation(fig, update_frame, frames=len(medoids), repeat=True, interval=3000)
    if save_ani:
        ani.save(os.path.join(save_dir, filename + '_latent_ani.mp4'), writer='ffmpeg', fps=1)
        print("\nAnimation saved.")

    if plot_show:
        plt.show()
    plt.clf()

def plot_latent_space_with_medoids2(umap_space, cluster_labels, medoid_indices, select_points_indices, index_info_df, 
                                   save_dir, filename, frames_to_show_per_medoid, selection_mode='closest',
                                   plot_3d=False, plot_show=False, save_ani=False):
    """
    Plots the latent space with medoids and creates an animation to display frames closest to or randomly selected from medoids.
    
    Parameters:
    - umap_space: np.ndarray
        The 2D or 3D umap_space representation of the data.
    - cluster_labels: np.ndarray
        Cluster labels for each data point (-1 for noise).
    - medoids: list
        Coordinates of the medoids for each cluster.
    - select_points_indices: list
        Indices of selected points (closest or random) for each medoid.
    - index_info_df: pd.DataFrame
        A DataFrame containing video paths and frame indices corresponding to the data points.
    - save_dir: str
        Directory to save the animation.
    - filename: str
        Filename for the animation.
    - frames_to_show_per_medoid: int
        Number of frames to display per medoid (including the medoid itself).
    - selection_mode: str
        Frame selection mode ('closest' or 'random').
    - plot_3d: bool
        Whether to plot in 3D.
    - plot_show: bool
        Whether to display the plot.
    - save_ani: bool
        Whether to save the animation as an MP4 file.

    Returns:
    - None
    """
    global fig, ax_latent, ax_frames, scatter, medoid_mark, colorbar
    colorbar = None

    def update_frame(frame):
        global ax_latent, ax_frames, scatter, medoid_mark, colorbar

        # Clear the latent space plot
        ax_latent.clear()

        # Medoid and selected points for the current frame
        medoid = medoid_indices[frame]
        closest_indices = select_points_indices[frame]

        # Adjust the number of rows and columns in the frames grid based on frames_to_show_per_medoid
        rows = math.ceil(frames_to_show_per_medoid / 2)  # At least 2 columns per row
        cols = 2 if frames_to_show_per_medoid > 1 else 1  # If 1 frame, use 1 column

        # Update frames grid dynamically
        frames_gs = gs[0, 1].subgridspec(rows, cols)
        ax_frames = np.array([fig.add_subplot(frames_gs[i // cols, i % cols]) for i in range(frames_to_show_per_medoid)])

        for ax in ax_frames.flatten():
            ax.axis('off')

        # Filter data to exclude -1 labels
        valid_indices = cluster_labels != -1
        filtered_labels = cluster_labels[valid_indices]
        filtered_data = umap_space[valid_indices]

        # Generate a colormap with a fixed number of colors (one for each cluster label)
        unique_labels = np.unique(filtered_labels)
        num_clusters = len(unique_labels)

        # Colormap과 BoundaryNorm 설정
        cmap = plt.cm.get_cmap('Spectral',num_clusters)  # viridis는 연속적이고 다양한 색상을 제공
        bounds = np.arange(-0.5, num_clusters + 0.5, 1)
        norm = BoundaryNorm(bounds, cmap.N)

        # Plot latent space
        if plot_3d:
            scatter = ax_latent.scatter(filtered_data[:, 0], filtered_data[:, 1], filtered_data[:, 2],
                                        c=filtered_labels, cmap=cmap, norm=norm, alpha=0.6, s=1.5)
            medoid_mark = ax_latent.scatter(umap_space[medoid,0], umap_space[medoid,1], umap_space[medoid,2], c='black', marker='X', s=100, label='Medoid', zorder=5)
            ax_latent.scatter(umap_space[closest_indices, 0],umap_space[closest_indices, 1], umap_space[closest_indices, 2],
                              c='red', marker='o', s=50, label='Selected Points')
            ax_latent.set_title('Latent Space with Clusters and Medoids (3D)', fontsize=14, pad=0.04)
            ax_latent.set_xlabel('X')
            ax_latent.set_ylabel('Y')
            ax_latent.set_zlabel('Z')
            ax_latent.grid(False)
        else:
            scatter = ax_latent.scatter(filtered_data[:, 0], filtered_data[:, 1],
                                        c=filtered_labels, cmap=cmap, norm=norm, alpha=0.6, s=1.5)
            medoid_mark = ax_latent.scatter(umap_space[medoid,0], umap_space[medoid,1], c='black', marker='X', s=100, label='Medoid', zorder=5)
            ax_latent.scatter(umap_space[closest_indices, 0], umap_space[closest_indices, 1],
                              c='red', marker='o', s=50, label='Selected Points')
            ax_latent.set_title('Latent Space with Clusters and Medoids (2D)', fontsize=14, pad=0.04)
            ax_latent.set_xlabel('X')
            ax_latent.set_ylabel('Y')
            ax_latent.grid(False)

        # Add colorbar (Cluster Labels)
        if colorbar:
            colorbar.remove()
        colorbar = fig.colorbar(scatter, ax=ax_latent, label='Cluster Label', orientation='horizontal', fraction=0.046, pad=0.04, aspect=50)

        # Ticker를 클러스터 라벨로 설정 (간격 조정)
        tick_positions = unique_labels[::5] if num_clusters > 20 else unique_labels[::2] # 고유 라벨만 표시
        colorbar.set_ticks(tick_positions)
        colorbar.ax.set_xticklabels([str(int(label)) for label in tick_positions])

        # Clear and plot video frames
        if isinstance(ax_frames, np.ndarray):
            for ax in ax_frames.flatten():
                ax.clear()
                ax.axis('off')
        else:
            ax_frames.clear()
            ax_frames.axis('off')

        # Retrieve video frames based on MultiIndex
        frames = []
        for idx in closest_indices[:frames_to_show_per_medoid]:
            video_path = index_info_df.loc[idx, 'VideoPath']
            frame_index = index_info_df.loc[idx, 'RowIndex']
            frame = load_video_frames(video_path, frame_index)
            frames.append(frame)
            
        for idx, frame_img in enumerate(frames):
            ax_frames[idx].imshow(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))

        # Add plot title
        cluster_label = cluster_labels[closest_indices[0]]
        fig.suptitle(f'Frames for Cluster {cluster_label} Medoid', fontsize=14)

    # Create figure and grid spec
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 4])
    ax_latent = fig.add_subplot(gs[0, 0], projection='3d' if plot_3d else None)

    # Create animation
    ani = FuncAnimation(fig, update_frame, frames=len(medoid_indices), repeat=True, interval=3000)
    if save_ani:
        ani.save(os.path.join(save_dir, filename + '_latent_ani.mp4'), writer='ffmpeg', fps=1)
        print("\nAnimation saved.")

    if plot_show:
        plt.show()
    plt.clf()

