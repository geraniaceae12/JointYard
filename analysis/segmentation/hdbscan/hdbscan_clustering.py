import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cupy as cp
import rmm

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from analysis.dim_reduction.pca import perform_pca, save_pca_results
from analysis.dim_reduction.umap import perform_umap, plot_umap
from cuml.cluster.hdbscan.hdbscan import HDBSCAN as cuml_HDBSCAN
from cuml.cluster.hdbscan.prediction import all_points_membership_vectors as cuml_all_points_membership_vectors
from hdbscan import HDBSCAN as cpu_HDBSCAN
from hdbscan.prediction import all_points_membership_vectors

def apply_clustering(latent_space, hdbscan_cfg, min_cluster_size, info_save_dir):
    run_gpu = hdbscan_cfg["run_gpu"]
    dim_reduction_type = hdbscan_cfg["dim_reduction_type"]
    min_samples = hdbscan_cfg["min_samples"] if hdbscan_cfg["min_samples"] is not None else min_cluster_size
    method = hdbscan_cfg["method"]
    noise_method = hdbscan_cfg["noise_method"] 
    prob_threshold = hdbscan_cfg["prob_threshold"]
    std_threshold = hdbscan_cfg["std_threshold"]
    save_plot = hdbscan_cfg["save_plot"]
    save_plotformat = hdbscan_cfg["save_plotformat"]
    save_plotdir = os.path.join(info_save_dir,'hdbscan') 
    if not os.path.exists(save_plotdir):
        os.makedirs(save_plotdir)

    if run_gpu:
        # Initialize GPU memory and check it
        if not rmm.is_initialized():
            print("Initializing RMM before clustering...")
            rmm.reinitialize(pool_allocator=True)
        print("GPU memory usage before clustering:", cp.get_default_memory_pool().used_bytes())
        print("RMM initialized:", rmm.is_initialized())
        # GPU-based HDBSCAN
        clusterer = cuml_HDBSCAN(
            min_cluster_size=min_cluster_size, prediction_data=True,
            min_samples=min_samples, cluster_selection_method= method).fit(latent_space)         
        soft_clusters = cuml_all_points_membership_vectors(clusterer)
        print(f"HDBSCAN completed on GPU with min_cluster_size = {min_cluster_size}")
    else:
        latent_space = latent_space.get() if isinstance(latent_space, cp.ndarray) else latent_space        
        # CPU-based HDBSCAN
        clusterer = cpu_HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=-1, prediction_data=True).fit(latent_space)
        soft_clusters = all_points_membership_vectors(clusterer)
        print(f"HDBSCAN completed on CPU with min_cluster_size = {min_cluster_size}")
    
    # Make saving directory
    model_dir = os.path.join(save_plotdir,'model')
    os.makedirs(model_dir, exist_ok=True)

    # Condensed tree plotting 
    cluster_labels = clusterer.labels_
    num_clusters = len(np.unique(cluster_labels))
    palette = sns.color_palette('Spectral', num_clusters) if num_clusters > 18 else sns.color_palette('tab20', num_clusters)
    condensed_tree_filename = f"condensed_tree_{min_cluster_size}.{save_plotformat}"
    clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=palette)
    plt.savefig(os.path.join(model_dir, condensed_tree_filename), format=save_plotformat)
    plt.close()
    condensed_tree_df = clusterer.condensed_tree_.to_pandas()
    condensed_tree_df.to_csv(os.path.join(model_dir,'condensed_tree.csv'), index=False)

    # Save HDBSCAN model
    model_filename = f"hdbscan_model_{min_cluster_size}.pkl"
    with open(os.path.join(model_dir, model_filename), 'wb') as f:
        pickle.dump(clusterer, f)

    # Save cluster data
    cluster_probs = clusterer.probabilities_
    soft_cluster_probs = soft_clusters
    np.save(os.path.join(model_dir, f'softcluster_prob_{min_cluster_size}.npy'), soft_cluster_probs)

    # Process noise with the specified method
    soft_labels, max_probs = process_noise(cluster_labels, soft_cluster_probs, prob_threshold, std_threshold, noise_method)

    # Append the cluster labels and probabilities to index_info.csv
    index_info_path = os.path.join(info_save_dir, 'index_info.csv')
    index_info_df = pd.read_csv(index_info_path)
    if len(index_info_df) == len(cluster_labels):
        index_info_df[f"Clusterlabel_{min_cluster_size}"] = cluster_labels
        index_info_df[f"Clusterprob_{min_cluster_size}"] = cluster_probs
        index_info_df[f"Softlabel_{min_cluster_size}"] = soft_labels
        index_info_df[f"Softprob_{min_cluster_size}"] = max_probs
    else:
        raise ValueError("The length of cluster labels does not match the number of rows in index_info.csv")
    index_info_df.to_csv(index_info_path, index=False)

    # Free GPU memory for HDBSCAN-related variables
    del clusterer, soft_clusters, cluster_probs, soft_cluster_probs
    cp.get_default_memory_pool().free_all_blocks()

    # # Plot
    reduced_data = None
    # plot_2d = hdbscan_cfg["plot_2d"]
    # plot_3d = hdbscan_cfg["plot_3d"]
    # plot_show = hdbscan_cfg["plot_show"]
    # save_plotdir = os.path.join(save_plotdir,'plot')
    # if not os.path.exists(save_plotdir):
    #     os.makedirs(save_plotdir)

    # if not plot_2d and not plot_3d:
    #     print("Warning: Both plot_2d and plot_3d are False. At least one should be True.")
    # else:
    #     if dim_reduction_type == "pca":
    #         if plot_2d:
    #             #print("\nPlot 2d start -------- pca")
    #             pcadata, pca, reduced_data, n_components, explained_variance_ratio = perform_pca(latent_space, n_components=2)
    #             filename = f"{dim_reduction_type}_{min_cluster_size}_2d_no_{explained_variance_ratio[1]:.3f}"
    #             plot_clusters_no_noise(reduced_data, soft_labels, plot_2d=True, plot_3d=False, plot_show=plot_show, save_plot=save_plot, file_name=filename, save_plotdir=save_plotdir, save_plotformat=save_plotformat)
    #             filename = f"{dim_reduction_type}_{min_cluster_size}_2d_{explained_variance_ratio[1]:.3f}"
    #             plot_clusters(reduced_data, soft_labels, plot_2d=True, plot_3d=False, plot_show=plot_show, save_plot=save_plot, file_name=filename, save_plotdir=save_plotdir, save_plotformat=save_plotformat)

    #         if plot_3d:
    #             #print("\nPlot 3d start -------- pca")
    #             pcadata, pca, reduced_data, n_components, explained_variance_ratio = perform_pca(latent_space, n_components=3)
    #             filename = f"{dim_reduction_type}_{min_cluster_size}_3d_no_{explained_variance_ratio[2]:.3f}"
    #             plot_clusters_no_noise(reduced_data, soft_labels, plot_2d=False, plot_3d=True, plot_show=plot_show, save_plot=save_plot, file_name=filename, save_plotdir=save_plotdir, save_plotformat=save_plotformat)
    #             filename = f"{dim_reduction_type}_{min_cluster_size}_3d_{explained_variance_ratio[2]:.3f}"
    #             plot_clusters(reduced_data, soft_labels, plot_2d=False, plot_3d=True, plot_show=plot_show, save_plot=save_plot, file_name=filename, save_plotdir=save_plotdir, save_plotformat=save_plotformat)

    #     elif dim_reduction_type == "umap":
    #         if plot_2d:
    #             #print("\nPlot 2d start -------- umap")
    #             reduced_data, _ = perform_umap(latent_space,n_components=2) # perform_umap(latent_space, n_neighbors=30, min_dist=0.1, n_components=2)
    #             filename = f"{dim_reduction_type}_{min_cluster_size}_2d_no"
    #             plot_clusters_no_noise(reduced_data, soft_labels, plot_2d=True, plot_3d=False, plot_show=plot_show, save_plot=save_plot, file_name=filename, save_plotdir=save_plotdir, save_plotformat=save_plotformat)
    #             filename = f"{dim_reduction_type}_{min_cluster_size}_2d"
    #             plot_clusters(reduced_data, soft_labels, plot_2d=True, plot_3d=False, plot_show=plot_show, save_plot=save_plot, file_name=filename, save_plotdir=save_plotdir, save_plotformat=save_plotformat)

    #         if plot_3d:
    #             #print("\nPlot 3d start -------- umap")
    #             reduced_data, _ = perform_umap(latent_space,n_components=3)
    #             filename = f"{dim_reduction_type}_{min_cluster_size}_3d_no"
    #             plot_clusters_no_noise(reduced_data, soft_labels, plot_2d=False, plot_3d=True, plot_show=plot_show, save_plot=save_plot, file_name=filename, save_plotdir=save_plotdir, save_plotformat=save_plotformat)
    #             filename = f"{dim_reduction_type}_{min_cluster_size}_3d"
    #             plot_clusters(reduced_data, soft_labels, plot_2d=False, plot_3d=True, plot_show=plot_show, save_plot=save_plot, file_name=filename, save_plotdir=save_plotdir, save_plotformat=save_plotformat)

    # plt.clf()
    cp.get_default_memory_pool().free_all_blocks()
    
    return reduced_data, cluster_labels, soft_labels, max_probs, index_info_df

def process_noise(cluster_labels, soft_cluster_probs, prob_threshold, std_threshold, noise_method):
    max_probs = soft_cluster_probs.max(axis=1)  # max probabilities in each point
    best_clusters = soft_cluster_probs.argmax(axis=1)  # cluster index of max probability

    # Initialize soft labels with cluster labels
    soft_labels = cluster_labels.copy()

    # Process only points labeled as noise (-1) in cluster_labels
    noise_points = soft_labels == -1
    
    if noise_method == "prob_threshold":
        # Noise processing by probability threshold for noise points only
        soft_labels[noise_points] = np.where(max_probs[noise_points] >= prob_threshold, best_clusters[noise_points], -1)
        
    elif noise_method == "std_threshold":
        # Compute mean and std_dev for each cluster based on all points assigned to that cluster
        cluster_means = {}
        cluster_std_devs = {}

        for cluster in np.unique(best_clusters[best_clusters >= 0]):  # Only valid clusters
            cluster_probs = max_probs[best_clusters == cluster]  # Get max_probs for all points in this cluster
            cluster_means[cluster] = cluster_probs.mean()
            cluster_std_devs[cluster] = cluster_probs.std()

        # Process noise points based on calculated thresholds
        for idx in np.where(noise_points)[0]:  # Only noise points
            cluster = best_clusters[idx]
            if cluster in cluster_means:  # Check if the cluster is valid
                noise_threshold = cluster_means[cluster] - std_threshold * cluster_std_devs[cluster]
                if max_probs[idx] >= noise_threshold:
                    soft_labels[idx] = cluster  # Assign to cluster if above threshold

    else:
        raise ValueError("Invalid noise_method. Choose either 'prob_threshold' or 'std_deviation'.")
    
    # Output total number of data points and number of noise points
    total_points = soft_labels.size
    noise_points_count = np.sum(soft_labels == -1)
    
    print(f"Total data points: {total_points}")
    print(f"Number of points classified as noise: {noise_points_count}")
    print(f"Number of clusters found: {len(np.unique(best_clusters[best_clusters >= 0]))}")
    
    return soft_labels, max_probs
        
def plot_clusters(data, cluster_labels, plot_2d=True, plot_3d=False, plot_show = False, save_plot=False,
                   file_name=None, save_plotdir=None, save_plotformat='pdf'):

    # Create a color palette
    palette = sns.color_palette('deep', np.unique(cluster_labels).max() + 1)

    # Assign colors to clusters; black for noise
    colors = [palette[x] if x >= 0 else (0.9, 0.9, 0.9) for x in cluster_labels]

    # Count data points
    num_points = data.shape[0]
    cluster_num_points = num_points - data[cluster_labels== -1].shape[0]
    print(f"Total data points: {num_points}")

    if plot_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot for clusters
        scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, alpha=0.6, s=2, linewidths=0)

        # Plot transparency around cluster points
        for i, color in enumerate(palette):
            cluster_points = data[cluster_labels == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                       color=color, alpha=0.0002, s=100, linewidths=0, edgecolor=color)

        # Plot noise points with reduced visibility
        noise_points = data[cluster_labels == -1]
        ax.scatter(noise_points[:, 0], noise_points[:, 1], noise_points[:, 2], 
                   color='black', alpha=1e-7, s=0.5, linewidths=0)

        ax.set_title(f'Clusters found by HDBSCAN\n cluster:{cluster_num_points}, total:{num_points}', fontsize=14)

        # Add legend with cluster colors and counts outside the plot
        unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[label], markersize=10) for label in unique_labels]
        labels = [f'Cluster {i}' for i in unique_labels]
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=5)
        plt.tight_layout(pad=1.0, rect=[0, 0, 0.85, 1])  # Adjust `rect` to make space for the legend

        if save_plot:
            save_plot_with_format(file_name, save_plotdir, plot_save_format=save_plotformat)

        if plot_show:
            plt.show()
        else:
            plt.close(fig)

    if plot_2d:
        fig = plt.figure()
        plt.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.6, s=2, linewidths=0)

        # Plot transparency around cluster points
        for i, color in enumerate(palette):
            cluster_points = data[cluster_labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                        color=color, alpha=0.0002, s=100, linewidths=0, edgecolor=color)

        # Plot noise points with reduced visibility
        noise_points = data[cluster_labels == -1]
        plt.scatter(noise_points[:, 0], noise_points[:, 1], 
                    color='black', alpha=1e-7, s=0.5, linewidths=0)

        plt.title(f'Clusters found by HDBSCAN\n cluster:{cluster_num_points}, total:{num_points}', fontsize=14)

        # Add legend with cluster colors and counts outside the plot
        unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[label], markersize=10) for label in unique_labels]
        labels = [f'Cluster {i}' for i in unique_labels]
        plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=5)
        plt.tight_layout(pad=1.0, rect=[0, 0, 0.85, 1])  # Adjust `rect` to make space for the legend

        if save_plot:
            save_plot_with_format(file_name, save_plotdir, plot_save_format=save_plotformat)

        if plot_show:
            plt.show()
        else:
            plt.close(fig)

    return cluster_labels

def plot_clusters_no_noise(data, cluster_labels, plot_2d=True, plot_3d=False, plot_show = False, save_plot=False,
                            file_name=None, save_plotdir=None, save_plotformat='pdf'):

    # Exclude noise points
    non_noise_indices = cluster_labels != -1
    filtered_data = data[non_noise_indices]
    filtered_labels = cluster_labels[non_noise_indices]
    
    # Create a color palette
    palette = sns.color_palette('deep', np.unique(filtered_labels).max() + 1)
    
    # Assign colors to clusters; black for noise
    colors = [palette[x] for x in filtered_labels]

    # Count data points
    num_points = filtered_data.shape[0]
    print(f"Total data points (no noise): {num_points}")

    if plot_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot for clusters
        scatter = ax.scatter(filtered_data[:, 0], filtered_data[:, 1], filtered_data[:, 2], c=colors, alpha=0.6, s=2, linewidths=0)

        # Plot transparency around cluster points
        for i, color in enumerate(palette):
            cluster_points = filtered_data[filtered_labels == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                       color=color, alpha=0.0002, s=100, linewidths=0, edgecolor=color)

        ax.set_title(f'Clusters found by HDBSCAN (No Noise)\n cluster:{num_points}, total:{data.shape[0]}', fontsize=14)

        # Add legend with cluster colors and counts outside the plot
        unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[label], markersize=10) for label in range(len(unique_labels))]
        labels = [f'Cluster {i}' for i in unique_labels]
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=5)
        plt.tight_layout(pad=1.0, rect=[0, 0, 0.85, 1])  # Adjust `rect` to make space for the legend

        if save_plot:
            save_plot_with_format(file_name, save_plotdir, plot_save_format=save_plotformat)

        if plot_show:
            plt.show()
        else:
            plt.close(fig)

    if plot_2d:
        fig = plt.figure()
        plt.scatter(filtered_data[:, 0], filtered_data[:, 1], c=colors, alpha=0.6, s=2, linewidths=0)

        # Plot transparency around cluster points
        for i, color in enumerate(palette):
            cluster_points = filtered_data[filtered_labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                        color=color, alpha=0.0002, s=100, linewidths=0, edgecolor=color)

        plt.title(f'Clusters found by HDBSCAN (No Noise)\n cluster:{num_points}, total:{data.shape[0]}', fontsize=14)

        # Add legend with cluster colors and counts outside the plot
        unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[label], markersize=10) for label in range(len(unique_labels))]
        labels = [f'Cluster {i}' for i in unique_labels]
        plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=5)
        plt.tight_layout(pad=1.0, rect=[0, 0, 0.85, 1])  # Adjust `rect` to make space for the legend

        if save_plot:
            save_plot_with_format(file_name, save_plotdir, plot_save_format=save_plotformat)

        if plot_show:
            plt.show()
        else:
            plt.close(fig)

    return filtered_labels

def save_plot_with_format(file_name, save_plotdir = None, plot_save_format='pdf'):
    # If save_plotdir is None, use the current directory
    if save_plotdir is None:
        save_plotdir = os.getcwd()  # Get the current working directory

    file_path = os.path.join(save_plotdir, f'{file_name}.{plot_save_format}')
    plt.savefig(file_path, format=plot_save_format)
    print(f"Saved plot as {file_path}")

