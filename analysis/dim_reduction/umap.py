import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cupy as cp
from cuml.manifold import UMAP as cuUMAP

def perform_umap(data, n_neighbors=15, min_dist=0.1, n_components=2, save_dir=None):
    '''
    perform umap in GPU, cuML

    data : n_samples * n_features, numpy format

    Returns
    umap_space_cpu: n_samples * n_compontents, numpy format
    '''

    print("\n***Embedding with UMAP***")

    # Change the data type : numpy(cpu) into cupy(gpu)
    data = cp.asarray(data)

    # Run UMAP
    #umap_model = cuUMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    #umap_model = cuUMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42, metric = 'manhattan')
    umap_model = cuUMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    
    umap_space = umap_model.fit_transform(data)

    umap_save_dir = None
    if save_dir is not None:
        # Create 'umap' folder if not exists
        umap_save_dir = os.path.join(save_dir, 'umap')
        os.makedirs(umap_save_dir, exist_ok=True)

        # Save the result
        umap_result_path = os.path.join(umap_save_dir, f'umap_space_{n_neighbors}_{min_dist}.npy')
        np.save(umap_result_path, umap_space.get())
        print(f"Saved the UMAP result at {umap_result_path}.")

    # cuml(gpu) into numpy(cpu)
    umap_space_cpu = umap_space.get()

    # Free GPU memory by deleting GPU variables and clearing GPU memory cache
    del data, umap_space, umap_model
    cp.get_default_memory_pool().free_all_blocks()

    return umap_space_cpu, umap_save_dir  

def plot_umap(umap_space, n_neighbors, min_dist, plot_2d = False, plot_3d = False, 
              save_plot = False, umap_save_dir = None):
    '''
    umap_space : umap result [n_samples * n_components], numpy(cpu) format
    '''
    
    if plot_3d:
        print("3D UMAP Plotting")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(umap_space[:, 0], umap_space[:, 1], umap_space[:, 2],
                   alpha=0.1,  # 투명도 설정
                   s=1,       # 포인트 크기 설정
                   c='b')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
        ax.set_title('3D UMAP Embedding')

        # 플롯 저장 여부에 따라 저장
        if save_plot:
            plot_path = os.path.join(umap_save_dir, f'3d_umap_embedding_{n_neighbors}_{min_dist}.png')
            plt.savefig(plot_path, dpi=300)
            print(f"Saved the UMAP 3D Plot at {plot_path}")

        plt.close(fig)

    if plot_2d:
        print("2D UMAP Plotting")
        plt.figure(figsize=(10, 8))
        plt.scatter(umap_space[:, 0], umap_space[:, 1],
                    alpha=0.1,
                    s=1,
                    c='b')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('2D UMAP Embedding')

        # 플롯 저장 여부에 따라 저장
        if save_plot:
            plot_path = os.path.join(umap_save_dir, f'2d_umap_embedding_{n_neighbors}_{min_dist}.png')
            plt.savefig(plot_path, dpi=300)
            print(f"Saved the UMAP 3D Plot at {plot_path}")
        plt.close()

