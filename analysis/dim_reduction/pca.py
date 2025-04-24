import os
import numpy as np
import pandas as pd
import cupy as cp
from sklearn.decomposition import PCA
from cuml.decomposition import PCA as cuPCA

def perform_pca(data, confidence=0.95, n_components=None, using_gpu = True):
    """
    Perform PCA on the data and determine the number of components needed to exceed the confidence level.

    Args:
        data : input data for PCA, n_samples * n_features, numpy(cpu)
        confidence : confidence level for explained variance

    Returns:
        pcadata : reconstructed data from PCA
        pca : PCA object fitted to the data
        principal_components : the principal components of the data
        n_components : the number of components needed to exceed the confidence level
        explained_variance_ratio : cumulative explained variance ratio
    """

    if using_gpu:
        # Change data type from numpy to cupy (GPU)
        data = cp.asarray(data)

        # Perform PCA without limiting the number of components
        N_components = n_components if n_components is not None else min(data.shape)
        pca = cuPCA(n_components= N_components)
        principal_components = pca.fit_transform(data)

        # Calculate the cumulative explained variance ratio
        explained_variance_ratio = cp.cumsum(pca.explained_variance_ratio_)

        # Determine the number of components needed to exceed the desired confidence level
        if n_components is None:
            n_components = int(cp.argmax(explained_variance_ratio >= confidence).get()) + 1
            
        # Perform PCA with the determined number of components
        pca = cuPCA(n_components=n_components)
        principal_components = pca.fit_transform(data)

        # Reconstruct the data from the principal components
        pcadata = pca.inverse_transform(principal_components)
        
        # Change data type from cupy to numpy (CPU)
        principal_components = principal_components.get()
        pcadata = pcadata.get()
        explained_variance_ratio = explained_variance_ratio.get()
            
    else: 
        # Perform PCA without limiting the number of components
        pca = PCA(n_components)
        principal_components = pca.fit_transform(data)
        
        # Calculate the cumulative explained variance ratio
        explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        
        # Determine the number of components needed to exceed the desired confidence level
        n_components = np.argmax(explained_variance_ratio >= confidence) + 1 if n_components is None else n_components
        
        # Perform PCA with the determined number of components
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(data)
        
        # Reconstruct the data from the principal components
        pcadata = pca.inverse_transform(principal_components)

    # Print the results
    print(f"Total number of PCA components, using GPU({using_gpu}): {n_components}")
    print(f"Explained variance ratio by the selected components: {explained_variance_ratio[n_components-1]:.4f}")

    return pcadata, pca, principal_components, n_components, explained_variance_ratio

def save_pca_results(arr_pca_lpdata, save_dir, explained_variance_ratio):
    """
    Save the reshaped PCA coordinates to a CSV file.

    Args:
        df_coordinates : DataFrame containing the reshaped PCA coordinates
        save_dir : directory to save the PCA results
        confidence : confidence level for explained variance
    """
    # Convert to DataFrame
    df_pca_lpdata = pd.DataFrame(arr_pca_lpdata)
    # Specify the full file path
    file_path = os.path.join(save_dir, f'pca_{explained_variance_ratio:.2f}.csv')
    
    # Save the DataFrame to a CSV file
    df_pca_lpdata.to_csv(file_path, index=False)
    print(f'PCA result saved to {file_path}')

