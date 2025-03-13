import numpy as np
import pandas as pd
from analysis.dim_reduction.pca import perform_pca
from preprocess.reshapedata import reshape_lpdata_for_pca, pcadata_to_lpdata

def load_fixedpoints(excel_path):
    """
    Load the fixed points from an Excel file, which contains coordinates for keypoints in multiple views.

    Args:
        excel_path (str): Path to the Excel file containing the fixed points.

    Returns:
        np.ndarray: An array of shape (n_keypoints, 2 * n_views) containing the x and y coordinates of keypoints across all views.
    """
    # df_fixedpoints = pd.read_excel(excel_path, header=None, usecols=range(8), skiprows=10, nrows=4) # 4(view)*8(keypoint * (x,y))
    df_fixedpoints = pd.read_excel(excel_path, header=None, usecols=range(8), skiprows=5, nrows=4) # 4(view)*8(keypoint * (x,y))
    
    n_views = df_fixedpoints.shape[0] # 4
    n_keypoints = df_fixedpoints.shape[1]//2 # 4
    
    reorganized_data = []

    for kp_idx in range(n_keypoints):
        # 현재 키포인트에 해당하는 x, y 좌표들
        kp_x = df_fixedpoints.iloc[:, kp_idx * 2]  # x좌표
        kp_y = df_fixedpoints.iloc[:, kp_idx * 2 + 1]  # y좌표

        # x, y 좌표를 한 줄로 이어서 재구성
        kp_row = np.hstack([kp_x.values, kp_y.values])
        reorganized_data.append(kp_row)
    
    return np.array(reorganized_data)

def reconstruct_3d(arr_reshapedlp, keypoint_names, n_views, pca_confidence):
    """
    Perform PCA with 3D reconstruction for the given LP data.

    Parameters:
    ----------
    arr_reshapedlp : np.ndarray
        Reshaped LP data ready for PCA.
    keypoint_names : list
        Names of keypoints in the data.
    n_views : int
        Number of views in the LP data.
    pca_confidence : float
        Confidence level for selecting PCA components.

    Returns:
    -------
    tuple: (PCA model, np.ndarray, float)
        - `pca`: The fitted PCA model.
        - `arr_pca_3d`: The LP data reconstructed into a 3D format after PCA, reshaped based on the number of views.
        - `explained_variance_ratio[n_components - 1]`: The accumulated explained variance ratio.
    """
    # Perform PCA with 3 principal components
    print("Performing PCA with 3D reconstruction...")
    _, pca, principal_components, n_components, explained_variance_ratio = perform_pca(
        arr_reshapedlp, pca_confidence, n_components=3)

    # Reconstruct data into 3D format
    n_bodypoints = len(keypoint_names) // n_views
    arr_pca_3d = principal_components.reshape(
        len(principal_components) // n_bodypoints, 3 * n_bodypoints
    )

    return pca, arr_pca_3d, principal_components, explained_variance_ratio[n_components - 1]


def denoise_posture(group, keypoint_names, n_views, pca_confidence, reconstruction_3d=False):
    """
    Denoise LP data using PCA, with optional 3D reconstruction.

    Parameters:
    ----------
    group : pd.DataFrame
        A single group of LP data to process (e.g., grouped by a source file).
    keypoint_names : list
        Names of keypoints in the data.
    n_views : int
        Number of views in the LP data.
    pca_confidence : float
        Confidence level for selecting PCA components.
    reconstruction_3d : bool, optional
        Whether to perform 3D reconstruction. Default is False.

    Returns:
    -------
    tuple: (PCA model, np.ndarray, float)
        - `pca`: The fitted PCA model.
        - `denoised_data` or `arr_pca_3d`: Denoised data (if no 3D reconstruction) or 3D reconstructed data (if `reconstruction_3d=True`).
        - `explained_variance_ratio`: The explained variance ratio for the selected PCA components.
    """
    # Reshape data for PCA
    arr_reshapedlp = reshape_lpdata_for_pca(group, keypoint_names, n_views)

    if reconstruction_3d:
        # Perform 3D reconstruction
        return reconstruct_3d(arr_reshapedlp, keypoint_names, n_views, pca_confidence)
    else:
        # Perform standard PCA denoising
        print("Performing standard PCA posture denoising...")
        pcadata, pca, principle_components, n_components, explained_variance_ratio = perform_pca(
            arr_reshapedlp, pca_confidence
        )
        # Convert PCA results back to the original LP data format
        denoised_data = pcadata_to_lpdata(pcadata, keypoint_names, n_views)
        return pca, denoised_data, principle_components, explained_variance_ratio[n_components - 1]

# def transform_pca_to_real(pca_fixpts, pca_coords):
#     """
#     Converts PCA-based 3D coordinates to physical 3D coordinates using affine transformation.
#     Includes Z-axis correction based on actual fixed points.

#     Parameters:
#         pca_fixpts (numpy.ndarray): PCA-based 3D coordinates of fixed points (4x3 matrix).
#         pca_coords (numpy.ndarray): PCA-based 3D coordinates to transform (Nx3 matrix).

#     Returns:
#         numpy.ndarray: Transformed physical 3D coordinates (Nx3 matrix).
#     """
#     # Define physical 3D coordinates of the fixed points (4x3 matrix)
#     real_fixpts = np.array([
#         [0.0, 0.0, 0.0],   # Fixed point 1
#         [0.0, 40.0, 0.0],  # Fixed point 2
#         [30.0, 0.0, 0.0],  # Fixed point 3
#         [30.0, 40.0, 0.0]  # Fixed point 4
#     ])

#     # Step 1: Compute affine transformation matrix (same as before)
#     pca_fixpts_h = np.hstack([pca_fixpts, np.ones((pca_fixpts.shape[0], 1))])
#     T, _, _, _ = np.linalg.lstsq(pca_fixpts_h, real_fixpts, rcond=None)
#     T = T.T  # Transpose to get a 3x4 matrix

#     # Step 2: Convert PCA coordinates into physical coordinates using affine transformation
#     pca_coords_h = np.hstack([pca_coords, np.ones((pca_coords.shape[0], 1))])
#     real_coords = np.dot(pca_coords_h, T.T)

#     # Step 3: Apply Z-axis correction
#     # Assuming that the real_fixpts' Z values are known and fixed (e.g., 0.0 for all fixed points)
#     # Calculate the average Z-value difference between the real and PCA coordinates
#     z_diff = np.mean(real_fixpts[:, 2] - pca_fixpts[:, 2])  # Average Z difference

#     # Apply Z correction to the transformed coordinates
#     real_coords[:, 2] += z_diff  # Add the Z-axis difference to the transformed coordinates

#     return real_coords

# def transform_pca_to_real(pca_fixpts, pca_coords):
#     """
#     Converts PCA-based 3D coordinates to physical 3D coordinates using affine transformation.

#     Parameters:
#         pca_fixpts (numpy.ndarray): PCA-based 3D coordinates of fixed points (4x3 matrix).
#         pca_coords (numpy.ndarray): PCA-based 3D coordinates to transform (Nx3 matrix).

#     Returns:
#         numpy.ndarray: Transformed physical 3D coordinates (Nx3 matrix).
#     """
#     # 1. Define physical 3D coordinates of the fixed points (4x3 matrix)
#     real_fixpts = np.array([
#         [0.0, 0.0, 0.0],   # Fixed point 1
#         [0.0, 40.0, 0.0],  # Fixed point 2
#         [30.0, 0.0, 0.0],  # Fixed point 3
#         [30.0, 40.0, 0.0]  # Fixed point 4
#     ])
#     print(pca_fixpts)
#     print(real_fixpts)
#     # 2. Add an additional column of ones to PCA-based fixed points (to make it 4x4 matrix)
#     pca_fixpts_h = np.hstack([pca_fixpts, np.ones((pca_fixpts.shape[0], 1))])

#     # 3. Use least squares method to estimate the affine transformation matrix (T)
#     T, _, _, _ = np.linalg.lstsq(pca_fixpts_h, real_fixpts, rcond=None)
#     T = T.T  # Transpose the matrix to shape it as 3x4
#     print(T)
#     # 4. Add an additional column of ones to PCA coordinates (to make it Nx4 matrix)
#     pca_coords_h = np.hstack([pca_coords, np.ones((pca_coords.shape[0], 1))])
    
#     # 5. Apply the affine transformation to convert PCA coordinates to physical coordinates
#     real_coords = np.dot(pca_coords_h, T.T)
#     print(real_coords[0])
#     # 6. Apply Z-axis adjustment factor to real coordinates (only to Z-values)
#     # Calculate the scaling factor for the Z-axis
#     z_factor = np.mean(pca_fixpts[:, 2]) / np.mean(real_fixpts[:, 2]) if np.mean(real_fixpts[:, 2]) != 0 else 1
#     real_coords[:, 2] *= z_factor
#     print(real_coords[0])
#     return real_coords


def process_groups_reconstruction(grouped_data, keypoint_names, n_views,
                                  pca_confidence, excel_paths, reconstruction_3d=False):
    """
    Process multiple groups of LP data, applying PCA or 3D reconstruction to each group.

    Parameters:
    ----------
    grouped_data : pd.core.groupby.DataFrameGroupBy
        Grouped LP data to process (e.g., grouped by SourceFile).
    keypoint_names : list
        Names of keypoints in the data.
    n_views : int
        Number of views in the LP data.
    pca_confidence : float
        Confidence level for selecting PCA components (i.e., the proportion of variance to be explained).
    excel_paths : list of str
        Paths to Excel files containing fixed points data for each group.
    reconstruction_3d : bool, optional
        Whether to perform 3D reconstruction (default is False).

    Returns:
    -------
    tuple: (np.ndarray, list, np.ndarray, np.ndarray)
        - `combined_data`: Combined processed LP data across all groups.
        - `explained_variance_ratios`: List of explained variance ratios for each group.
        - `fixedpts_all`: Fixed points data from all groups transformed by PCA.
        - `fixedpts_groups`: Fixed points data from each group transformed by PCA.
    """
    group_results = []  # To store processed results for each group
    explained_variance_ratios = []  # To store explained variance ratios for each group
    fixedpts_pca = []
    fixedpts_group = []
    group_indices = []

    for idx, (sourcefile, group) in enumerate(grouped_data):
        print(f"\nProcessing SourceFile: {sourcefile}")

        # Perform denoising or 3D reconstruction for the current group
        pca, processed_data, principle_components, explained_variance_ratio = denoise_posture(
            group, keypoint_names, n_views, pca_confidence, reconstruction_3d
        )
        
        # Load fixedpoints and apply PCA
        fixedpoints = load_fixedpoints(excel_paths[idx]) # 4(keypoints) * 8(view*(x,y))
        pca_transformed = pca.fit_transform(fixedpoints).reshape(1, -1)  # 3d (1, 12) or 2d (1, 32)
        fixedpts_group.append(pca_transformed)

        # Repeat the PCA result to match the number of rows in processed_data
        repeated_pca = np.tile(pca_transformed, (processed_data.shape[0], 1))  # Repeat for each row of processed_data
        fixedpts_pca.append(repeated_pca)

        # # Align the pca axis into real world
        # if reconstruction_3d:
        #     pca_fixpts = pca_transformed.reshape(4,3)
        #     pca_coords = principle_components # N * 3
        #     real_coords = transform_pca_to_real(pca_fixpts, pca_coords)
        #     processed_data = real_coords.reshape(processed_data.shape[0],processed_data.shape[1])
        
        # Store results
        group_results.append(processed_data)
        explained_variance_ratios.append(explained_variance_ratio)

        # Assign group number to each frame
        group_indices.extend([idx] * processed_data.shape[0])

    # Combine processed results from all groups
    combined_data = np.vstack(group_results)
    fixedpts_groups = np.vstack(fixedpts_group)
    fixedpts_all = np.vstack(fixedpts_pca)
    group_indices = np.array(group_indices)

    return combined_data, explained_variance_ratios, fixedpts_all, fixedpts_groups, group_indices
