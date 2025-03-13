import numpy as np

def reshape_lpdata_for_pca(lpdata, keypoint_names, n_views):
    """
    Format the LP data for PCA analysis.

    Args:
        lpdata : DataFrame containing the LP data
        keypoint_names : List of keypoint names
        n_views : Number of camera views

    Returns:
        arr_reshapedlp : Reshaped array for PCA
    """
    arr_lp = lpdata.values
    n_bodypoints = len(keypoint_names) // n_views
    
    reshaped_views = []
    for view in range(n_views):
        view_data = arr_lp[:, view * n_bodypoints * 2 : (view + 1) * n_bodypoints * 2]
        reshaped_view = view_data.reshape(-1, 2)
        reshaped_views.append(reshaped_view)
    
    arr_reshapedlp = np.hstack(reshaped_views)
    
    return arr_reshapedlp

def pcadata_to_lpdata(pcadata, keypoint_names, n_views):
    """
    Convert PCA reconstructed data back to the original LP data format.

    Args:
        pcadata : PCA reconstructed data
        keypoint_names : List of keypoint names
        n_views : Number of camera views

    Returns:
        arr_pca_lpdata : Numpy array formatted as LP data
    """
    n_keypoints = len(keypoint_names)
    n_bodypoints = n_keypoints // n_views
    n_frames = len(pcadata) // n_bodypoints

    split_pcadata = np.split(pcadata, n_views, axis=1)
    reshaped_pcadata = [view.reshape(n_frames, n_bodypoints * 2) for view in split_pcadata]
    arr_pca_lpdata = np.hstack(reshaped_pcadata)
    return arr_pca_lpdata
