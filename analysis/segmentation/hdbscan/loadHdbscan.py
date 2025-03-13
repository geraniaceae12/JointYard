import pandas as pd
import numpy as np

import pickle
import os
from cuml.cluster.hdbscan.hdbscan import HDBSCAN as cuml_HDBSCAN

def load_hdbscan_model(model_path):
    # load the saved .pkl files    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            hdbscan_model = pickle.load(f)
        print(f"Loaded HDBSCAN model from {model_path}")

        # Optionally, you can check the type of the loaded model
        if not isinstance(hdbscan_model, cuml_HDBSCAN):
            print("Warning: Loaded model is not an instance of cuml.HDBSCAN")
    else:
        raise FileNotFoundError(f"{model_path} not found.")

    return hdbscan_model

def load_cluster_labels(index_info_path, sourcefile_list=None, min_cluster_size=None):
    """
    Load specific cluster label column from an index_info.csv file with multi-index set.

    Args:
        index_info_path (str): Full path to the index_info.csv file.
        sourcefile_list (list, optional): List of specific SourceFile indices to select. Uses all if None.
        min_cluster_size (int, optional): Value of min_cluster_size for the Clusterlabel_{min_cluster_size} column.

    Returns:
        pandas.Series: Returns specific cluster label column or whole cluster label column if min_cluster_size = None.
    """
    # Load index_info.csv file with multi-index set to SourceFile, RowIndex, and VideoPath
    index_info_df = pd.read_csv(index_info_path, index_col=["SourceFile", "RowIndex", "VideoPath"])

    # If min_cluster_size is provided, select the corresponding cluster label column
    if min_cluster_size is not None:
        #cluster_label_column = f"Clusterlabel_{min_cluster_size}"
        cluster_label_column = f"Softlabel_{min_cluster_size}"
        
        if cluster_label_column not in index_info_df.columns:
            raise ValueError(f"{cluster_label_column} column does not exist in index_info.csv.")
        
        # Select only the cluster label column (as a Series)
        cluster_labels = index_info_df[cluster_label_column]
        
        # If sourcefile_list is provided, select only the corresponding indices
        if sourcefile_list is not None:
            # Select only the SourceFile indices corresponding to sourcefile_list
            cluster_labels = cluster_labels.loc[sourcefile_list]
            
        return cluster_labels
    
    # If min_cluster_size is not provided, return the entire DataFrame
    return index_info_df



