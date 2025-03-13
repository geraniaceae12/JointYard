import numpy as np

def compute_vector_magnitudes(data):
    """
    Compute the magnitude of each keypoint vector.

    Args:
        data (ndarray): Input data of shape (n_frames, n_keypoints * 3), where each keypoint has (x, y, z).

    Returns:
        ndarray: Magnitudes of shape (n_frames, n_keypoints).
    """
    n_frames, n_features = data.shape
    n_keypoints = n_features // 3  # Each keypoint has x, y, z

    # Reshape into (n_frames, n_keypoints, 3)
    data_reshaped = data.reshape(n_frames, n_keypoints, 3)

    # Compute vector magnitudes for each keypoint
    magnitudes = np.linalg.norm(data_reshaped, axis=2)  # Compute sqrt(x^2 + y^2 + z^2) along the last axis

    return magnitudes

def compute_velocities(data, dt=1.0):
    """
    Compute the velocity of each keypoint.

    Args:
        data (ndarray): Input data of shape (n_frames, n_keypoints * 3), where each keypoint has (x, y, z).
        dt (float): Time interval between frames. Default is 1.0.

    Returns:
        ndarray: Velocities of shape (n_frames, n_keypoints * 3).
    """
    # Compute the finite difference along the time axis (frames)
    velocities = np.gradient(data, axis=0) / dt  # Shape: (n_frames, n_keypoints * 3)

    return velocities

def compute_direction_changes(data, fill_last="repeat"):
    """
    Compute the direction change rate (angle difference) for each keypoint vector.

    Args:
        data (ndarray): Input data of shape (n_frames, n_keypoints * 3), where each keypoint has (x, y, z).
        fill_last (str): How to handle the last frame ('repeat' to copy last value, 'zero' to fill with 0).
                         Default is 'repeat'.

    Returns:
        ndarray: Direction changes of shape (n_frames, n_keypoints).
    """
    n_frames, n_features = data.shape
    n_keypoints = n_features // 3  # Each keypoint has x, y, z

    # Reshape into (n_frames, n_keypoints, 3)
    data_reshaped = data.reshape(n_frames, n_keypoints, 3)

    # Compute normalized vectors to extract direction
    norms = np.linalg.norm(data_reshaped, axis=2, keepdims=True)
    directions = data_reshaped / (norms + 1e-8)  # Avoid division by zero

    # Compute cosine similarity between consecutive frames
    dot_products = np.einsum('ijk,ijk->ij', directions[:-1], directions[1:])  # Dot product of consecutive directions
    dot_products = np.clip(dot_products, -1.0, 1.0)  # Clip to avoid numerical issues

    # Compute angle differences (acos of dot product)
    delta_theta = np.arccos(dot_products)  # Shape: (n_frames-1, n_keypoints)

    # Expand to (n_frames, n_keypoints) by handling the last frame
    if fill_last == "repeat":
        last_row = delta_theta[-1:, :]  # Repeat last row
    elif fill_last == "zero":
        last_row = np.zeros((1, delta_theta.shape[1]))  # Add zero row
    else:
        raise ValueError("Invalid value for fill_last. Use 'repeat' or 'zero'.")
    
    delta_theta_expanded = np.vstack([delta_theta, last_row])  # Shape: (n_frames, n_keypoints)

    return delta_theta_expanded
