import numpy as np
import pandas as pd

def egocentric_alignment_3d(data, bodypoint_names, egocentric_keypoint):
    """
    Perform egocentric alignment on 3D LP data using the specified center keypoint.

    This function adjusts the positions of all keypoints relative to a chosen center keypoint
    and removes the center keypoint from the resulting data. The transformation is applied
    frame-by-frame, maintaining temporal alignment across all frames.

    Parameters:
    ----------
    data : np.ndarray
        A 2D array of shape (n_frames, n_keypoints * 3), where each row represents a frame,
        and each keypoint is represented by its (x, y, z) coordinates.
    bodypoint_names : list
        A list of keypoint names, corresponding to the keypoints in the data. The order of the names
        must match the order of keypoints in the `data` array.
    egocentric_keypoint : str
        The name of the keypoint to use as the egocentric center. All other keypoints' coordinates
        will be adjusted relative to this keypoint.

    Returns:
    -------
    tuple:
        - np.ndarray: The aligned 3D data, of shape (n_frames, (n_keypoints - 1) * 3), where the center
          keypoint has been removed and the remaining keypoints are shifted relative to it.
        - np.ndarray: The original coordinates of the center keypoint for each frame, of shape (n_frames, 3).

    Raises:
    ------
    ValueError:
        If the shape of the `data` array does not match the expected 3D format, or if the
        `center_keypoint` is not found in `bodypoint_names`.
    """
    n_frames, n_cols = data.shape
    n_bodypoints = len(bodypoint_names)

    # Validate the shape of the data array
    if n_cols != n_bodypoints * 3:
        raise ValueError("Data shape and keypoint count do not match the 3D format. Expected "
                         f"{n_bodypoints * 3} columns, but got {n_cols}.")

    # Validate the presence of the center keypoint
    try:
        center_idx = bodypoint_names.index(egocentric_keypoint)
    except ValueError:
        raise ValueError(f"Center keypoint '{egocentric_keypoint}' not found in keypoint names.")

    # Extract the center keypoint's coordinates (x, y, z) for all frames
    center_coords = data[:, center_idx * 3:(center_idx + 1) * 3]

    # Initialize a list to hold the adjusted keypoints
    aligned_data = []

    # Iterate over all keypoints and adjust their positions relative to the center keypoint
    for i, keypoint in enumerate(bodypoint_names):
        if keypoint == egocentric_keypoint:
            continue  # Skip the center keypoint
        start_idx = i * 3
        end_idx = start_idx + 3
        # Subtract the center keypoint's coordinates from the current keypoint's coordinates
        aligned_data.append(data[:, start_idx:end_idx] - center_coords)

    # Combine the adjusted coordinates of all keypoints into a single array
    aligned_data = np.hstack(aligned_data)

    return aligned_data, center_coords

def align_3dvector_to_axis(data, vector_from_name, vector_to_name, bodypoint_names):
    """
    Rotate egocentrically aligned 3D data so that the vector from `vector_from_name` to `vector_to_name`
    aligns with the global x-axis.

    Parameters
    ----------
    data : np.ndarray
        Aligned 3D data of shape (n_frames, (n_keypoints - 1) * 3), output from egocentric_alignment_3d.
    vector_from_name : str
        The name of the starting point of the vector (usually the egocentric keypoint).
    vector_to_name : str
        The name of the target point to define the direction vector.
    bodypoint_names : list
        The list of all bodypoint names before removing the egocentric keypoint.

    Returns
    -------
    np.ndarray
        Rotated aligned data of shape (n_frames, (n_keypoints - 1) * 3)
    """
    n_frames = data.shape[0]
    reduced_names = [name for name in bodypoint_names if name != vector_from_name]
    name_to_idx = {name: i for i, name in enumerate(reduced_names)}

    if vector_to_name not in name_to_idx:
        raise ValueError(f"Target keypoint '{vector_to_name}' not found in aligned data.")

    # Index of the target keypoint (after removing the center keypoint)
    to_idx = name_to_idx[vector_to_name]
    to_coords = data[:, to_idx * 3:(to_idx + 1) * 3]

    # Each frame's vector from origin (0,0,0) to the target point
    rotation_matrices = []
    for i in range(n_frames):
        vec = to_coords[i]
        if np.linalg.norm(vec) == 0:
            rot = np.eye(3)
        else:
            # Create a rotation that aligns vec to x-axis
            vec_unit = vec / np.linalg.norm(vec)
            rot_vec = np.cross(vec_unit, [1, 0, 0])
            sin_angle = np.linalg.norm(rot_vec)
            if sin_angle == 0:
                rot = np.eye(3)
            else:
                cos_angle = np.dot(vec_unit, [1, 0, 0])
                rot_axis = rot_vec / sin_angle
                angle = np.arctan2(sin_angle, cos_angle)
                rot = R.from_rotvec(rot_axis * angle).as_matrix()
        rotation_matrices.append(rot)

    # Apply rotation to each frame
    aligned_rotated_data = np.zeros_like(data)
    n_keypoints = len(reduced_names)
    for i in range(n_frames):
        frame = data[i].reshape((n_keypoints, 3))
        rotated_frame = rotation_matrices[i] @ frame.T
        aligned_rotated_data[i] = rotated_frame.T.flatten()

    return aligned_rotated_data

def egocentric_alignment_2d(data, keypoint_names, egocentric_keypoints, n_views):
    """
    Perform egocentric alignment on 2D LP data using the specified center keypoints.

    Parameters:
    ----------
    data : np.ndarray
        Array of shape (n_frames, n_keypoints * 2), containing x, y coordinates.
    keypoint_names : list
        List of keypoint names corresponding to the data (length should be 44).
    egocentric_keypoints : list
        List of keypoint names to use as the egocentric centers, one for each view (length should be n_views).
    n_views : int
        The number of views for which the alignment is performed.

    Returns:
    -------
    np.ndarray
        Aligned 2D data of shape (n_frames, (n_keypoints - n_views) * 2).
    np.ndarray
        Array of center x and y coordinates of shape (n_frames, n_views * 2).
    """
    n_frames, n_cols = data.shape
    n_keypoints = len(keypoint_names)
    
    # Ensure that the data shape matches the expected format (n_frames, n_keypoints * 2)
    if n_cols != n_keypoints * 2:
        raise ValueError("Data shape and keypoint count do not match 2D format.")
    
    # Calculate the number of columns per view (for 2D, we have 2 columns per keypoint, so cols_per_view = 2 * n_keypoints / n_views)
    cols_per_view = 2 * n_keypoints // n_views  # 22
    
    # Create an empty list to store center coordinates
    center_coords = []

    # Iterate over each view and align the keypoints
    for view_idx in range(n_views):
        # Find the index of the center keypoint for the current view
        center_keypoint = egocentric_keypoints[view_idx]  # Keypoint name like 'body_1', 'body_2', ...
        center_idx = keypoint_names.index(center_keypoint)  # Get the index of the center keypoint in keypoint_names

        # Extract the center keypoint's coordinates (x, y) for the current view
        center_x = data[:, center_idx * 2]  # x coordinate for the center keypoint
        center_y = data[:, center_idx * 2 + 1]  # y coordinate for the center keypoint

        # Add the center coordinates to the list
        center_coords.append(np.column_stack((center_x, center_y)))

        # Extract x and y coordinate columns for the current view (the columns for this view)
        view_start_idx = view_idx * cols_per_view
        view_end_idx = (view_idx + 1) * cols_per_view
        view_data = data[:, view_start_idx:view_end_idx]

        # Subtract the center keypoint's coordinates from all keypoints in the view
        # For each keypoint in the current view, subtract the center keypoint coordinates
        for i in range(0, cols_per_view, 2):  # For each keypoint (x, y pair)
            view_data[:, i] -= center_x  # Subtract center x
            view_data[:, i + 1] -= center_y  # Subtract center y

        # Update the data with the aligned view
        data[:, view_start_idx:view_end_idx] = view_data

    # Combine center coordinates from all views into a single array (n_frames, n_views * 2)
    center_coords = np.hstack(center_coords)

    # Drop the columns corresponding to the center keypoints to get the final result (n_frames * 80)
    aligned_data = np.delete(data, [keypoint_names.index(kp) * 2 for kp in egocentric_keypoints] + 
                              [keypoint_names.index(kp) * 2 + 1 for kp in egocentric_keypoints], axis=1)

    return aligned_data, center_coords  # The updated data with aligned keypoints, and the center coordinates

