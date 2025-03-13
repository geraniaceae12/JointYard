import numpy as np

def calculate_position(center_coords, fixedpoints, recon_3d = True):
    """
    Calculate the Euclidean distances between the center coordinates and fixed points, 
    either in 3D or 2D space.

    Parameters:
    ----------
    center_coords : np.ndarray
        Array of shape (n_frames, 3) or (n_frames, 8) depending on whether the data is 3D or 2D.
        - For 3D, it contains (x, y, z) coordinates of the center keypoint for each frame.
        - For 2D, it contains (x1, y1, x2, y2, x3, y3, x4, y4) coordinates of the center keypoint for each frame.
        
    fixedpoints : np.ndarray
        Array of shape (n_frames, 4*3) for 3D or (n_frames, 4*4*2) for 2D.
        - For 3D, it contains (x, y, z) coordinates of 4 fixed points for each frame.
        - For 2D, it contains (x, y) coordinates of 4 fixed points, for 4 views, for each frame.
        
    recon_3d : bool, optional, default=True
        If True, the function calculates 3D Euclidean distances. 
        If False, the function calculates 2D Euclidean distances.
    
    Returns:
    -------
    np.ndarray
        Array of shape (n_frames, 4) for 3D or (n_frames, 16) for 2D, containing the distances 
        from the center keypoint to each fixed point across the specified number of views.
    """   
    n_frames = center_coords.shape[0]

    if recon_3d: 
        distances = np.zeros((n_frames, 4)) # four fixedpoints
        
        # Reshape fixedpoints to separate the 4 fixed points into (x, y, z) coordinates
        fixedpoints_reshaped = fixedpoints.reshape(n_frames, 4, 3)
        
        for frame_idx in range(n_frames):
            center_x, center_y, center_z = center_coords[frame_idx]  # Center point coordinates
            for fixed_idx in range(4):
                fixed_x, fixed_y, fixed_z = fixedpoints_reshaped[frame_idx, fixed_idx]
                
                # Calculate Euclidean distance between center and each fixed point
                distance = np.sqrt((fixed_x - center_x)**2 + (fixed_y - center_y)**2 + (fixed_z - center_z)**2)
                distances[frame_idx, fixed_idx] = distance
    else:
        distances = np.zeros((n_frames, 16))  # To store the distances for each frame and each view
            
        # Reshape fixedpoints to separate the 4 fixed points into 4 views with (x, y) coordinates
        fixedpoints_reshaped = fixedpoints.reshape(n_frames, 4, 4, 2)  # (n_frames, 4 fixed points, 4 views, 2 (x, y))
        
        # Iterate over each frame
        for frame_idx in range(n_frames):
            # Get the center keypoint coordinates (x1, y1, x2, y2, x3, y3, x4, y4)
            center_x = center_coords[frame_idx, ::2]  # x1, x2, x3, x4
            center_y = center_coords[frame_idx, 1::2]  # y1, y2, y3, y4

            # Iterate over each fixed point (4 fixed points)
            for fixed_idx in range(4):
                # Get the fixed point coordinates (x1, y1, x2, y2, x3, y3, x4, y4) for the current fixed point
                for view_idx in range(4):  # For each view (4 views)
                    fixed_x = fixedpoints_reshaped[frame_idx, fixed_idx, view_idx, 0]  # x
                    fixed_y = fixedpoints_reshaped[frame_idx, fixed_idx, view_idx, 1]  # y
                    
                    # Calculate Euclidean distance for the current view
                    distance = np.sqrt((fixed_x - center_x[view_idx])**2 + (fixed_y - center_y[view_idx])**2)
                    
                    # Store the distance in the corresponding position
                    distances[frame_idx, fixed_idx * 4 + view_idx] = distance
                
    return distances


