import numpy as np

def calculate_normal_vectors_per_group(group_indices, fixed_points):
    """
    Calculate the normal vectors for each group and expand them to match the frame count.

    Parameters:
    ----------
    group_indices : np.ndarray
        Array of shape (n_frames,) where each frame has a group index (e.g., [0, 0, 1, 1, 1, 2, 2, ...]).
    fixed_points : list of np.ndarray
        List where each element corresponds to a group's fixed points (4 points, shape (4, 3)).
        Each fixed points array contains 3D coordinates of 4 points.

    Returns:
    -------
    np.ndarray
        `normal_vectors` of shape (n_frames, 3) where each frame has the normal vector for its group.
    """
    unique_groups = np.unique(group_indices)
    group_normals = {}

    # Step 1: Calculate the normal vector for each group
    for group in unique_groups:
        points = fixed_points[group].reshape(4,3)
        if points.shape != (4, 3):
            raise ValueError(f"Each group's fixed points must have shape (4, 3), but got {points.shape} for group {group}.")

        # Calculate two vectors on the plane
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]

        # Compute the cross product to find the normal vector
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)  # Normalize
        group_normals[group] = normal

    # Step 2: Expand normal vectors to match the frame count
    normal_vectors = np.zeros((len(group_indices), 3))
    for group in unique_groups:
        normal = group_normals[group]
        normal_vectors[group_indices == group] = normal  # Broadcast to frames belonging to the group

    return normal_vectors


def calculate_angle_between_line_and_plane(bodypoint_names, data, normal_vectors, keypoints_to_check):
    """
    Calculate the angle between the line segment defined by two keypoints and the plane defined by the normal vector.
    
    Parameters:
    ----------
    bodypoint_names : list
        List of bodypoint names (e.g., ["nose","l_ear","r_ear","body","l_fft","r_fft","l_hft","r_hft","top_tail","mid_tail","end_tail"]).
    data : np.ndarray
        n_frames x (bodypoint * (x, y, z)) array of 3D coordinates.
    normal_vectors : np.ndarray
        n_frames x 3 array of normal vectors for each frame.
    keypoints_to_check : list
        List of two keypoint names to calculate the angle between them (e.g., ['nose', 'top_tail']).
        
    Returns:
    -------
    np.ndarray
        Angles (in degrees) between the line segment and the plane for each frame.
    """
    # 인덱스를 기반으로 키포인트 좌표 추출
    keypoint_indices = [bodypoint_names.index(keypoint) * 3 for keypoint in keypoints_to_check]
    
    # 키포인트 좌표 추출 (각 프레임에 대해)
    keypoints_3d = [data[:, idx:idx + 3] for idx in keypoint_indices]  # [(n_frames, 3), (n_frames, 3)]
    
    # 두 키포인트 사이의 선분 계산
    line_segment = keypoints_3d[1] - keypoints_3d[0]  # (n_frames, 3)
    
    # 법선 벡터가 프레임별로 주어졌는지 확인
    if normal_vectors.shape[0] != data.shape[0] or normal_vectors.shape[1] != 3:
        raise ValueError("`normal_vectors` must be of shape (n_frames, 3).")
    
    # 선분 벡터와 법선 벡터 간의 각도 계산
    #dot_product = np.sum(line_segment * normal_vectors, axis=1)  # (n_frames,)
    dot_product = np.abs(np.sum(line_segment * normal_vectors, axis=1))  # 절대값으로 변경

    line_segment_magnitude = np.linalg.norm(line_segment, axis=1)  # (n_frames,)
    normal_vector_magnitude = np.linalg.norm(normal_vectors, axis=1)  # (n_frames,)

    # cos(θ) 계산
    cos_theta = dot_product / (line_segment_magnitude * normal_vector_magnitude)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # acos의 입력 범위를 -1, 1로 제한

    # 각도 계산 (radians 단위)
    theta_radians = np.arccos(cos_theta)
    
    # 각도를 degree로 변환
    theta_degrees = np.degrees(theta_radians)  # (n_frames,)

    return theta_degrees
