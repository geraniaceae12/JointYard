import cv2
import matplotlib.pyplot as plt
import numpy as np

def load_video_frames(video_path, frame_index):
    """
    Load a specific frame from the given video file.

    Parameters:
    ----------
    video_path: str
        The path to the video file.
    frame_index: int
        The index of the frame to load.

    Returns:
    -------
    frame: array-like
        The loaded frame from the video.
    """

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    return frame

def extract_cluster_frame_indices(cluster_labels):
    cluster_indices = {}
    for label in np.unique(cluster_labels):
        cluster_indices[label] = np.where(cluster_labels == label)[0]
    #print(f"Cluster frame indices: {cluster_indices}")
    return cluster_indices

def plot_frames(frames, frames_to_show, cluster_id):
    # 한 행에 3개씩 배치
    num_rows = int(np.ceil(len(frames) / 3))
    fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    
    # 단일 행인 경우
    if num_rows == 1:
        axs = np.expand_dims(axs, axis=0)
    
    for idx, frame in enumerate(frames):
        row = idx // 3
        col = idx % 3
        axs[row, col].imshow(frame)
        axs[row, col].axis('off')
    
    # 클러스터 ID를 제목으로 설정
    fig.suptitle(f'Cluster {cluster_id}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 제목을 위한 여백 추가
    
    plt.show()
    

