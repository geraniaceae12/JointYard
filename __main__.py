import os
import json
import pandas as pd
import numpy as np
import torch
import random
import shutil
import rmm
import cupy as cp

from datetime import datetime
from preprocess.loadConfig import ConfigLoader
from preprocess.loadLPdata import LoadLPData
from preprocess.reconstruct3D import process_groups_reconstruction
from preprocess.egocentric import egocentric_alignment_3d, egocentric_alignment_2d
from preprocess.computemotion import compute_vector_magnitudes, compute_velocities,compute_direction_changes
from preprocess.addposition import calculate_position
from preprocess.addangle import calculate_normal_vectors_per_group, calculate_angle_between_line_and_plane
from analysis.dim_reduction.pca import perform_pca, save_pca_results
from analysis.cwt import CWT, cwt_filter
from analysis.dim_reduction.umap import perform_umap, plot_umap
from analysis.embedding.vae.vae_main import vae_run
from analysis.embedding.vae.vae_visualize import extract_latent_space
from analysis.embedding.vae.vae_utils import numpy_to_tensor, load_savedmodel, load_hyperparameters
from analysis.segmentation.hdbscan.hdbscan_clustering import apply_clustering
from analysis.segmentation.hdbscan.hdbscan_confirm import load_video_frames, plot_frames, extract_cluster_frame_indices
from analysis.segmentation.hdbscan.hdbscan_medoid import calculate_cluster_medoids, plot_latent_space_with_medoids,plot_latent_space_with_medoids2

def main():
    ######## Configuration ######################################################
    print("\n***Reading the configuration file***")
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    cfg = ConfigLoader(config_path)

    # Load configuration in each process
    info_cfg = cfg.get_info()
    preprocess_cfg = cfg.get_preprocess()
    cwt_cfg = cfg.get_cwt()
    vae_cfg = cfg.get_vae()
    umap_cfg = cfg.get_umap()
    hdbscan_cfg = cfg.get_hdbscan()


    ######## Preprocess ########################################################
    if preprocess_cfg["use_saved_preprocess"]:
        print(f"\n***Loaded preprocess data from {preprocess_cfg['use_savedpreprocess_path']}***")
        denoised_lpdata = np.load(preprocess_cfg["use_savedpreprocess_path"])
        print("Loaded preprocess data. Shape:", denoised_lpdata.shape)
    else:
        print("\n***Start the Preprocess***")        
        #### Initialize Data Loader
        loader = LoadLPData(info_cfg["base_file"],info_cfg["eks_csv"],
                            info_cfg["n_keypoints"],info_cfg["keypoint_names"],
                            info_cfg["save_dir"],info_cfg["n_views"])
        lpdata = loader.concatenated_df

        #### Denoise the Posture w/ 3D Reconstruction (raw LP data)
        print("\n***Denoise the Posture w/ 3D Reconstruction***")

        grouped = lpdata.groupby("SourceFile") # Group data by source files for per-file PCA
        arr_pca_lpdata, group_explained_variance_ratios, fixedpts_all, fixedpts_group, group_indices = process_groups_reconstruction(
            grouped,
            loader.keypoint_names,
            loader.n_views,
            preprocess_cfg["pca_confidence"],
            loader.fixedpoint_paths,
            reconstruction_3d=preprocess_cfg["3d_reconstruction"])
            
        # Save PCA results
        if preprocess_cfg["save_pca_result"]:
            avg_explained_variance_ratio = np.mean(group_explained_variance_ratios)
            save_pca_results(arr_pca_lpdata, info_cfg["save_dir"], avg_explained_variance_ratio)

        denoised_lpdata = arr_pca_lpdata
        print("3D reconstruction completed. Shape:", denoised_lpdata.shape)

        #### Change the alignment: Allocentric to Egocentric
        n = loader.n_keypoints//loader.n_views #11
        bodypoint_names = [name.rsplit("_", 1)[0] for name in loader.keypoint_names[:n]]

        # Egocentric Alignment
        if preprocess_cfg["3d_reconstruction"]:
            # 3D case
            arr_ego_lpdata, center_coords = egocentric_alignment_3d(
                arr_pca_lpdata, bodypoint_names, preprocess_cfg["egocentric_keypoint"])
        else:
            # 2D case
            arr_ego_lpdata, center_coords = egocentric_alignment_2d(
                arr_pca_lpdata, loader.keypoint_names, preprocess_cfg["egocentric_keypoint"],loader.n_views)

        # Handle Egocentric Alignment print output            
        if preprocess_cfg["egocentric_alignment"]: 
            print("\n***Performing Egocentric Alignment***")
            if preprocess_cfg["3d_reconstruction"]:
                print("Egocentric Alignment completed for 3D data. Shape:", arr_ego_lpdata.shape)
            else: 
                print("Egocentric Alignment completed for 2D data. Shape:", arr_ego_lpdata.shape)
            denoised_lpdata = arr_ego_lpdata

        #### Compute velocity: coords to velocities
        if preprocess_cfg["compute_velocity"]:
            print("\n***Converting into velocities***")
            arr_v_lpdata = compute_velocities(denoised_lpdata)
            denoised_lpdata = arr_v_lpdata
            print("Completion of computing velocities. Shape:", denoised_lpdata.shape)

        #### Add the Position (distance) 
        # Add position data if configured
        if preprocess_cfg["add_position"]:
            print("\n***Add the position***")
            arr_position = calculate_position(center_coords, fixedpts_all, preprocess_cfg["3d_reconstruction"])
            denoised_lpdata = np.hstack((denoised_lpdata, arr_position))
            print("Add position completed. Shape:", denoised_lpdata.shape)   
        
        #### Add the angle
        if preprocess_cfg["add_angle"]:
            print("\n***Add the angle***")
            normal_vectors = calculate_normal_vectors_per_group(group_indices, fixedpts_group)
            theta_list = []
            for keypoints in preprocess_cfg["line_keypoints"]:
                theta = calculate_angle_between_line_and_plane(
                    bodypoint_names, arr_pca_lpdata, normal_vectors, keypoints)
                theta_list.append(theta)
            theta_array = np.column_stack(theta_list)
            denoised_lpdata = np.hstack((denoised_lpdata, theta_array))
            print("Add angle completed. Shape:", denoised_lpdata.shape)
        
        preprocess_path = os.path.join(
            info_cfg["save_dir"],
            f'preprocess_{preprocess_cfg["3d_reconstruction"]}_{preprocess_cfg["egocentric_alignment"]}'
            f'_{preprocess_cfg["compute_velocity"]}'
            f'_{preprocess_cfg["add_position"]}_{preprocess_cfg["add_angle"]}.npy'
        )

        np.save(preprocess_path, denoised_lpdata)
        print(f"Preprocess result saved to {preprocess_path}. Shape:", denoised_lpdata.shape)

        # #### Denoising the input of CWT
        # print(f"Before denoising :{denoised_lpdata.shape}") 
        # _, _ , denoised_lpdata,_, _ = perform_pca(denoised_lpdata, preprocess_cfg["pca_confidence"], using_gpu=False)
        # print(f"After denoising, the cwt input data shape is :{denoised_lpdata.shape}") 


      
        
    ######## Continuous Wavelet Transform ##########################################
    if cwt_cfg["use_saved_cwt"]:
        print(f"\n***Loaded precomputed CWT data from {cwt_cfg['use_savedcwt_path']}***")
        amp_reshaped = np.load(cwt_cfg["use_savedcwt_path"])
        print("Loaded CWT data. Shape.T:", amp_reshaped.T.shape)
    else: 
        print("\n***Performing Continuous wavelet transform***")
        # Create an instance of the CWT class
        cwt = CWT(cwt_cfg["frequencies"], cwt_cfg["cwt_omega0"], cwt_cfg["cwt_dt"], cwt_cfg["scaler"])  

        # Perform the wavelet transform
        amp, W = cwt.fast_wavelet_morlet_convolution(denoised_lpdata)

        # Save the cwt result 
        L, N, num_features = amp.shape # (50,17546,104)
        amp_reshaped = np.reshape(amp, (L*num_features, N)) # (50*104, 17546)

        cwt_dir = os.path.join(info_cfg["save_dir"], 'cwt')
        os.makedirs(cwt_dir, exist_ok=True)

        cwt_path = os.path.join(cwt_dir,f'cwt_{cwt_cfg["frequencies_start"]}_{cwt_cfg["frequencies_end"]}_{cwt_cfg["frequencies_step"]}_amp_reshaped.npy')
        np.save(cwt_path, amp_reshaped)
        print(f"CWT result saved to {cwt_path}. Shape.T:", amp_reshaped.T.shape)

        # Plot and save the CWT results
        if cwt_cfg["cwt_plot_separate"]:
            cwt.plot_cwt_separate(amp, save_path=cwt_dir)
            print("Saved CWT plots")
        
        if cwt_cfg['cwt_filtering']:
            amp_reshaped, retained_freq = cwt_filter(amp,cwt_cfg['freq_removal_threshold'],cwt_dir)
            cwt_path = os.path.join(cwt_dir,f'cwt_{cwt_cfg["frequencies_start"]}_{cwt_cfg["frequencies_end"]}_{cwt_cfg["frequencies_step"]}_amp_reshaped_filter.npy')
            cwt_path2 = os.path.join(cwt_dir,f'cwt_{cwt_cfg["frequencies_start"]}_{cwt_cfg["frequencies_end"]}_{cwt_cfg["frequencies_step"]}_filterinfo.npy')
            np.save(cwt_path, amp_reshaped)
            np.savez(cwt_path2, *retained_freq)
            print(f"Complete filtering the CWT data before VAE. saved to {cwt_path}")
            print('Shape.T:', amp_reshaped.T.shape)
        
    # # Denoising before performing vae
    # print("\n***Denoising the input of VAE")
    # amp_reshaped,_,_,_,_ = perform_pca(amp_reshaped.T, confidence = preprocess_cfg["pca_confidence"])
    # amp_reshaped = amp_reshaped.T

    ######## Embedding: Variational AutoEncoder ##################################
    if not umap_cfg["use_saved_umap"]:   
        if vae_cfg["use_saved_latentspace"]:
            print(f"\n***Loaded VAE latentspace data from {vae_cfg['use_savedlatent_path']}***")
            latent_space = np.load(vae_cfg['use_savedlatent_path'])
            print("Loaded VAE latent space. Shape:",latent_space.shape)
        else:
            if vae_cfg["use_saved_vae"]:
                print("\n***Using already existed vae_model***")
                amp_reshaped = numpy_to_tensor(amp_reshaped) # (17546,50*104) = (n_samples*n_features)
                vae_trained_model, device = load_savedmodel(vae_cfg["vaemodel_type"],vae_cfg["use_saved_vaemodel_path"],
                                                            vae_cfg["use_saved_vaehyparam_path"], amp_reshaped.shape[1])
            else:
                print("\n***Start training of vae_model***") 
                vae_trained_model, device = vae_run(config_path, amp_reshaped)
                # vae_trained_model = vae_run(config_path) # vae_cfg["data_path"]에 원하는 Data(.npy) 경로 넣으면 해당 데이터로 vae run
                amp_reshaped = numpy_to_tensor(amp_reshaped)

            # Extract latent space from VAE model and save it
            latent_space = extract_latent_space(vae_trained_model, amp_reshaped, device) #(n_samples*n_features)
            np.save(os.path.join(info_cfg["save_dir"],'vae_latentspace.npy'),latent_space)
            print("Extract VAE latent space. Shape:",latent_space.shape)
            del vae_trained_model

        del amp_reshaped

    ######## Dim Reduction: UMAP ################################################
    if umap_cfg["use_saved_umap"]:
        print(f"\n***Loaded precomputed UMAP data from {umap_cfg['use_savedumap_path']}***")
        umap_space = np.load(umap_cfg["use_savedumap_path"])
    else: 
        umap_space, umap_save_dir = perform_umap(latent_space, umap_cfg["n_neighbors"],
                                        umap_cfg["min_dist"],umap_cfg["n_components"],
                                        info_cfg["save_dir"])
        plot_umap(umap_space, umap_cfg['n_neighbors'], umap_cfg['min_dist'], umap_cfg['plot_2d'],
                umap_cfg['plot_3d'], umap_cfg['save_plot'], umap_save_dir)

    torch.cuda.empty_cache()

    ######## Clustering: HDBSCAN ################################################
    min_cluster_sizes = hdbscan_cfg["min_cluster_sizes"]
    for min_cluster_size in min_cluster_sizes:
        print(f"\n***Running HDBSCAN with min_cluster_size = {min_cluster_size}***")

        # HDBSCAN
        try:
            # reduced_data, cluster_labels, soft_labels, max_probs, index_info_df = apply_clustering(
            #     umap_space, hdbscan_cfg, min_cluster_size, info_cfg["save_dir"])
            reduced_data, cluster_labels, soft_labels, max_probs, index_info_df = apply_clustering(
                latent_space, hdbscan_cfg, min_cluster_size, info_cfg["save_dir"])
            print("Clustering completed successfully.")
        except Exception as e:
            print(f"HDBSCAN clustering failed: {e}")

        # GPU Memory Cleanup
        cp.get_default_memory_pool().free_all_blocks()

        #Calculate Cluster Centroid and Save it
        medoids, medoid_indices, select_points_indices = calculate_cluster_medoids(reduced_data, cluster_labels)
        medoid_save_dir= os.path.join(info_cfg["save_dir"],'hdbscan/medoid')
        os.makedirs(medoid_save_dir, exist_ok=True)
        np.save(os.path.join(medoid_save_dir, 'medoid_indices.npy'),np.array(medoid_indices))

        # Plot latent space with centroids and frames
        ani_save_dir = os.path.join(info_cfg["save_dir"],'hdbscan/ani')
        os.makedirs(ani_save_dir, exist_ok=True)

        plot_latent_space_with_medoids(
            reduced_data,
            cluster_labels,
            medoids,
            select_points_indices,
            index_info_df,
            ani_save_dir,
            f'{hdbscan_cfg["dim_reduction_type"]}_{min_cluster_size}',
            hdbscan_cfg["frames_to_show_per_centroid"],
            hdbscan_cfg["selection_mode"],
            plot_3d=hdbscan_cfg["plot_3d"],
            plot_show=hdbscan_cfg["plot_show"],
            save_ani=hdbscan_cfg["save_ani"]
        )
        
        plot_latent_space_with_medoids2(
        umap_space,
        cluster_labels,
        medoid_indices,
        select_points_indices,
        index_info_df,
        ani_save_dir,
        f'HDBSCAN_{hdbscan_cfg["method"]}_{min_cluster_size}',
        hdbscan_cfg["frames_to_show_per_medoid"],
        hdbscan_cfg["selection_mode"],
        plot_3d=hdbscan_cfg["plot_3d"],
        plot_show=hdbscan_cfg["plot_show"],
        save_ani=hdbscan_cfg["save_ani"]
    )
   
if __name__ == "__main__":
    main()