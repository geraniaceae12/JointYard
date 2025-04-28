import json
import os
import shutil
import numpy as np

from datetime import datetime

class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = None
        self.load_configuration()
        self.copy_config_file() 

    def load_configuration(self):
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

    def copy_config_file(self):
        now = datetime.now()
        timestamp = now.strftime("%y%m%d_%H%M%S")
        save_dir = self.config.get('info', {}).get('save_dir', '.')
        copy_filename = f"{timestamp}_config.json"
        copy_filepath = os.path.join(save_dir, copy_filename)
        if not os.path.exists(save_dir):  # Ensure the save directory exists
            os.makedirs(save_dir)
        shutil.copy(self.config_path, copy_filepath)

    # Methods that return the required data for each part to the dictionary
    def get_info(self):
        info = self.config.get('info', {})
        return {
            'base_file': info.get('base_file', None),
            'eks_csv': info.get('eks_csv', None),
            'save_dir': info.get('save_dir'),
            
            'n_views': info.get('n_views'),
            'n_keypoints': info.get('n_keypoints'),
            'keypoint_names': info.get('keypoint_names', None)

        }

    def get_preprocess(self):
        preprocess = self.config.get('preprocess', {})
        return {
            'use_saved_preprocess': preprocess.get('use_saved_preprocess',False),
            'use_savedpreprocess_path': preprocess.get('use_savedpreprocess_path', None),

            'pca_confidence': preprocess.get('pca_confidence', 0.95),
            'save_pca_result': preprocess.get('save_pca_result', False),

            '3d_reconstruction': preprocess.get('3d_reconstruction', True),

            'egocentric_alignment': preprocess.get('egocentric_alignment', False),
            'egocentric_keypoint': preprocess.get('egocentric_keypoint', None), 

            'compute_velocity': preprocess.get('compute_velocity', False),

            'add_position': preprocess.get('add_position', False),

            'add_angle': preprocess.get('add_angle', False),
            'line_keypoints': preprocess.get('line_keypoints', [])
        }
    
    def get_cwt(self):
        cwt = self.config.get('cwt', {})
        frequencies_start = cwt.get('cwt_frequencies_start', 1)
        frequencies_end = cwt.get('cwt_frequencies_end', 50)
        frequencies_step = cwt.get('cwt_frequencies_step', 1)
        frequencies = np.arange(frequencies_start, frequencies_end + 1, frequencies_step)

        return {
            'use_saved_cwt': cwt.get('use_saved_cwt', False),
            'use_savedcwt_path': cwt.get('use_savedcwt_path', None),

            'scaler': cwt.get('scaler', True),
            
            'frequencies_start': frequencies_start,
            'frequencies_end': frequencies_end,
            'frequencies_step': frequencies_step,
            'frequencies': frequencies,
            'cwt_omega0': cwt.get('cwt_omega0', 6),
            'cwt_dt': cwt.get('cwt_dt', 0.01),
            'cwt_plot_separate': cwt.get('cwt_plot_separate', False),

            'cwt_filtering': cwt.get('cwt_filtering', False),
            'freq_removal_threshold': cwt.get('freq_removal_threshold', 0.9)
        }
    
    def get_umap(self):
        umap = self.config.get('umap', {})
        return {
            'use_saved_umap': umap.get('use_saved_umap', False),
            'use_savedumap_path': umap.get('use_savedumap_path', False),
            
            'n_neighbors': umap.get('n_neighbors', 10),
            'min_dist': umap.get('min_dist', 0.1),
            'n_components': umap.get('n_components', 10),
            'plot_2d': umap.get('plot_2d', False),
            'plot_3d': umap.get('plot_3d', False),
            'save_plot': umap.get('save_plot', False)
        }
    
    def get_vae(self):
        vae = self.config.get('vae', {})
        return {
            'use_saved_latentspace': vae.get('use_saved_latentspace', False),
            'use_savedlatent_path': vae.get('use_savedlatent_path', False),
            
            'use_saved_vae': vae.get('use_saved_vae', False),
            'use_saved_vaemodel_path': vae.get('use_savedvae_path', False),
            'use_saved_vaehyparam_path': vae.get('use_savedvaehyparam_path', False),

            'optuna_db_path': vae.get('optuna_db_path', False),
            'optuna_study_name': vae.get('optuna_study_name', False),
            'optuna_checkpoint': vae.get('optuna_checkpoint', False),

            'scaler': vae.get('scaler', True),

            'vaemodel_type': vae.get('vaemodel_type', "deepvae3"),
            'test_split': vae.get('test_split', 0.2),

            'beta': vae.get('beta_range'),
            'hidden_dim_range': vae.get('hidden_dim_range'),
            'latent_dim_range': vae.get('latent_dim_range'),
            'epochs_range': vae.get('epochs'),
            'batch_size_options': vae.get('batch_size_options'),
            'learning_rate_range': vae.get('learning_rate_range'),
            'optuna_n_trials': vae.get('optuna_n_trials'),
            'patience': vae.get('patience'),        

        }

    def get_hdbscan(self):
        hdbscan = self.config.get('hdbscan', {})
        return {
            'run_gpu': hdbscan.get('run_gpu', True),
            'min_cluster_sizes': hdbscan.get('min_cluster_sizes'),
            'min_samples': hdbscan.get('min_samples', None),
            'method': hdbscan.get('method','eom'),

            'noise_method': hdbscan.get('noise_method', 'prob_threshold'),
            'prob_threshold': hdbscan.get('prob_threshold', 0.5),
            'std_threshold': hdbscan.get('std_threshold', 3),

            'dim_reduction_type': hdbscan.get('dim_reduction_type', "pca"),
            'plot_2d': hdbscan.get('plot_2d', False),
            'plot_3d': hdbscan.get('plot_3d', False),
            'plot_show': hdbscan.get('plot_show', True),
            'save_plot': hdbscan.get('save_plot', False),
            'save_plotformat': hdbscan.get('save_plotformat', "png"),
                                              
            'frames_to_show_per_medoid': hdbscan.get('frames_to_show_per_medoid', 6),
            'selection_mode': hdbscan.get('selection_mode', "closest"),        
            'save_ani': hdbscan.get('save_ani', False)
        }

