import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

class CWT:
    def __init__(self, frequencies, omega0, dt, scaler):
        self.frequencies = frequencies
        self.omega0 = omega0
        self.dt = dt
        self.scaler = scaler

    def fast_wavelet_morlet_convolution(self, data):
        """
        Calculate the Morlet wavelet transform resulting from a time series.

        Args:
            data (array-like): array (n_frames x number of features).

        Returns:
            amp (ndarray): Wavelet amplitudes (L x N x num_features).
            W (ndarray): Wavelet coefficients (complex-valued) (L x N x num_features).
        """
        # Scaling the data before cwt
        if self.scaler:
            scaler_pre = StandardScaler()
            data = scaler_pre.fit_transform(data)
            print("Scaling before CWT...")

        # Get dimensions
        L = len(self.frequencies) # frequency window number
        N = data.shape[0] # n_frames
        num_features = data.shape[1] # num_features

        # Pad zeros to ensure even length
        padded = False
        if N % 2 == 1:
            data = np.vstack((data, np.zeros((1, num_features))))
            N += 1
            padded = True

        # Initialize arrays for wavelet amplitudes and coefficients
        amp = np.zeros((L, N, num_features), dtype=np.float64)
        W = np.zeros_like(amp, dtype=np.complex128)

        for feature_index in range(num_features): # Iterate over each feature (time series)
            # Extract data for this feature
            feature_data = data[:, feature_index]

            scales = (self.omega0 + np.sqrt(2 + self.omega0**2)) / (4 * np.pi * self.frequencies) # Calculate scales for wavelet transform
            omega_vals = 2 * np.pi * np.arange(-N / 2, N / 2) / (N * self.dt) # Frequency values for FFT

            # Compute FFT
            x_hat = np.fft.fft(feature_data)
            x_hat = np.fft.fftshift(x_hat)

            idx = np.arange(N // 2 + 1, N // 2 + N + 1) # Index for slicing the frequency domain

            # Iterate over each frequency channel
            for i, freq in enumerate(self.frequencies):
                # Compute the Morlet wavelet in frequency domain
                m = self.morlet_conj_ft(-omega_vals * scales[i])

                # Convolve Morlet wavelet with the data in frequency domain
                q = np.fft.ifft(m * x_hat) * np.sqrt(scales[i])

                # Adjust the length of idx to match the length of q
                amp[i, :, feature_index] = np.abs(q) * np.pi ** -0.25 * np.exp(
                    0.25 * (self.omega0 - np.sqrt(self.omega0 ** 2 + 2)) ** 2) / np.sqrt(2 * scales[i])
                W[i, :, feature_index] = q

        # If data was padded, remove the last row from amp and W
        if padded:
            amp = amp[:, :-1, :]
            W = W[:, :-1, :]
            
        return amp, W

    def morlet_conj_ft(self, w):
        """
        Compute the Morlet wavelet in frequency domain.

        Args:
            w (array-like): Frequency values.

        Returns:
            array-like: Morlet wavelet in frequency domain.
        """
        return np.pi**(-1/4) * np.exp(-0.5 * (w - self.omega0)**2)

    def plot_cwt_separate(self, amp, save_path=None):
        """
        Plot Continuous Wavelet Transform (CWT) coefficients separately for each features.

        Args:
            amp (ndarray): Wavelet amplitudes (L x N x num_features).
            save_path (str, optional): Path to save the plot images.
        """
        L, N, num_features = amp.shape
    
        for feature_idx in range(num_features):
            amp_single_feature = np.abs(amp[:, :, feature_idx])

            fig = plt.figure(figsize=(12, 8))
            plt.imshow(np.log(amp_single_feature+1), aspect='auto', origin='lower', cmap='jet', interpolation='none')
            # plt.imshow(
            #     np.log(amp_single_feature + 1), 
            #     aspect='auto', 
            #     origin='lower', 
            #     cmap='jet', 
            #     interpolation='none',
            #     vmin=0,  # 최소값
            #     vmax=np.log(np.max(amp) + 1)  # 전체 데이터의 최대값
            # )
            plt.colorbar(label='Magnitude')
            plt.title(f"CWT Time-Frequency Representation for Feature {feature_idx}")
            plt.xlabel('Time')
            plt.ylabel('Frequency')

            plt.tight_layout()

            if save_path:
                plt.savefig(os.path.join(save_path, f"cwt_separated_feature{feature_idx}.png"))
                plt.close(fig)
            else:
                plt.show()

def cwt_filter(amp, threshold=0.9, save_path=None):
    """
    Filters out high frequency bands based on the cumulative magnitude, keeping frequencies 
    that contribute to the first 90% of the total magnitude.

    Args:
        amp (ndarray): Wavelet amplitudes (L x N x num_features), where L is the number of frequencies,
                       N is the number of time points, and num_features is the number of features.
        threshold (float): The cumulative magnitude threshold (default is 0.9).

    Returns:
        filtered_amp (ndarray): The filtered wavelet amplitudes with high frequencies removed.
        retained_frequencies (list): List of retained frequency indices for each feature.
    """
    L, N, num_features = amp.shape
    filtered_amps = []
    retained_frequencies = []  # To store the retained frequencies for each feature

    for feature_idx in range(num_features):
        # Extract the magnitude for the current feature (L x N)
        amp_feature = np.abs(amp[:, :, feature_idx])

        # Calculate the total magnitude for each frequency across all time points (L x 1)
        total_magnitude_per_freq = np.sum(amp_feature, axis=1)

        # Calculate the cumulative sum of the magnitude
        cumulative_magnitude = np.cumsum(total_magnitude_per_freq)

        # Find the threshold for the cumulative magnitude
        total_magnitude = cumulative_magnitude[-1]
        cutoff_value = total_magnitude * threshold

        # Find the index of the frequency where cumulative magnitude exceeds the threshold
        cutoff_freq_idx = np.argmax(cumulative_magnitude >= cutoff_value)

        # Store the indices of the retained frequencies for this feature
        retained_frequencies.append(np.arange(cutoff_freq_idx + 1))  # Store indices of retained frequencies

        # Filter out the frequencies higher than the threshold (L' x N)
        filtered_amp = amp_feature[:cutoff_freq_idx+1, :]

        # Store filtered amplitude for the current feature
        filtered_amps.append(filtered_amp)

        # Plot the filtered density plot
        fig=plt.figure(figsize=(12, 8))
        plt.imshow(np.log(filtered_amp + 1), aspect='auto', origin='lower', cmap='jet', interpolation='none')
        # plt.imshow(
        #     np.log(filtered_amp + 1), 
        #     aspect='auto', 
        #     origin='lower', 
        #     cmap='jet', 
        #     interpolation='none',
        #     vmin=0,  # 최소값
        #     vmax=np.log(np.max(amp) + 1)  # 전체 데이터의 최대값
        # )
        plt.colorbar(label='Magnitude')
        plt.title(f"CWT Density Plot for Feature {feature_idx} (Filtered by Cumulative Magnitude)\n cutoff_frequency is {cutoff_freq_idx}")
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        
        if save_path:
            plt.savefig(os.path.join(save_path, f"cwt_separated_feature{feature_idx}_filter.png"))
            plt.close(fig)
        else:
            plt.show()
        
    # Stack all filtered features vertically
    filtered_amp_reshaped = np.vstack(filtered_amps)  # Shape becomes (L_filtered1 + L_filtered2 + ... , N)
    return filtered_amp_reshaped, retained_frequencies