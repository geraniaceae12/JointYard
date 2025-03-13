import os
import pandas as pd

class LoadLPData:
    """
    A class to load and format LP data, and concatenate multiple files based on an Excel file.

    -------
    Methods
    -------
    __init__(base_file, eks_csv, n_keypoints, save_dir, n_views):
        Initialize the class and concatenate multiple CSV files.

    concat_csv_files(base_file, eks_csv):
        Concatenate multiple CSV files vertically while tracking source information using MultiIndex.

    load_from_excel(base_file):
        Load CSV paths from an Excel file.

    format_lpdata(df_rawlp, keypoint_names, model_name):
        Format LP data by extracting x, y coordinates and excluding likelihood columns.

    save_index_info(concatenated_df, csv_paths, video_paths, fixedpoint_paths, save_dir):
        Save MultiIndex information as a separate CSV file.

    -------
    Attributes
    -------
    concatenated_df: pd.DataFrame
        The concatenated LP data from multiple CSV files.
        Example structure:
            SourceFile    RowIndex    Nose_x    Nose_y    Ear_x    Ear_y 
            file1.csv     0           100.0     200.0     300.0    400.0
            file1.csv     1           105.0     205.0     305.0    405.0
            file2.csv     0           110.0     210.0     310.0    410.0
            file2.csv     1           115.0     215.0     315.0    415.0

    csv_paths: list
        List of CSV file paths extracted from the Excel file.

    video_paths: list
        List of video file paths corresponding to each CSV file.

    fixedpoint_paths: list or None
        List of fixed-point file paths corresponding to each CSV file, or None if not provided.
    """
    def __init__(self, base_file, eks_csv=False, n_keypoints=None,
                 keypoints_name=None, save_dir=None, n_views=4):
        """
        Initialize the class and load LP data from the given base file.

        Parameters:
        ----------
        base_file: str
            Path to the Excel file containing paths to the CSV files.
        eks_csv: bool, optional
            Whether to exclude the last n_keypoints columns. Default is False.
        n_keypoints: int, optional
            Number of keypoints to retain. If eks_csv is True, it drops the last n_keypoints columns.
        keypoints_name: list, optional
            List of keypoint names to use. Extracted automatically if None.
        save_dir: str, optional
            Directory to save processed files. Defaults to the current working directory.
        n_views: int, optional
            Number of views in the LP data. Default is 4.
        """
        self.n_keypoints = n_keypoints
        self.n_views = n_views
        self.keypoint_names = keypoints_name
        self.model_name = None
        self.save_dir = save_dir if save_dir else os.getcwd()
        os.makedirs(self.save_dir, exist_ok=True)

        # Automatically concatenate all CSV files listed in the base_file upon initialization
        self.concatenated_df, self.csv_paths, self.video_paths, self.fixedpoint_paths = self.concat_csv_files(base_file, eks_csv)
        
        # Save the index information into a separate CSV file
        self.save_index_info(self.concatenated_df, self.csv_paths, self.video_paths, self.fixedpoint_paths, self.save_dir)

    def concat_csv_files(self, base_file, eks_csv):
        """
        Concatenate multiple CSV files based on paths from an Excel file,
        and return a DataFrame with MultiIndex indicating source file paths.

        Parameters:
        ----------
        base_file: str
            Path to the Excel file containing CSV paths.
        eks_csv: bool
            Option to remove specific columns based on n_keypoints.

        Returns:
        -------
        tuple: (pd.DataFrame, list, list, list)
            - concatenated_df: Concatenated DataFrame with data from multiple CSV files.
            - csv_paths: List of CSV file paths from the specified Excel file.
            - video_paths: List of video file paths from the specified Excel file.
            - fixedpoint_paths: List of fixed-point file paths from the specified Excel file.
        """
        # Load CSV file paths from the provided base Excel file
        csv_paths, video_paths, fixedpoint_paths = self.load_from_excel(base_file)
        dataframes = [] 
        keys = []

        # Iterate through each CSV file path
        for path in csv_paths:
            # Load raw LP data from the CSV file
            raw_data = pd.read_csv(path, header=[0, 1, 2], index_col=0)
            
            # Drop last n_keypoints z-score columns(44) if eks_csv is True
            if eks_csv:
                raw_data = raw_data.iloc[:, :-self.n_keypoints]
            
            # Extract keypoint names and model name from the first file
            if self.keypoint_names is None:
                self.keypoint_names = [c[1] for c in raw_data.columns[::3]]
            
            self.model_name = raw_data.columns[0][0]
            
            # Format the raw LP data, excluding likelihood columns
            formatted_data= self.format_lpdata(raw_data, self.keypoint_names, self.model_name)
            
            # Add the formatted data to the list
            dataframes.append(formatted_data)
            keys.append(os.path.basename(path))  # Store the file name only for MultiIndex

        # Concatenate all DataFrames vertically, using MultiIndex keys
        concatenated_df = pd.concat(dataframes, axis=0, keys=keys, names=['SourceFile', 'RowIndex'])
        print(f"Total frames:{len(concatenated_df)} of {len(csv_paths)} files")
        
        return concatenated_df, csv_paths, video_paths, fixedpoint_paths

    def load_from_excel(self, base_file):
        """
        Load CSV file paths from an Excel file.

        Parameters:
        ----------
        base_file: str
            Path to the Excel file.

        Returns:
        -------
        tuple: (list, list, list)
            - csv_paths: List of CSV file paths from the Excel file.
            - video_paths: List of video file paths.
            - fixedpoint_paths: List of fixed-point file paths, or None if not provided.
        """
        if not os.path.isfile(base_file):
            raise ValueError(f'Base file not found: {base_file}')
        
        # Read the Excel file without specifying sheet name since it contains all paths in one sheet
        df = pd.read_excel(base_file, header=None)

        # Retrieve CSV file paths from the first column
        csv_paths = df.iloc[:, 0].tolist()
        video_paths = df.iloc[:, 1].tolist()

        # Check if there is a third column for fixed point paths
        fixedpoint_paths = df.iloc[:, 2].tolist() if df.shape[1] > 2 else None

        return csv_paths, video_paths, fixedpoint_paths

    def format_lpdata(self, df_rawlp, keypoint_names, model_name):
        """
        Format the raw LP data by extracting x, y coordinates, and excluding likelihood columns.

        Parameters:
        ----------
        df_rawlp: pd.DataFrame
            The raw LP data in wide format (columns are multi-indexed).
        keypoint_names: list
            List of keypoint names to extract.
        model_name: str
            Model name associated with the LP data.

        Returns:
        -------
        pd.DataFrame
            - df_lp: Formatted LP data with x, y coordinates only, without likelihood.
        """
        # Construct column names and add to the dictionary(df_lp)
        df_lp = {f'{feat}_{feat2}': df_rawlp.loc[:, (model_name, feat, feat2)] 
                 for feat in keypoint_names for feat2 in ['x', 'y']} 
        
        # Create a new DataFrame with formatted data (allocentric data)
        df_lp = pd.DataFrame(df_lp, index=df_rawlp.index)

        return df_lp

    def save_index_info(self, concatenated_df, csv_paths, video_paths, fixedpoint_paths, save_dir):
        """
        Save MultiIndex information (SourceFile, RowIndex, VideoPath, FixedPointPath) as a separate CSV file.

        Parameters:
        ----------
        concatenated_df: pd.DataFrame
            The concatenated DataFrame with MultiIndex.
        csv_paths: list
            List of CSV file paths used in the concatenated DataFrame.
        video_paths: list
            List of video file paths corresponding to each CSV.
        fixedpoint_paths: list, optional
            List of fixed-point file paths corresponding to each CSV, or None if not provided.
        save_dir: str
            Path to save the index information CSV file.
        """
        # Extract values from each MultiIndex level and convert them to a list
        source_files = concatenated_df.index.get_level_values('SourceFile').tolist()
        row_indices = concatenated_df.index.get_level_values('RowIndex').tolist()
        
        # Create a dictionary mapping each SourceFile to its corresponding VideoPath
        video_path_map = {os.path.basename(path): video_path for path, video_path in zip(csv_paths, video_paths)}
        video_paths_for_index = [video_path_map[source] for source in source_files]
        
        # If fixedpoint_paths is provided, create a dictionary for it as well. If not, use None or an empty string for each row
        fixedpoint_paths_for_index = ([{os.path.basename(path): fixedpoint_path for path, fixedpoint_path in zip(csv_paths, fixedpoint_paths)}.get(source)
                                       for source in source_files] if fixedpoint_paths else [None] * len(source_files))
        
        # Create a new DataFrame using the extracted index information
        index_info_df = pd.DataFrame({
            'SourceFile': source_files,
            'RowIndex': row_indices,
            'VideoPath': video_paths_for_index,
            'FixedPointPath': fixedpoint_paths_for_index
        })
        
        # Save the index information as a CSV file (without storing the index)
        index_info_df.to_csv(os.path.join(save_dir, 'index_info.csv'), index=False)
        print(f"Index information saved to {os.path.join(save_dir, 'index_info.csv')}")
