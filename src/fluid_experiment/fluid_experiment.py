import os
import copy
import pandas as pd
import numpy as np
import h5py
from typing import Union, List, Callable, Tuple
from analysis.utilities import (
                                      plot_frame_cv2_jupyter_dict,
                                      sort_folder_names,
                                      )
from analysis.growth_rate import calculate_growth_rate
from analysis.local_neighborhood import compute_neighborhood_segmentation
from plotting.histogram import plot_histogram, plot_value_count_histogram
from plotting.rate_plots import plot_growth_rate_with_ribbon
from plotting.qc_plots import plot_qc_xy_correlation
from mutate.fuse import fuse_track_output
from mutate.filter import filter_by_column
from mutate.load import load_tracking_data, load_segmentations_h5

class FluidExperiment:

# ==========================================================    
# ==================== CLASS METHODS =======================
# ==========================================================

    def __init__(self, 
                 path : str, 
                 color_channels: Union[str, List[str]] = None, 
                 positions: Union[str, List[str]] = None, 
                 name: str = "experiment"):
        """
        Initializes the data structure for fluid experiment data organized by position and color channel.
        This constructor scans a given directory path to identify experimental data, optionally filters by 
        specified positions and color channels, and loads the corresponding tracking data for each combination.

        Args:
            path (str): Path to the root directory containing subdirectories for each position. Each position
                directory is expected to contain subdirectories corresponding to different color channels.
            color_channels (Union[str, List[str]], optional): A specific color channel or list of channels to load.
                If None, all color channel directories found under the first position will be used. Defaults to None.
            positions (Union[str, List[str]], optional): A specific position or list of positions to load. If None, 
                all subdirectories in `path` will be considered as positions. Defaults to None.
            name (str, optional): A name for the experiment. Used for identification or metadata purposes. 
                Defaults to "experiment".
        
        Attributes:
            path (str): The root directory path.
            positions (List[str]): List of position names.
            color_channels (List[str]): List of color channel names.
            data (dict): Nested dictionary holding loaded tracking data indexed by position and channel.
            filter_history (dict): Nested dictionary tracking filters applied to the data.
            metadata (Any): Placeholder for experiment metadata (initially None, pd.DataFrame once loaded).
            name (str): Name of the experiment.
        """
        self.path = path
        
        if positions is None:
            self.positions = [os.path.basename(f) for f in os.scandir(path) if f.is_dir()]
            self.positions = sort_folder_names(self.positions)
        else:
            self.positions = self._save_select(positions)
        
        if color_channels is None:
            self.color_channels =[os.path.basename(f) for f in os.scandir(os.path.join(path,self.positions[0])) if f.is_dir()]
        else:
            self.color_channels = self._save_select(color_channels)
            
        self.data = {}
        self.filter_history = {}
        self.metadata = None
        self.name = name
        for p in self.positions:
            self.data[p] = {}
            self.filter_history[p] = {}
            for g in self.color_channels:
                print(f"Loading sample at position {p} for group {g}")
                self.data[p][g] = load_tracking_data(os.path.join(path,p),g)
                self.filter_history[p][g] = []
        
        self._update_information()

                
        if not self.unequal_lengths:
            print(f"Successfully loaded data with consistent number of frames: {self.n_frames}")
    
    @classmethod
    def from_copy(cls, other):
        """Create a copy of an existing FluidExperiment without reloading from disk."""
        new_instance = cls.__new__(cls)  # Bypass __init__
        new_instance.path = other.path
        new_instance.name = other.name
        new_instance.positions = copy.deepcopy(other.positions)
        new_instance.color_channels = copy.deepcopy(other.color_channels)
        new_instance.data = copy.deepcopy(other.data)
        new_instance.n_frames = other.n_frames
        new_instance.unequal_lengths = other.unequal_lengths
        new_instance.headers = copy.deepcopy(other.headers)
        new_instance.unequal_header = other.unequal_header
        new_instance.filter_history = copy.deepcopy(other.filter_history)
        if other.metadata is not None:
            new_instance.metadata = other.metadata.copy()
        else:
            new_instance.metadata = None
        return new_instance

    @classmethod
    def load(cls, file_path: str):
        """
        Load a FluidExperiment object from a single HDF5 file using h5py.

        Args:
            file_path (str): Path to the HDF5 file where the experiment is saved.

        Returns:
            FluidExperiment: An instance of the FluidExperiment class.
        """
        with h5py.File(file_path, 'r') as h5file:
            # Load experiment metadata
            experiment_info = h5file['experiment_info']
            instance = cls.__new__(cls)  # Bypass __init__
            instance.path = experiment_info.attrs['path']
            instance.name = experiment_info.attrs['name']
            instance.positions = list(experiment_info.attrs['positions'])
            instance.color_channels = list(experiment_info.attrs['color_channels'])
            instance.n_frames = experiment_info.attrs['n_frames']
            instance.unequal_lengths = experiment_info.attrs['unequal_lengths']
            instance.headers = list(experiment_info.attrs['headers'])
            instance.unequal_header = experiment_info.attrs['unequal_header']

            # Load metadata
            if 'metadata' in h5file:
                metadata_group = h5file['metadata']
                metadata_dict = {
                    col: [val.decode('utf-8') if isinstance(val, bytes) else val for val in metadata_group[col][...]]
                    for col in metadata_group
                }
                instance.metadata = pd.DataFrame(metadata_dict)
                instance.metadata.index = instance.metadata["position"]
            else:
                instance.metadata = None

            # Load data
            instance.data = {}
            data_group = h5file['data']
            for p in instance.positions:
                instance.data[p] = {}
                position_group = data_group[p]
                for c in instance.color_channels:
                    channel_group = position_group[c]
                    df_dict = {
                        col: [val.decode('utf-8') if isinstance(val, bytes) else val for val in channel_group[col][...]]
                        if channel_group[col].dtype.kind == 'S' else channel_group[col][...]
                        for col in channel_group
                    }
                    instance.data[p][c] = pd.DataFrame(df_dict)

            # Load filter history
            instance.filter_history = {p: {c: [] for c in instance.color_channels} for p in instance.positions}
            if 'filter_history' in h5file:
                filter_history_group = h5file['filter_history']
                for p in filter_history_group:
                    position_group = filter_history_group[p]
                    for c in position_group:
                        channel_group = position_group[c]
                        for i in channel_group:
                            history_group = channel_group[i]
                            history_entry = {key: history_group.attrs[key] for key in history_group.attrs}
                            instance.filter_history[p][c].append(history_entry)

        print(f"Successfully loaded experiment with data from {len(instance.positions)} positions and {len(instance.color_channels)} color channels")
        return instance
   
# DUNDER METHODS

    def __str__(self):
        info = (f"FluidExperiment with path: {self.path}\n" +
            f"{len(self.positions)} positions: {', '.join(map(str, self.positions))}\n" +
            f"{len(self.color_channels)} color channels: {', '.join(map(str, self.color_channels))}\n")
        
        if self.n_frames is not None and not self.unequal_lengths:
            info = info + f"length of experiment is consistent: {self.n_frames}\n"
        
        if self.headers is not None and not self.unequal_header:
            info = info + f"experiment has consistent headers: {', '.join(map(str,self.headers))}\n"    
            
        if self.metadata is not None:
            info = info + "Experiment has metadata:\n"
            info = info + self.metadata.to_string()
            
        
        return info

    def __add__(self, other):
        """dunder method to add two FluidExperiments. returns a new FluidExperiment with the data of both experiments. does not fuse in place

        Args:
            other (FluidExperiment): the other FluidExperiment to be added

        Returns:
            FluidExperiment: a new fluid experiment with the data of both experiments
        """
        new_exp = FluidExperiment.from_copy(self)
        new_exp.fuse(other)
        return new_exp
    
    def __getitem__(self, position):
        """
        Allow dictionary-like access to the data of a specific position.

        Args:
            position (str): The position key to access.

        Returns:
            dict: A dictionary of color channels and their corresponding data for the specified position.

        Raises:
            KeyError: If the position does not exist in the experiment.
        """
        if position not in self.data:
            raise KeyError(f"Position '{position}' does not exist in the experiment.")
        return self.data[position]
 
    def __getattr__(self, position):
        """
        Allow attribute-like access to the metadata of a specific position.

        Args:
            position (str): The position key to access.

        Returns:
            pd.Series: The metadata for the specified position.

        Raises:
            AttributeError: If the position does not exist in the metadata.
        """
        if self.metadata is None:
            raise AttributeError("Metadata is not loaded for this experiment.")
        if position not in self.metadata.index:
            raise AttributeError(f"Position '{position}' does not exist in the metadata.")
        return self.metadata.loc[position]
       
# PRIVATE METHODS    

    def _save_select(self, selected: Union[str, List[str]]) -> List[str]:
        """private method used to select positions or color channels savely (i.e if only one is selected, it returns a list with this single element)

        Args:
            selected (Union[str, List[str]]): the selected positions or color channels

        Returns:
            [str]: a list of selected positions or color channels compatible with any other method of FluidExperiment
        """
        if isinstance(selected,str):
            return [selected]
        return selected

    def _update_information(self):
        """
        checks length of all files (n frames) and headers for consistency. updates print information
        """
        self.n_frames = None
        self.unequal_lengths = False
        self.headers = None
        self.unequal_header = False

        for p in self.positions:
            for g in self.color_channels:
                n_sample = np.max(self.data[p][g]['frame'])
                header = list(self.data[p][g].columns.values)

                if self.n_frames is not None:
                    if n_sample != self.n_frames:
                        print(
                            f"Warning: Unequal number of frames detected. "
                            f"Expected {self.n_frames}, but position {p}, channel {g} has {n_sample} frames."
                        )
                        self.unequal_lengths = True

                if self.headers is not None:
                    if header != self.headers:
                        print(
                            f"Warning: Unequal headers detected. "
                            f"Expected {', '.join(map(str, self.headers))}, but position {p}, channel {g} has "
                            f"header {', '.join(map(str, header))}."
                        )
                        self.unequal_header = True
                self.n_frames = n_sample
                self.headers = header
        
    def _fuse_nested_dict(self, target, source, renamed_positions):
        """
        Custom method to fuse nested dictionaries (e.g., data or filter history) with unique position names.

        Args:
            target (dict): The target dictionary to be updated.
            source (dict): The source dictionary to be merged into the target.
            renamed_positions (list): List of unique position names for the source dictionary.

        Returns:
            None: The target dictionary is updated in place.
        """
        for original_position, new_position in zip(source.keys(), renamed_positions):
            if new_position not in target:
                target[new_position] = source[original_position]
            else:
                raise ValueError(f"Conflict detected: Position '{new_position}' already exists in the target dictionary.")

    def _generate_unique_positions(self, positions: List[str]) -> List[str]:
        """
        Generate unique position names to avoid conflicts with existing positions.

        Args:
            positions (list): List of positions to be renamed.

        Returns:
            list: List of unique position names.
        """
        existing_positions = set(self.positions)
        unique_positions = []
        for position in positions:
            new_position = position
            counter = 1
            while new_position in existing_positions:
                new_position = f"{position}.{counter}"
                counter += 1
            unique_positions.append(new_position)
            existing_positions.add(new_position)
        return unique_positions

# ==========================================================    
# ==================== MUTATION METHODS ====================
# ==========================================================

    def filter_data(self, 
                      column: str, 
                      min_occurences: int = 0, 
                      min_value: float = None, 
                      max_value: float = None, 
                      custom_function: Callable = None, 
                      **custom_kwargs):
        """
        Filters a dataframe base on a specified track. supports two modes (or combined mode)
        if min_occurence > 0, filters out any row for which the unique value in the target column has less than min_occurence entries
        if min_value or max_value not None, filters out any row for which this column has a value below or above these parameters
        if multiple set, will first filter by min occurence, and then on top by min and max values
        the function returns a tuple of a filtered data frame and a summary dictionary that informs about filtering statistics (method used and number of filtered rows / values)
        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Column to filter with. i.e "trackID" for min occurences or "area" for min / max values.
            min_occurences (int, optional): Minimum number of occurrences to retain. Defaults to 0.
            min_value (float, optional): Minimum value threshold. Defaults to None.
            max_value (float, optional): Maximum value threshold. Defaults to None.
            custom_function (function): can be set to a custom function
        """
        if min_occurences >= 0:
            print(f"Filtering out {column} with less than {min_occurences} occurences")
        if min_value is not None or max_value is not None:
            print(f"Filtering out {column} with min value {min_value} and max value {max_value}")
        
        for p in self.positions:
            for g in self.color_channels:
                print(f"Filtering channel {g} at position {p}:")
                if custom_function is not None:
                    # apply the custom function with optional additional kwargs
                    self.data[p][g], summary = custom_function(self.data[p][g], 
                                                               column=column, 
                                                               min_occurences=min_occurences, 
                                                               min_value = min_value, 
                                                               max_value = max_value, 
                                                               **custom_kwargs)
                else:
                    #Default function used by midap-tools
                    self.data[p][g], summary = filter_by_column(self.data[p][g], 
                                                                column, 
                                                                min_occurences,
                                                                min_value, 
                                                                max_value)
                self.filter_history[p][g].append(summary)
        self._update_information()
        
    def drop_positions(self, positions: Union[str,List[str]]):
        """
        Removes specified positions from the experiment.
        This function deletes the data, filter history, and metadata associated with the specified positions.
        Args:
            positions (str or list of str): The positions to remove from the experiment.
        Returns:
            None: The function updates the experiment by removing the specified positions.
        """
        positions = self._save_select(positions)  
        self.positions = [p for p in self.positions if p not in positions]
        for p in positions:
            print(f"Dropping position {p} from experiment")
            self.data.pop(p)
            self.filter_history.pop(p)
            if self.metadata is not None:
                self.metadata.drop(index = p, inplace= True)
        self._update_information()
           
    def drop_color_channels(self, color_channels: Union[str,List[str]]):
        """
        Removes specified color channels from the experiment.
        This function deletes the data and filter history associated with the specified color channels.

        Args:
            color_channels (str or list of str): The color channels to remove from the experiment.

        Returns:
            None: The function updates the experiment by removing the specified color channels.
        """
        color_channels = self._save_select(color_channels)
        self.color_channels = [c for c in self.color_channels if c not in color_channels]
        for p in self.positions:
            for c in color_channels:
                print(f"Dropping color_channel {c} from position {p} in experiment")
                self.data[p].pop(c)
                self.filter_history[p].pop(c)
        self._update_information()

    def create_metdata_template(self, path = None, overwrite = False):
            """
            Creates and exports a metadata template as a CSV file.
            The template includes columns for position, group, experiment, and device_channel. 
            It is useful for initializing metadata for the experiment.

            Args:
                path (str, optional): The file path to save the metadata template. Defaults to None, 
                                    which saves the file in the experiment's path with a default name.
                overwrite (bool, optional): If False, raises an error if the file already exists. 
                                            If True, overwrites the file. Defaults to False.
            Returns:
                None: The function writes the metadata template to a CSV file.

            """
            if path is None:
                path = os.path.join(self.path, self.name + "_metadata.csv")
            if not overwrite and os.path.exists(path):
                print(f"Skipped: metadata file already exists at {path}")
                return
            df = pd.DataFrame({
                'position': self.positions,
                'group': 'default',
                'experiment': self.name,
                'device_channel': ''})
            df.to_csv(path, index=False)
            print(f"Empty metadata file written to {path}")
            
    def load_metadata_template(self, path = None):
            """
            Load metadata from a CSV file generated by `create_metdata_template`.
            Validates presence of expected columns: position, region, experiment, device_channel.
            
            Args:
                path (str, optional): Path to the metadata CSV file. If None, defaults to the experiment's path with a default name.
            
            Returns:
                None: The function updates the metadata attribute of the FluidExperiment instance.
            
            Raises:
                ValueError if required columns are missing
                FileNotFoundError if the file doesn't exist
            """
            if path is None:
                path = os.path.join(self.path, self.name + "_metadata.csv")
                
            expected_columns = {'position', 'group', 'experiment', 'device_channel'}
            
            df = pd.read_csv(path)
            df.index = df["position"]
            missing = expected_columns - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {', '.join(missing)} in file {path}")
            
            missing_positions = set(self.positions) - set(df['position'])
            if missing_positions:
                raise ValueError(f"Positions not found in metadata file: {', '.join(missing_positions)}")
            self.metadata = df.loc[self.positions]

    def fuse(self, other):
        """
        Fuse the current FluidExperiment with another FluidExperiment.

        Args:
            other (FluidExperiment): The second FluidExperiment to be fused with the current one.

        Returns:
            None: The current FluidExperiment is updated with the data from the second experiment.
        """
        if not isinstance(other, FluidExperiment):
            raise TypeError("The input must be an instance of FluidExperiment.")
        if set(self.color_channels) != set(other.color_channels):
            raise ValueError(
                "The color_channels of the two experiments are not identical. "
                f"self.color_channels: {self.color_channels}, other.color_channels: {other.color_channels}"
            )

        # Ensure unique position names for the second experiment
        other_positions_renamed = self._generate_unique_positions(other.positions)
        self.positions.extend(other_positions_renamed)
        self._fuse_nested_dict(self.data, other.data, other_positions_renamed)
        self._fuse_nested_dict(self.filter_history, other.filter_history, other_positions_renamed)
        
        # Fuse metadata
        if self.metadata is not None and other.metadata is not None:
            other.metadata = other.metadata.copy()
            other.metadata["position"] = other_positions_renamed
            self.metadata = pd.concat([self.metadata, other.metadata]).drop_duplicates().reset_index(drop=True)
        elif self.metadata is not None:
            print("New FluidExperiment does not contain metadata. Resetting metadata for fused object")
            self.metadata = None
        elif other.metadata is not None:
            print("Existing FluidExperiment does not contain metadata. Resetting metadata for fused object")

        self._update_information()   
        
       
# ==========================================================    
# ==================== CALCULATION METHODS =================
# ==========================================================
            
    def calculate_growth_rate(self, 
                              integration_window: int, 
                              id_column: str, 
                              value_column: str, 
                              frame_column: str = "frame",
                              growth_rate_column: str = "growth_rate", 
                              custom_method: Callable = None, 
                              **custom_kwargs):
        """
        Determines the growth rate and R-squared of value_column over a specified integration window in entire experiment.
        Args:
            df (pd.DataFrame): track output dataframe
            integration_window (int): number of frames over which to calculate growth and R-squared
            id_column (str): entity identifier column (e.g., "trackID")
            value_column (str): column representing the metric to track (e.g., "area")
            frame_column (str): column representing frame index. Defaults to "frame"
            growth_rate_column (str): name for the new growth rate column. Defaults to "growth_rate"
            r_squared_column (str): name for the new R-squared column. Defaults to "growth_rsquared"
            custom_method (function): custom method to be used for calculation. defaults to = None
            **custom_kwargs : additional arguments the custom function may take
        """
        print(f"Calculate growth rate for {id_column} measured with {value_column} over an integration window of {integration_window}")
        for p in self.positions:
            for c in self.color_channels:
                if custom_method is not None:
                    # apply the custom function with optional additional kwargs
                    self.data[p][c] = custom_method(self.data[p][c], 
                                                            integration_window, 
                                                            id_column, 
                                                            value_column,
                                                            frame_column,
                                                            growth_rate_column,
                                                            **custom_kwargs)
                else:    
                    # Default midap-tools method
                    self.data[p][c] = calculate_growth_rate(self.data[p][c], 
                                                            integration_window, 
                                                            id_column, 
                                                            value_column,
                                                            frame_column,
                                                            growth_rate_column)
        self._update_information()
   
    def calculate_local_neighborhood(self, 
                                     distance_threshold: int, 
                                     neighborhood_prefix = "density_", 
                                     custom_function: Callable = None, 
                                     **custom_kwargs):
        """
        Calculates the local neighborhood density for all positions and color channels.
        This function computes the density of overlap between a circular region around each target point 
        and a set of binary masks. The results are added as new columns to the data.

        Args:
            distance_threshold (int): The radius of the circular region (in pixels) used to compute the neighborhood density.
            neighborhood_prefix (str, optional): A prefix for the column names added to the data. Defaults to "density_".
            custom_function (function, optional): A custom function for neighborhood calculation. Defaults to None, 
                                                which uses the default method.
            **custom_kwargs: Additional arguments for the custom function.

        Returns:
            None: The function updates the data with neighborhood density columns.
        """
        print(f"Calculating neighborhoods of all channels in a radius of {distance_threshold} px")
        for p in self.positions:
            print(f"Calculate neighborhoods for position {p}")
            mask = {}
            for c in self.color_channels:
                mask[c] = load_segmentations_h5(os.path.join(self.path, p),c)
            if custom_function is None:
                #default method for neighborhood calculation from segmentations
                self.data[p] = compute_neighborhood_segmentation(self.data[p],
                                                                 mask, 
                                                                 neighborhood_prefix,
                                                                 distance_threshold)
            else:
                #in case the user specified a custom function
                self.data[p] = custom_function(self.data[p],
                                               mask, 
                                               distance_threshold, 
                                               neighborhood_prefix,
                                               **custom_kwargs)
        self._update_information()
 
    def calculate_transform_data(self, 
                           column: str, 
                           postfix: str = "_log",
                           type: str = "log", 
                           custom_function: Callable = None,
                           **custom_kwargs):
        """
        Adds a transformed version of a specified column to the data.
        The transformation can be logarithmic or any other type specified by the user.

        Args:
            column (str): The column to transform.
            postfix (str, optional): The postfix to append to the new column name. Defaults to "_log".
            type (str, optional): The type of transformation to apply. Defaults to "log". valid alternatives are "square" and "inverse"
            custom_method (function): custom method to be used for calculation (applied to a pd.Series). defaults to = None. If custom function defined, will ignore type and apply this function instead
            **custom_kwargs : additional arguments the custom function may take

        Returns:
            None: The function updates the data with the transformed column.
        """
        for p in self.positions:
            for c in self.color_channels:
                if custom_function is not None:
                    self.data[p][c][column + postfix] = custom_function(self.data[p][c][column], **custom_kwargs)
                match type:
                    case "log":
                        self.data[p][c][column + postfix] = np.log(self.data[p][c][column])
                    case "square":
                        self.data[p][c][column + postfix] = np.square(self.data[p][c][column])
                    case "inverse":
                        self.data[p][c][column + postfix] = np.reciprocal(self.data[p][c][column])
        self._update_information()            
            
# ==========================================================    
# ==================== PLOTTING / REPORTING METHODS ========
# ==========================================================                
        
    def plot_qc_histograms(self, 
                           columns: Union[str, List[str]], 
                           positions: Union[str, List[str]] = None, 
                           color_channels: Union[str, List[str]] = None, 
                           group_by: str = None):
        """
        Plots a QC historgram for selected samples

        Args:
            columns [str]: name of columns that should be shown
            position str : Name of position to be plotted. Defaults to None = one plot for each position.
            color_channel str : Name of channel to be shown. Defaults to None = all channels shown next to each other.
            group_by  str: name of metadata column by which data should be aggregated prior to plotting
        """
        if positions is not None and group_by is not None:
            print("can not select groups and positions, ignoring positions selection for plot")    
        if positions is None:
            positions = self.positions
        if color_channels is None:
            color_channels = self.color_channels
        
        if group_by is not None:
            pdat = self.get_aggregate_data(group_by, color_channels)
            for k, v in pdat.items():
                plot_histogram(v,columns, title = f"histograms for aggregated data {k}")
            return
        else:
            pdat = self.get_data(positions, color_channels)
            for k, v in pdat.items():
                plot_histogram(v,columns, title = f"histograms for position data {k}")
                   
    def plot_qc(self, 
                value_column: str, 
                n_samples: int = 8,
                id_column: str = "trackID", 
                frame_column: str = "frame",
                positions: Union[str, List[str]] = None, 
                color_channels: Union[str, List[str]] = None, 
                group_by: str = None):
        """
        Plots QC (Quality Control) scatter plots for selected samples.

        Args:
            value_column (str): The column representing the Y-axis values (e.g., "major_axis_length").
            n_samples (int, optional): The number of random examples to plot. Defaults to 8.
            id_column (str, optional): The column used to group the data. Defaults to "trackID".
            frame_column (str, optional): The column representing the X-axis values. Defaults to "frame".
            positions (str or list of str, optional): List of positions to include in the plot. Defaults to None, 
                                            which includes all positions.
            color_channels (str or list of str, optional): List of color channels to include in the plot. Defaults to None, 
                                                    which includes all color channels.
            group_by (str, optional): Metadata column name to group data by for aggregation. If specified, 
                                    positions are ignored.
        """
        if positions is not None and group_by is not None:
            print("can not select groups and positions, ignoring positions selection for plot")    
        if positions is None:
            positions = self.positions
        if color_channels is None:
            color_channels = self.color_channels
            
        if group_by is not None:
            pdat = self.get_aggregate_data(group_by, 
                                           color_channels)
        else:
            pdat = self.get_data(positions, color_channels)
            
        for k, v in pdat.items():
            for k2, v2 in v.items():
                plot_qc_xy_correlation(v2,
                                    id_column = id_column,
                                    value_column = value_column,
                                    frame_column = frame_column,
                                    n = n_samples,
                                    title = f"{k}, {k2}, {id_column}")

    def plot_life_cycle_histograms(self, 
                                    columns: Union[str, List[str]], 
                                    positions: Union[str, List[str]] = None, 
                                    color_channels: Union[str, List[str]] = None, 
                                    group_by: str = None):
        """
        Plots a QC historgram for selected samples

        Args:
            columns (Union[str, List[str]], optional): name of columns that should be shown
            position (Union[str, List[str]], optional) : Name of position to be plotted. Defaults to None = one plot for each position.
            color_channel (Union[str, List[str]], optional) : Name of channel to be shown. Defaults to None = all channels shown next to each other.
            group_by  str: name of metadata column by which data should be aggregated prior to plotting
        """
        if positions is not None and group_by is not None:
            print("can not select groups and positions, ignoring positions selection for plot")    
        if positions is None:
            positions = self.positions
        if color_channels is None:
            color_channels = self.color_channels
        
        if group_by is not None:
            pdat = self.get_aggregate_data(group_by, 
                                           color_channels)
            for k, v in pdat.items():
                plot_value_count_histogram(v,
                                           columns, 
                                           title = f"life cycle histogram for aggregated data {k}")
            return
        else:
            pdat = self.get_data(positions, color_channels)
            for k, v in pdat.items():
                plot_value_count_histogram(v,
                                           columns, 
                                           title = f"life cycle histogram for position data {k}")
      
    def plot_rates(self, 
                   rate_column: str = "growth_rate", 
                   frame_column: str = "frame",
                   positions: Union[str, List[str]] = None, 
                   color_channels: Union[str, List[str]] = None, 
                   title: str = None, 
                   group_by: str = None):
        """
        Plots the growth rate over time for the experiment.
        This function generates line plots with ribbons representing the growth rate over time. 
        It supports plotting for individual positions and color channels or aggregated data based on metadata grouping.

        Args:
            rate_column (str): The column name representing the growth rate to be plotted. Defaults to "growth_rate".
            frame_column (str): The column name representing the frame index. Defaults to "frame".
            positions (list of str or str, optional): List of positions to include in the plot. Defaults to None, which includes all positions.
            color_channels (list of str or str, optional): List of color channels to include in the plot. Defaults to None, which includes all color channels.
            title (str, optional): Title of the plot. Defaults to "Mean Growth Rate per Frame".
            group_by (str, optional): Metadata column name to group data by for aggregation. If specified, positions are ignored.

        Returns:
            None: The function generates plots but does not return any value.
        """
        if positions is not None and group_by is not None:
            print("can not select groups and positions, ignoring positions selection for plot")    
        if positions is None:
            positions = self.positions
        if color_channels is None:
            color_channels = self.color_channels
        if title is None:
            title = "Mean Growth Rate per Frame"
            
        if group_by is not None:
            pdat = self.get_aggregate_data(group_by, color_channels)
            for k, v in pdat.items():
                plot_growth_rate_with_ribbon(v, 
                                             rate_column,
                                             frame_column, 
                                             title = f"{title} for aggregated data {k}")
            return
        else:
            pdat = self.get_data(positions, color_channels)
            for k, v in pdat.items():
                plot_growth_rate_with_ribbon(v, 
                                             rate_column, 
                                             frame_column,
                                             title = f"{title} for position data {k}")
        
    def plot_selected_frame(self, 
                             frame: int, 
                             positions: Union[str, List[str]] = None, 
                             color_channels: Union[str, List[str]] = None, 
                             color: List[Tuple[int, int, int]] = None):
        """
        Plots a selected frame from the experiment data with optional filtering by positions and color channels.
        This method overlays segmentation data for the specified frame from each selected color channel
        and displays the result as image

        Args:
            frame (int): Index of the frame to plot.
            positions (Union[str, List[str]], optional): A single position or list of positions to include in the plot.
                If None, all loaded positions are used. Defaults to None.
            color_channels (Union[str, List[str]], optional): A single color channel or list of channels to plot.
                If None, all loaded color channels are used. Defaults to None.
            color (List[Tuple[int, int, int]], optional): List of RGB color tuples to use for visualizing each channel.
                If None, default color mappings are used by the plotting function. Defaults to None. needs to be of length(color_channels)
        """
        if color_channels is None:
            color_channels = self.color_channels   
        color_channels = self._save_select(color_channels)
        
        if positions is None:
            positions = self.positions
        positions = self._save_select(positions)
        
        for p in positions:
            array = {}
            for c in color_channels:
                array[c] = load_segmentations_h5(os.path.join(self.path, p),c)
            plot_frame_cv2_jupyter_dict(array,frame, color, title = f"{p}: Overlay of Channels")
 
    def report_filter_history(self):
        """
        Prints the filter history for all positions and color channels.
        The filter history includes details about the filters applied to each color channel at each position, 
        such as the number of filters and their parameters.
        """
        for p in self.positions:
            for c in self.color_channels:
                print(f"Filtering applied to color channel {c} at position {p}:")
                history = self.filter_history[p][c]
                print(f"Number of filters applied: {len(history)}")
                for i, h in enumerate(history):
                    print(f"Filter {1 + i}:")
                    for k, v in h.items():
                        print(f"\t{k}: {v}")
                print("\n")
      
# ==========================================================    
# ==================== EXPORT METHODS ======================
# ==========================================================            

    def get_data(self, 
        positions: Union[str, List[str]] = None, 
        color_channels: Union[str, List[str]] = None
        ) -> dict:
        """
        Retrieve data from nested dictionary structure based on position and color_channel.
        
        Args:
            positions (str or list of str): Single or multiple position keys. Defaults to None = take all positions
            color_channels (str or list of str): Single or multiple color_channel keys. Defaults to None = take all color_channels
        
        Returns:
            dict: the nested results dictionary {"position": {"color_channel": pd.DataFrame}}.
        """
        if positions is None:
            positions = self.positions
        if color_channels is None:
            color_channels = self.color_channels
        positions = self._save_select(positions)
        color_channels = self._save_select(color_channels)
        
        return {
            pos: {
                ch: self.data[pos][ch].copy()
                for ch in color_channels
            }
            for pos in positions
        }
        
    def get_aggregate_data(self, 
                           column: str, 
                           color_channels: Union[str, List[str]] = None) -> dict:
        """
        Aggregates tracking data across groups defined by a metadata column.
        Groups positions using a specified column from the loaded metadata, then fuses tracking data 
        across all positions in each group for the specified color channels. Returns a nested dictionary 
        containing aggregated data per group and channel.

        Args:
            column (str): The metadata column name to group positions by. Each unique value in the column
                defines a group of positions whose data will be aggregated.
            color_channels (Union[str, List[str]], optional): A single color channel or list of channels to include
                in aggregation. If None, all loaded color channels are used. Defaults to None.

        Returns:
            dict: the nested results dictionary {"group": {"color_channel": pd.DataFrame}}.
        """
        if self.metadata is None:
            raise AttributeError("Experiment does not have any metadata, use load_metadata_template first")
        if column not in self.metadata.columns:
            raise ValueError(f"Column '{column}' not found in metadata.")
        
        if color_channels is None:
            color_channels = self.color_channels
        
        grouped_positions = (
            self.metadata.groupby(column)['position']
            .apply(list)
            .to_dict()
        )
        agg_dat = {}
        for k, v in grouped_positions.items():
            agg_dat[k] = {}
            for c in color_channels:
                dat = [self.data[i][c] for i in v]
                agg_dat[k][c] = fuse_track_output(dat)     
        return agg_dat

    def save(self, file_path: str = None):
        """
        Save the FluidExperiment object to a single HDF5 file using h5py.

        Args:
            file_path (str, optional): Path to the HDF5 file where the experiment will be saved. defaults to the results folder and experiment name
        """
        if file_path is None:
            file_path = os.path.join(self.path,self.name + ".h5")
        
        with h5py.File(file_path, 'w') as h5file:
            # Save metadata
            if self.metadata is not None:
                metadata_group = h5file.create_group('metadata')
                for col in self.metadata.columns:
                    metadata_group.create_dataset(col, data=self.metadata[col].values)
            
            # Save data
            data_group = h5file.create_group('data')
            for p in self.positions:
                position_group = data_group.create_group(p)
                for c in self.color_channels:
                    channel_group = position_group.create_group(c)
                    df = self.data[p][c]
                    for col in df.columns:
                        channel_group.create_dataset(col, data=df[col].values)

            # Save filter history
            filter_history_group = h5file.create_group('filter_history')
            for p in self.positions:
                position_group = filter_history_group.create_group(p)
                for c in self.color_channels:
                    channel_group = position_group.create_group(c)
                    for i, history in enumerate(self.filter_history[p][c]):
                        history_group = channel_group.create_group(str(i))
                        for key, value in history.items():
                            history_group.attrs[key] = value

            # Save experiment metadata
            experiment_info_group = h5file.create_group('experiment_info')
            experiment_info_group.attrs['path'] = str(self.path)
            experiment_info_group.attrs['name'] = self.name
            experiment_info_group.attrs['positions'] = self.positions
            experiment_info_group.attrs['color_channels'] = self.color_channels
            experiment_info_group.attrs['n_frames'] = self.n_frames
            experiment_info_group.attrs['unequal_lengths'] = self.unequal_lengths
            experiment_info_group.attrs['headers'] = self.headers
            experiment_info_group.attrs['unequal_header'] = self.unequal_header

        print(f"Saved experiment at {file_path}")
                
    
   
