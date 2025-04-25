
import os
import copy
import pandas as pd
import numpy as np
import inspect
from IPython.display import display, JSON
from typing import Union, List
from analysis.utilities import (load_tracking_data,
                                      plot_growth_rate_with_ribbon,
                                      load_segmentations_h5,
                                      compute_densities_segmentation,
                                      plot_frame_cv2_jupyter_dict,
                                      sort_folder_names,
                                      )
from analysis.growth_rate import calculate_growth_rate
from plotting.histogram import plot_histogram, plot_value_count_histogram
from mutate.fuse import fuse_track_output
from mutate.filter import filter_by_column

class FluidExperiment:
    def __init__(self, path, color_channels = None, positions = None, name = "experiment"):
        self.path = path
        
        if positions is None:
            self.positions = [os.path.basename(f) for f in os.scandir(path) if f.is_dir()]
            self.positions = sort_folder_names(self.positions)
        else:
            self.positions = self._save_select(positions)
        
        if color_channels is None:
            self.color_channels =[os.path.basename(f) for f in os.scandir(os.join(path,self.positions[0])) if f.is_dir()]
        else:
            self.color_channels = self._save_select(color_channels)
            
        self.data = {}
        self.filter_history = {}
        self.n_frames = None
        self.unequal_lengths = False
        self.headers = None
        self.unequal_header = False
        self.metadata = None
        self.name = name
        for p in self.positions:
            self.data[p] = {}
            self.filter_history[p] = {}
            for g in self.color_channels:
                print(f"Loading sample at position {p} for group {g}")
                self.data[p][g] = load_tracking_data(os.path.join(path,p),g)
                self.filter_history[p][g] = []
                n_sample = np.max(self.data[p][g]['frame'])
                header = list(self.data[p][g].columns.values)
                if self.n_frames is not None:
                    if n_sample != self.n_frames:
                        Warning(f"unequal number of frames for some files. Was {self.n_frames}, but new sample at position {p} and channel {g} has length of {n_sample}")
                        self.unequal_lengths = True
                if self.headers is not None:
                    if header != self.headers:
                        Warning(f"unequal header for some files. Was {', '.join(map(str,self.headers))}, but new sample at position {p} and channel {g} has header of {', '.join(map(str,header))}")
                        self.unequal_header = True
                        
                self.n_frames = n_sample
                self.headers = header
                
        if not self.unequal_lengths:
            print(f"Successfully loaded data with consistent number of frames: {self.n_frames}")
    
    @classmethod
    def from_copy(cls, other):
        """Create a copy of an existing FluidExperiment without reloading from disk."""
        new_instance = cls.__new__(cls)  # Bypass __init__
        new_instance.path = other.path
        new_instance.positions = copy.deepcopy(other.positions)
        new_instance.color_channels = copy.deepcopy(other.color_channels)
        new_instance.data = copy.deepcopy(other.data)
        new_instance.n_frames = other.n_frames
        new_instance.unequal_lengths = other.unequal_lengths
        new_instance.headers = copy.deepcopy(other.headers)
        new_instance.unequal_header = other.unequal_header
        return new_instance
            
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
    
    def filter_data(self, 
                      column: str, 
                      min_occurences: int = 0, 
                      min_value: float = None, 
                      max_value: float = None, 
                      custom_function = None, 
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
                
        
    def print_QC_histograms(self, columns, positions = None, color_channels = None, group_by = None):
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


    def print_life_cycle_histograms(self, 
                                    columns, 
                                    positions = None, 
                                    color_channels = None, 
                                    group_by = None):
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

                
    def calculate_growth_rate(self, 
                              integration_window: int, 
                              id_column: str, 
                              value_column: str, 
                              frame_column: str = "frame",
                              growth_rate_column: str = "growth_rate", 
                              custom_method = None, 
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
                
    def plot_rates(self, 
                   rate_column="growth_rate", 
                   frame_column = "frame",
                   positions = None, 
                   color_channels = None, 
                   title = None, 
                   group_by = None):
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

                
    def max_value(self, column, position, group):
        return np.max(self.data[position][group][column])
          
    def collapse_all_positions(self, new_name = "fused"):
        print(f"Fuseing all exisiting positions into a single position with the name {new_name}")
        self.fuse_positions(self.positions[0], self.positions[1:], silent= True)
        self.data[new_name] = self.data[self.positions[0]]
        self.data.pop(self.positions[0])
        self.positions = [new_name]
        
    def create_metdata_template(self, path = None, overwrite = False):
            """
            Export a list of strings to a CSV file with a single column named 'position'.
    
            Parameters:
            - string_list: list of strings to write
            - output_path: destination file path
            - overwrite: if False, raises error if file exists; if True, overwrites file
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
            Load metadata from a CSV file generated by `export_list_to_csv`.
            Validates presence of expected columns: position, region, experiment, device_channel.
            
            Parameters:
            - input_path: path to the CSV file
            
            Returns:
            - DataFrame with metadata
            
            Raises:
            - ValueError if required columns are missing
            - FileNotFoundError if the file doesn't exist
            """
            if path is None:
                path = os.path.join(self.path, self.name + "_metadata.csv")
                
            expected_columns = {'position', 'group', 'experiment', 'device_channel'}
            
            df = pd.read_csv(path)
            missing = expected_columns - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {', '.join(missing)} in file {path}")
            
            missing_positions = set(self.positions) - set(df['position'])
            if missing_positions:
                raise ValueError(f"Positions not found in metadata file: {', '.join(missing_positions)}")
            self.metadata = df
                    
        
    def calculate_densities(self, distance_threshold = 50):
        print(f"Calculating densities of all channels in a radius of {distance_threshold} px")
        for p in self.positions:
            print(f"Calculate density for position {p}")
            mask = {}
            for c in self.color_channels:
                mask[c] = load_segmentations_h5(os.path.join(self.path, p),c)
            self.data[p] = compute_densities_segmentation(self.data[p],mask, distance_threshold)
            
    def get_data(self, 
        positions: Union[str, List[str]] = None, 
        color_channels: Union[str, List[str]] = None
        ):
        """
        Retrieve data from nested dictionary structure based on position and color_channel.
        
        Args:
            positions (str or list of str): Single or multiple position keys. Defaults to None = take all positions
            color_channels (str or list of str): Single or multiple color_channel keys. Defaults to None = take all color_channels
        
        Returns:
            dict: data from selected positions and channels
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
        
    def report_filter_history(self):
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

                
    
    def get_aggregate_data(self, column, color_channels = None):
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
    
    def _save_select(self, selected):
        if isinstance(selected,str):
            return [selected]
        return selected
    
    def print_selected_frame(self, frame, positions = None, color_channels = None, color = None):
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
        
    def drop_positions(self, positions):
        positions = self._save_select(positions)
            
        self.positions = [p for p in self.positions if p not in positions]
        for p in positions:
            print(f"Dropping position {p} from experiment")
            self.data.pop(p)

        
                
            