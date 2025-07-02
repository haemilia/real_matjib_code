import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import ast
from pathlib import Path
from PIL import Image
####################################################################################################
# Things to include in config
# "image_dir": C://~
# "params"
    # target_image_size: [224, 224]

#### DATASET ##########################################################################################
class NaverblogReviewDataset(Dataset):
    """
    A simplified PyTorch Dataset that loads raw data from the DataFrame.
    It performs initial NaN handling and prepares raw Python objects (strings, lists, floats)
    for subsequent processing by dedicated Preprocessor modules.
    """
    def __init__(self, config, dataframe: pd.DataFrame, image_paths_dict:dict):
        """
        Initializes the dataset with the DataFrame, performing only essential
        preprocessing that's best done once on the full DataFrame (like NaN handling).

        Args:
            dataframe (pd.DataFrame): The input pandas DataFrame containing review data.
        """
        self.dataframe = dataframe.copy()
        self.image_paths_dict = image_paths_dict
        params = config.get("params", {})
        self.target_image_size = tuple(params.get("target_image_size", [224, 224]))
        image_dir = Path(config["image_dir"]) # MUST BE PRESENT!!!

        # Use post_url as index
        if not self.dataframe["post_url"].is_unique():
            # Actual behaviour
            # self.dataframe.drop_duplicates(["post_url"], inplace=True)
            # DEBUG behaviour
            raise ValueError("Dataframe's post_url is not unique and cannot be used as index")
        self.dataframe.set_index("post_url", inplace=True)

        # Use dictionaries to define column types for clarity and processing logic
        self.text_cols =  ["text", "post_title", "author", "blogname"]
        self.image_link_cols = ["img_url", "sticker_url", "vid_thumb_url"]

        self.tabular_numerical_cols = ["commentcount", "post_date"]

        # Consolidated list of all columns we expect to process for existence and NaNs
        all_expected_cols = self.text_cols + \
                            self.image_link_cols + \
                            self.tabular_numerical_cols + \
                            ['is_advert']

        # This loop just ensures column presence; type-specific filling is done below.
        for col in all_expected_cols:
            if col not in self.dataframe.columns:
                # Add missing column; value will be overwritten by type-specific processing below
                self.dataframe[col] = np.nan # Use NaN initially for clearer type handling
                print(f"Warning: Column '{col}' not found in DataFrame. Adding as NaN for later processing.")

        # Standardize all text columns to final Python types
        for col_name in self.text_cols:
            self.dataframe[col_name] = self.dataframe[col_name].apply(self._parse_to_python_string)

        # Replace all image columns with values from image_paths_dict
        if self.image_paths_dict:
            for post_url, images in image_paths_dict.items():
                for image_col, img_list in images.items():
                    self.dataframe.loc[post_url, image_col] = [image_dir / img_path for img_path in img_list]

        # Numerical data preprocessing (Imputation for NaNs and type conversion)
        count_cols_to_zero_impute = ['commentcount']
        for col in count_cols_to_zero_impute:
            self.dataframe[col] = self.dataframe[col].fillna(0.0)

        # Ensure "post_date" column is datetime & convert to numerical representation
        self.dataframe["post_date"] = pd.to_datetime(self.dataframe["post_date"], errors='coerce').astype(np.int64)
        for col in self.tabular_numerical_cols:
            self.dataframe[col] = pd.to_numeric(self.dataframe[col], errors='coerce').fillna(0.0)

        self.actual_tabular_cols = [col for col in self.tabular_numerical_cols if col in self.dataframe.columns]

    def _parse_to_python_string(self, value):
        """
        Helper method to robustly convert a cell value into a single Python string.
        Handles None/NaN values and converts lists/arrays to their string representation.
        Always returns a string.
        """
        if pd.isna(value): # Catches None and numpy.nan
            return ""
        elif isinstance(value, (list, np.ndarray)): # If it's a list/array, convert to string representation
            return str(value) # e.g., ['a', 'b'] -> "['a', 'b']"
        else:
            return str(value) # Convert any other type to string

    def _parse_to_python_list_of_strings(self, value):
        """
        Helper method to robustly convert a cell value into a Python list of strings.
        Handles string representation of lists, actual lists/tuples/arrays, and None/NaN.
        Always returns a list of strings.
        """
        if isinstance(value, str):
            # Try ast.literal_eval first if it looks like a list string
            if value.startswith('[') and value.endswith(']'):
                try:
                    parsed_list = ast.literal_eval(value)
                    if isinstance(parsed_list, list):
                        return [str(item) for item in parsed_list] # Ensure elements are strings
                    else:
                        return [str(value)] # If literal_eval returns a non-list, treat original as single item
                except (ValueError, SyntaxError):
                    return [str(value)] # If literal_eval fails, treat original as single item
            else:
                return [str(value)] # It's a string, but not list-like, treat as single item in list
        elif isinstance(value, (list, tuple)):
            return [str(item) for item in value] # Already a list/tuple, ensure elements are strings
        elif isinstance(value, np.ndarray):
            return [str(item) for item in value.tolist()] # Convert np.ndarray to list and ensure strings
        elif pd.isna(value): # Catches None and numpy.nan
            return []
        else:
            return [str(value)] # Fallback for other unexpected types, convert to string and put in list

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        row = self.dataframe.iloc[idx]
        sample_data = {}

        for col_name in self.text_cols:
            sample_data[col_name] = row[col_name]
        
        for col_name in self.image_link_cols:
            if isinstance(row[col_name], list):
                loaded_images = [self._load_image(img_path) for img_path in row[col_name]]
            else:
                loaded_images = []
            sample_data[col_name] = loaded_images
        try:
            sample_data['tabular_data'] = [float(row[col]) for col in self.actual_tabular_cols]
        except Exception as e:
            # This fallback is useful if a specific numerical conversion issue occurs for a sample.
            # In your main `__init__`, we already handle NaNs and conversion, so this should ideally not be hit.
            print(f"Warning: Error processing tabular data for sample {idx}: {e}. Returning empty list.")
            sample_data['tabular_data'] = []

        sample_data['labels'] = torch.tensor(row['is_advert'], dtype=torch.float)

        return sample_data
    def _load_image(self, image_path: str) -> Image.Image:
        """
        Loads an image from a local file path.
        Returns a PIL Image object or a dummy black image if loading fails.
        """
        if not Path(image_path).exists() or not Path(image_path).is_file():
            return Image.new('RGB', self.target_image_size, color = 'black')

        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading image from {image_path}: {e}. Creating dummy black image.")
            return Image.new('RGB', self.target_image_size, color = 'black')

    def get_tabular_columns(self):
        """Helper to get the list of tabular columns actually used."""
        return self.actual_tabular_cols

    def get_text_columns(self):
        """Helper to get the list of text columns actually used."""
        return self.text_cols

    def get_image_link_columns(self):
        """Helper to get the list of image link columns actually used."""
        return self.image_link_cols