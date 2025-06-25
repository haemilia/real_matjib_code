import torch
import torch.nn as nn
from transformers import AutoModel, CLIPVisionModel
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import os
import ast
import re
from soynlp.normalizer import repeat_normalize # type: ignore[import-untyped]

###################################################################################################################################################################
# Dataset
class NavermapReviewDataset(Dataset):
    """
    A simplified PyTorch Dataset that loads raw data from the DataFrame.
    It performs initial NaN handling and prepares raw Python objects (strings, lists, floats)
    for subsequent processing by dedicated Preprocessor modules.
    """
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initializes the dataset with the DataFrame, performing only essential
        preprocessing that's best done once on the full DataFrame (like NaN handling).

        Args:
            dataframe (pd.DataFrame): The input pandas DataFrame containing review data.
        """
        self.dataframe = dataframe.copy()

        # Use dictionaries to define column types for clarity and processing logic
        self.text_cols = {
            'review_text': 'string',
            'store_naver_name': 'string',
            'visit_keywords': 'list_of_strings',
            'keyword_tags_hangul': 'list_of_strings',
            'category': 'list_of_strings'
        }
        self.image_link_cols = {
            'image_links': 'list_of_strings',
            'video_thumbnail_links': 'list_of_strings'
        }
        self.tabular_numerical_cols = [
            'num_of_media', 'visit_count', 'author_total_reviews',
            'author_total_images', 'reactions_fun', 'reactions_helpful',
            'reactions_wannago', 'reactions_cool', 'review_year', 'rating'
        ]

        # Consolidated list of all columns we expect to process for existence and NaNs
        all_expected_cols = list(self.text_cols.keys()) + \
                            list(self.image_link_cols.keys()) + \
                            self.tabular_numerical_cols + \
                            ['is_advert']

        # This loop just ensures column presence; type-specific filling is done below.
        for col in all_expected_cols:
            if col not in self.dataframe.columns:
                # Add missing column; value will be overwritten by type-specific processing below
                self.dataframe[col] = np.nan # Use NaN initially for clearer type handling
                print(f"Warning: Column '{col}' not found in DataFrame. Adding as NaN for later processing.")

        # Standardize all text and image-link columns to final Python types in __init__ ---
        for col_name, col_type in self.text_cols.items():
            if col_type == 'string':
                # Apply _parse_to_python_string to ensure native str (handles None/NaN to "")
                self.dataframe[col_name] = self.dataframe[col_name].apply(self._parse_to_python_string)
            else: # col_type == 'list_of_strings'
                # Apply _parse_to_python_list_of_strings to ensure native list[str] (handles None/NaN to [])
                self.dataframe[col_name] = self.dataframe[col_name].apply(self._parse_to_python_list_of_strings)

        for col_name, col_type in self.image_link_cols.items():
            # All image_link_cols are expected to be 'list_of_strings'
            self.dataframe[col_name] = self.dataframe[col_name].apply(self._parse_to_python_list_of_strings)

        # No change to numerical fillna logic (this part remains as is, as it's separate)
        # Note: 'is_advert' handling is also separate below and is fine.

        # Numerical data preprocessing (Imputation for NaNs and type conversion)
        count_cols_to_zero_impute = ['author_total_reviews', 'author_total_images', 'rating']
        for col in count_cols_to_zero_impute:
            self.dataframe[col] = self.dataframe[col].fillna(0.0)

        if 'is_advert' in self.dataframe.columns:
            self.dataframe['is_advert'] = self.dataframe['is_advert'].astype(float)
        else:
            raise ValueError("Label column 'is_advert' was missing and defaulted to 0.0.")

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

        for col_name, col_type in self.text_cols.items(): # Use col_name to iterate
            # row[col_name] is now guaranteed to be 'str' for 'string' types, and 'list[str]' for 'list_of_strings' types
            if col_type == 'string':
                sample_data[col_name] = row[col_name]
            else: # col_type == 'list_of_strings'
                sample_data[col_name] = list(row[col_name]) # It's already a list, make a defensive copy

        for col_name in self.image_link_cols.keys(): # Iterate through just the keys
            sample_data[col_name] = list(row[col_name]) # It's already a list, make a defensive copy

        try:
            sample_data['tabular_data'] = [float(row[col]) for col in self.actual_tabular_cols]
        except Exception as e:
            # This fallback is useful if a specific numerical conversion issue occurs for a sample.
            # In your main `__init__`, we already handle NaNs and conversion, so this should ideally not be hit.
            print(f"Warning: Error processing tabular data for sample {idx}: {e}. Returning empty list.")
            sample_data['tabular_data'] = []

        sample_data['labels'] = torch.tensor(row['is_advert'], dtype=torch.float)

        return sample_data


    def get_tabular_columns(self):
        """Helper to get the list of tabular columns actually used."""
        return self.actual_tabular_cols

    def get_text_columns(self):
        """Helper to get the list of text columns actually used."""
        # Note: This returns the *defined* text columns, not necessarily all present.
        # However, __init__ will ensure they exist (even if added as empty) in the DataFrame.
        return self.text_cols

    def get_image_link_columns(self):
        """Helper to get the list of image link columns actually used."""
        return self.image_link_cols

# --- The Custom Collate Function for DataLoader ---
def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader to handle raw Python objects.
    It takes a list of samples (dictionaries from __getitem__) and
    converts them into a dictionary of lists suitable for batch processing
    by the new Preprocessor modules.
    """
    collated_batch = {}

    # Identify all keys from the first item in the batch
    # (assuming all items have the same keys for simplicity)
    sample_keys = batch[0].keys()

    for key in sample_keys:
        if key == 'labels':
            # Labels are already tensors, just stack them
            collated_batch[key] = torch.stack([item[key] for item in batch])
        elif key == 'tabular_data':
            # Tabular data are lists of floats, convert to tensor and stack
            collated_batch[key] = torch.tensor([item[key] for item in batch], dtype=torch.float)
        elif key in ['visit_keywords', 'keyword_tags_hangul', 'image_links']:
            # These are lists of lists of strings, keep as lists of lists
            collated_batch[key] = [item[key] for item in batch]
        else:
            # All other keys (single text strings) are collected as lists of strings
            collated_batch[key] = [item[key] for item in batch]

    return collated_batch

###################################################################################################################################################################
# --- Text Cleaning Function ---
url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

def clean_text(x):
    """
    Cleans a given text string by:
    - Removing characters not in the specified pattern (non-standard characters)
    - Removing URLs
    - Stripping leading/trailing whitespace
    - Normalizing repeated characters (e.g., 'ㅋㅋㅋㅋ' to 'ㅋㅋ')
    Emojis will be retained.
    """
    if not isinstance(x, str): # Ensure input is a string
        return ""
    x = url_pattern.sub('', x) # Remove URLs
    x = x.strip() # Remove leading/trailing whitespace
    x = repeat_normalize(x, num_repeats=2) # Normalize repeated characters
    return x

def load_and_resize_image(image_path: str, target_size: tuple = (224, 224)) -> Image.Image | None:
    """
    Loads an image from a specified file path and resizes it to the target dimensions.

    Args:
        image_path (str): The full path to the image file.
        target_size (tuple): A tuple (width, height) specifying the desired output size.
                             Defaults to (224, 224), a common size for many vision models.

    Returns:
        PIL.Image.Image: The resized PIL Image object if successful.
        None: If the image file is not found or an error occurs during loading/resizing.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    try:
        img = Image.open(image_path).convert('RGB') # Ensure image is in RGB format
        img = img.resize(target_size, Image.Resampling.LANCZOS) # Use LANCZOS for high-quality downsampling
        return img
    except Exception as e:
        print(f"Error loading or resizing image from {image_path}: {e}")
        return None

class KcELECTRATextEncoder(torch.nn.Module):
    def __init__(self, model_name="monologg/koelectra-small-discriminator"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.embedding_dim = self.model.config.hidden_size # Store embedding dimension

    def forward(self, input_ids, attention_mask):
        original_shape = input_ids.shape
        if len(original_shape) == 3: # (batch_size, num_tags, sequence_length)
            batch_size, num_tags, seq_len = original_shape
            flat_input_ids = input_ids.view(-1, seq_len)
            flat_attention_mask = attention_mask.view(-1, seq_len)

            output = self.model(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
            flat_embeddings = output.last_hidden_state[:, 0, :] # (batch_size * num_tags, embed_dim)
            embeddings = flat_embeddings.view(batch_size, num_tags, -1)
            return embeddings
        elif len(original_shape) == 2: # (batch_size, sequence_length)
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = output.last_hidden_state[:, 0, :]
            return embeddings
        else:
            raise ValueError(f"Unexpected input_ids shape: {original_shape}")

class CLIPImageEncoder(torch.nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initializes the CLIPImageEncoder.

        Args:
            model_name (str): The name of the pre-trained CLIP model to load.
                              Defaults to "openai/clip-vit-base-patch32".
        """
        super().__init__()
        # Load only the vision model part of CLIP.
        # CLIPVisionModel is more direct if you only need the vision component.
        model = CLIPVisionModel.from_pretrained(model_name)

        # Ensure we got a CLIPVisionModel instance
        assert isinstance(model, CLIPVisionModel), \
            "The loaded model is not a CLIPVisionModel instance."
        self.model = model

        # Store the embedding dimension for potential external use
        self.embedding_dim = self.model.config.hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encodes image pixel values into CLIP image features.

        Args:
            pixel_values (torch.Tensor): A tensor of image pixel values.
                                         Expected shape: (batch_size, channels, height, width)
                                         If a single image, you might pass (1, channels, height, width)
                                         or unsqueeze it to (1, C, H, W) before passing.

        Returns:
            torch.Tensor: The encoded image features.
                          Shape: (batch_size, embedding_dim)
        """
        # The CLIPVisionModel directly accepts a batch of pixel values
        # and returns a BaseModelOutputWithPooling.
        # We extract the 'pooler_output' which is the [CLS] token's pooled representation.
        image_features = self.model(pixel_values=pixel_values).pooler_output

        return image_features

class TabularEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.embedding_dim = output_dim

    def forward(self, tabular_data):
        return self.relu(self.fc(tabular_data))
##########################################################################################################################################
# --- Attention Modules ---
class SimpleCrossAttention(nn.Module):
    """
    A simplified cross-attention module designed for single-vector queries
    attending to single or multi-vector keys/values.
    It uses `nn.MultiheadAttention` internally.
    """
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # MultiheadAttention expects batch_first=True for (batch, seq_len, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query_emb, key_value_emb):
        """
        Calculates attention where `query_emb` attends to `key_value_emb`.

        Args:
            query_emb (torch.Tensor): The query embedding, shape `(batch_size, embed_dim)`.
            key_value_emb (torch.Tensor): The key/value embeddings, shape `(batch_size, N, embed_dim)`
                                          where N is 1 for single auxiliary texts, or `max_tags`
                                          for multi-tag auxiliary texts.

        Returns:
            torch.Tensor: The context vector, shape `(batch_size, embed_dim)`.
        """
        # Unsqueeze query_emb to (batch_size, 1, embed_dim) to act as a sequence of length 1
        query = query_emb.unsqueeze(1)

        attn_output, _ = self.attention(query, key_value_emb, key_value_emb)
        return attn_output.squeeze(1)