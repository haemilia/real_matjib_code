import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from navermap_utils import KoELECTRATextEncoder, CLIPImageEncoder, TabularEncoder, clean_text
from navermap_model import ClassifierHead
from transformers import AutoTokenizer, CLIPProcessor
from typing import Any, List, Dict, Tuple
####################################################################################################
# Things to include in config
# "image_dir": C://~
# params:
#   tokenizer_model_name: "monologg/koelectra-small-discriminator"
#   review_text_max_len: 512
#   review_text_chunk_overlap: 50
#   aux_text_max_len: 64
#   fusion_embedding_dim: 256
  
#   # Choose your text fusion strategy:
#   text_fusion_strategy: "concat" # or "mean", or "max"

#   target_image_size: [224, 224] # Used by Dataset for dummy images
#   image_encoder_name: "openai/clip-vit-base-patch32"
  
#   # Choose your image fusion strategy:
#   # "mean_all"
#   # "max_all"
#   # "mean_by_col_concat", "max_by_col_concat"
#   image_fusion_strategy: "mean_by_column_concat" 
  
#   fusion_embedding_dim: 128 

  # Global fusion dimension (should be consistent across all modalities)
#   fusion_embedding_dim: 128 

#   # Multi-modal fusion strategy:
#   # Choose from: "concat_proj", "mean_pool", "max_pool"
#   fusion_strategy: "concat_proj" 
  
#   # Classifier Head parameters
#   classifier_hidden_dim: 256 
#   classifier_dropout: 0.1 

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

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        row = self.dataframe.iloc[idx]
        sample_data = {}

        sample_data["post_url"] = row["post_url"]
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
def create_blog_collate_fn(config:Any, device:torch.device):
    """
    Creates custom collate_fn to deal with batching for NaverblogReviewDataset.
    Gathers raw PIL Image objects for ImagePreprocessor to deal with later.
    Gathers raw text for TextPreprocessor to deal with later.
    """
    def blog_collate_fn(batch:List[Dict])->Dict[str, Any]:
        """
        A custom collate_fn to deal with batching NaverblogReviewDataset.
        """
        # Containers for batched data
        batched_texts = {
            "text":[],
            "post_title":[],
            "author":[],
            "blogname":[]
        }
        batched_raw_images_flattened = []
        batched_images_vars_flattened = []
        image_counts_per_sample = []

        batched_tabular_data = []
        batched_labels = []
        batched_post_urls = []

        for sample in batch:
            # Batch text
            batched_texts["text"].append(sample["text"])
            batched_texts["post_title"].append(sample["post_title"])
            batched_texts["author"].append(sample["author"])
            batched_texts["blogname"].append(sample["blogname"])

            # Batch images
            # Flatten image lists into one per sample
            current_sample_raw_images = []
            current_sample_image_vars = []
            for img_list_col in ["img_url", "sticker_url", "vid_thumb_url"]:
                for img_pil in sample[img_list_col]:
                    if isinstance(img_pil, Image.Image):
                        current_sample_raw_images.append(img_pil)
                        current_sample_image_vars.append(img_list_col)

            if not current_sample_raw_images:
                # No images across all fields
                # Add one dummy blank image
                image_size = tuple(config.get("params", {}).get("target_image_size", [224, 224]))
                dummy_img_pil = Image.new('RGB', image_size, color='black')
                batched_raw_images_flattened.append(dummy_img_pil)
                batched_images_vars_flattened.append("img_url")
                image_counts_per_sample.append(1)
            else:
                batched_raw_images_flattened.extend(current_sample_raw_images)
                batched_images_vars_flattened.extend(current_sample_image_vars)
                image_counts_per_sample.append(len(current_sample_raw_images))
            
            # Batch tabular & other information
            batched_tabular_data.append(torch.tensor(sample["tabular_data"], dtype=torch.float))
            batched_labels.append(sample["labels"])
            batched_post_urls.append(sample["post_url"])

        final_batched_tabular_data = torch.stack(batched_tabular_data).to(device)
        final_batched_labels = torch.stack(batched_labels).to(device)

        return {
            "text": batched_texts,
            "images_flat": batched_raw_images_flattened,
            "image_counts": torch.tensor(image_counts_per_sample, dtype=torch.long, device=device),
            "image_vars_flat": batched_images_vars_flattened,
            "tabular": final_batched_tabular_data,
            "labels": final_batched_labels,
            "post_urls": batched_post_urls,
        }
    return blog_collate_fn

class TextPreprocessor(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        params = config.get("params", {})

        # Initialise Tokenizer
        tokenizer_model_name = params.get("tokenizer_model_name", "monologg/koelectra-small-discriminator")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)

        # Get max lengths & chunking params from config
        ## 'text'
        self.post_text_max_len = params.get("post_text_max_len", 512)
        self.post_text_chunk_overlap = params.get("post_text_chunk_overlap", 50)

        ## others ('post_title', 'author', 'blogname')
        self.aux_text_max_len = params.get("aux_text_max_len", 64)

        # Ensure max_length accommodates for special tokens
        if self.post_text_max_len < 3:
            raise ValueError("post_text_max_len must be at least 3 to accommodate for special tokens")
        if self.aux_text_max_len < 3:
            raise ValueError("aux_text_max_len must be at least 3 to accommodate for special tokens")
    def _tokenize_and_chunk_long(self, texts:List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Tokenizes and chunks a batch of long texts. Returns flattened input_ids, attention_mask, and chunk counts
        per original text.
        """
        # Containers
        all_input_ids_flat = []
        all_attention_masks_flat = []
        chunk_counts_per_original_text = [] # how many chunks the text of one sample generated

        for text in texts:
            cleaned_text = clean_text(text)
            # Tokenize full text w/o special tokens initially
            token_ids = self.tokenizer.encode(
                cleaned_text,
                add_special_tokens=False,
                truncation=False
            )
            # Chunking strategy
            max_chunk_tokens = self.post_text_max_len - 2

            if not token_ids or len(token_ids) <= max_chunk_tokens:
                # If text is short enough to be processed with one chunk
                encoded_chunk = self.tokenizer(
                    cleaned_text,
                    add_special_tokens=True,
                    max_length=self.post_text_max_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                all_input_ids_flat.append(encoded_chunk['input_ids'].squeeze(0))
                all_attention_masks_flat.append(encoded_chunk['attention_mask'].squeeze(0))
                chunk_counts_per_original_text.append(1)
            else:
                # Sliding window chunking
                current_text_chunks_input_ids = []
                current_text_chunks_attention_masks = []

                # Stride = max tokens per chunk - overlap
                stride = max_chunk_tokens - self.post_text_chunk_overlap
                if stride <= 0:
                    stride = max_chunk_tokens # No effective overlap if overlap is too large
                for i in range(0, len(token_ids), stride):
                    # One chunk of token ids
                    chunk_token_ids = token_ids[i : i + max_chunk_tokens]

                    # Convert chunk token ids back to string to ensure proper re-encoding
                    chunk_text = self.tokenizer.decode(chunk_token_ids)
                    # This handles cases where incomplete subwords exist at boundaries

                    encoded_chunk = self.tokenizer(
                        chunk_text,
                        add_special_token = True,
                        max_length=self.post_text_max_len,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    current_text_chunks_input_ids.append(encoded_chunk['input_ids'].squeeze(0))
                    current_text_chunks_attention_masks.append(encoded_chunk['attention_masks'].squeeze(0))
                
                # Add chunks for this sample to overall flat list
                all_input_ids_flat.append(current_text_chunks_input_ids)
                all_attention_masks_flat.append(current_text_chunks_attention_masks)
                chunk_counts_per_original_text.append(len(current_text_chunks_input_ids))
            # Stack all flattened chunks into single tensors
            # if all_input_ids_flat is empty, return empty tensors
            if not all_input_ids_flat:
                return(
                    torch.empty(0, self.post_text_max_len, dtype=torch.long),
                    torch.empty(0, self.post_text_max_len, dtype=torch.long),
                    []
                )
            return (torch.stack(all_input_ids_flat),
                    torch.stack(all_attention_masks_flat),
                    chunk_counts_per_original_text)
    def _tokenize_short(self, texts:List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch of texts with padding & truncation"""
        cleaned_texts = [clean_text(text) for text in texts]

        encoded = self.tokenizer(
            cleaned_texts,
            max_length=self.aux_text_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoded['input_ids'], encoded['attention_mask']
    def forward(self, batch_raw_texts:Dict[str, List[str]]) -> Dict[str, Any]:
        """Preprocess raw text batch based on their source.
        Args:
            batch_raw_texts(Dict[str, List[str]]): Dictionary where keys are text column names,
                                                    and values are lists of raw text strings
        Returns:
            Dict[str, Any]: A dictionary containing processed input_ids, attention_masks structured
                            for KoELECTRATextEncoder, and chunk counts for 'text' field.
        """
        processed_data = {}

        # Process 'text' column, which is long and needs to be chunked
        text_input_ids_flat, text_attention_mask_flat, text_chunk_counts = \
            self._tokenize_and_chunk_long(batch_raw_texts['text'])
        processed_data['text_input_ids'] = text_input_ids_flat
        processed_data['text_attention_masks'] = text_attention_mask_flat
        # Counts as a tensor, batch_size x 1
        processed_data['text_chunk_counts'] = torch.tensor(text_chunk_counts, dtype=torch.long)


        # Process other columns
        # 'post_title'
        post_title_input_ids, post_title_attention_masks = self._tokenize_short(batch_raw_texts['post_title'])
        processed_data['post_title_input_ids'] = post_title_input_ids
        processed_data['post_title_attention_masks'] = post_title_attention_masks

        # 'author'
        author_input_ids, author_attention_masks = self._tokenize_short(batch_raw_texts['author'])
        processed_data['author_input_ids'] = author_input_ids
        processed_data['author_attention_masks'] = author_attention_masks

        # 'blogname'
        blogname_input_ids, blogname_attention_masks = self._tokenize_short(batch_raw_texts['blogname'])
        processed_data['blogname_input_ids'] = blogname_input_ids
        processed_data['blogname_attention_masks'] = blogname_attention_masks

        return processed_data

class TextEmbedder(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.text_preprocessor = TextPreprocessor(config)
        self.text_encoder = KoELECTRATextEncoder(config.get("params", {}).get("tokenizer_model_name", "monologg/koelectra-small-discriminator")).to(self.device)
        self.encoder_output_dim = self.text_encoder.embedding_dim
        self.fusion_embedding_dim = config.get("params", {}).get("fusion_embedding_dim", 128)
        self.text_fusion_strategy = config.get("params", {}).get("text_fusion_strategy", "concat")

        self.num_of_total_text_fields = 4
        if self.text_fusion_strategy == "concat":
            self.total_input_text_dim = self.num_of_total_text_fields * self.encoder_output_dim
        elif self.text_fusion_strategy in ["mean", "max"]:
            # pooling
            self.total_input_text_dim = self.encoder_output_dim
        else:
            raise ValueError(f"Unknown text_fusion_strategy: {self.text_fusion_strategy}")
        
        self.text_fusion_projection = torch.nn.Linear(self.total_input_text_dim, self.fusion_embedding_dim)
    def forward(self, batch_raw_texts:Dict[str, List[str]]) -> Dict[str, torch.Tensor]:
        # Preprocess
        preprocessed_data = self.text_preprocessor(batch_raw_texts)

        # Move preprocessed tensors to device before encoding
        for key in preprocessed_data:
            if isinstance(preprocessed_data[key], torch.Tensor):
                preprocessed_data[key] = preprocessed_data[key].to(self.device)

        output_embeddings = []
        batch_size = preprocessed_data['text_chunkcounts'].size(0)

        # If 'text' is empty
        if batch_size == 0:
            # Return empty tensor if entire batch is empty for rest of text features
            return torch.empty(0, self.fusion_embedding_dim, device=self.device)
        # Encode 'text' chunks
        text_emb_all_chunks = self.text_encoder(
            preprocessed_data['text_input_ids'],
            preprocessed_data['text_attention_mask']
        ) # (total_chunks, embed_dim)

        # Mean pool chunks back to original batch size
        pooled_text_embs = []
        current_chunk_idx = 0
        for i in range(batch_size):
            num_chunks = preprocessed_data['text_chunk_counts'][i].item()
            if num_chunks > 0:
                sample_chunks = text_emb_all_chunks[current_chunk_idx : current_chunk_idx + num_chunks]
                pooled_text_embs.append(torch.mean(sample_chunks, dim=0))
            else:
                # 0 chunks
                pooled_text_embs.append(torch.zeros(self.encoder_output_dim, device=self.device))
            current_chunk_idx += num_chunks
        text_emb = torch.stack(pooled_text_embs)

        # Other columns
        post_title_emb = self.text_encoder(
            preprocessed_data['post_title_input_ids'],
            preprocessed_data['post_title_attention_mask']
        )
        author_emb = self.text_encoder(
            preprocessed_data['author_input_ids'],
            preprocessed_data['author_attention_mask']
        )
        blogname_emb = self.text_encoder(
            preprocessed_data['blogname_input_ids'],
            preprocessed_data['blogname_attention_mask']
        )
        indiv_text_embs = [
            text_emb,
            post_title_emb,
            author_emb,
            blogname_emb
        ]
        # fusion strategy
        if self.text_fusion_strategy == "concat":
            # concat along feature dimension
            combined_text_features = torch.cat(indiv_text_embs, dim=1)
        elif self.text_fusion_strategy == "mean":
            # stack along new dimension 
            # then mean-pool across that dimension
            stacked_embs = torch.stack(indiv_text_embs, dim=1) # batch_size, num_of_total_text_fields, embed_dim
            combined_text_features = torch.mean(stacked_embs, dim=1) # batch_size, embed_dim
        elif self.text_fusion_strategy == "max":
            # stack along new dimension
            # then max pool across that dimension
            stacked_embs = torch.stack(indiv_text_embs, dim=1) # batch_size, num_of_total_text_fields, embed_dim
            combined_text_features = torch.max(stacked_embs, dim=1).values # batch_size, embed_dim
        else:
            raise ValueError(f"Invalid text fusion strategy: {self.text_fusion_strategy}")
        # Project to fusion_embedding_dim
        projected_text_features = self.text_fusion_projection(combined_text_features)

        return projected_text_features
    
class ImagePreprocessor(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        params = config.get('params', {})

        self.image_encoder_name = params.get("image_encoder_name", "openai/clip-vit-base-patch32")

        self.processor = CLIPProcessor.from_pretrained(self.clip_model_name)

        # image size after CLIP
        self.processed_image_size = self.processor.feature_extractor.size
    
    def forward(self, raw_images: List[Image.Image]) -> torch.Tensor:
        """
        Preprocesses a list of raw PIL Image objects using CLIP's image processor
        Args:
            raw_images(List[Image.Image]): List of PIL objects.
        Returns:
            torch.Tensor: Batch tensor of preprocessed images (N, C, H, W)
            Returns an empty tensor if raw_images is empty
        """
        if not raw_images: # (channels, height, width)
            return torch.empty(0, 3, self.processed_image_size, self.processed_image_size)
        
        processed_inputs = self.processor(images=raw_images, return_tensors="pt")
        return processed_inputs['pixel_values']
    
class ImageEmbedder(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.image_preprocessor = ImagePreprocessor(config)

        params = config.get('params', {})
        self.image_encoder_name = params.get("image_encoder_name", "openai/clip-vit-base-patch32")

        self.image_encoder = CLIPImageEncoder(model_name=self.image_encoder_name)

        self.encoder_output_dim = self.image_encoder.embedding_dim

        self.fusion_embedding_dim = params.get("fusion_embedding_dim", 128)
        self.image_fusion_strategy = params.get("image_fusion_strategy", "mean_all")

        self.image_link_cols = ["img_url", "sticker_url", "vid_thumb_ur"]
        self.num_image_columns = len(self.image_link_cols)

        if self.image_fusion_strategy in ["mean_all", "max_all"]:
            self.total_input_image_dim = self.encoder_output_dim
        elif self.image_fusion_strategy in ["mean_by_col_concat", "max_by_col_concat"]:
            self.total_input_image_dim = self.num_image_columns * self.encoder_output_dim
        else:
            raise ValueError(f"Unknown image_fusion_strategy: {self.image_fusion_strategy}")
        
        self.image_fusion_projection = torch.nn.Linear(self.total_input_image_dim, self.fusion_embedding_dim)

    def forward(self, batch_raw_images_flattened, image_counts, image_vars_flat):
        # Process the raw images
        processed_image_batch = self.image_preprocessor(batch_raw_images_flattened).to(self.device)
        batch_size = image_counts.size(0)

        # Handle empty batch
        if processed_image_batch.size(0) == 0:
            return torch.empty(0, self.fusion_embedding_dim, device=self.device)
        
        # Encode
        image_embeddings_flat = self.image_encoder(pixel_values=processed_image_batch)
        # image_embeddings_flat shape: (total_num_images_in_batch_across_all_samples, encoder_output_dim)

        # Aggregate image embeddings per original smaple 
        # Follow strategy
        aggregated_sample_embeddings = []
        current_image_idx = 0

        for i in range(batch_size):
            num_images_for_sample = image_counts[i].item()

            sample_images_embs = image_embeddings_flat[current_image_idx : current_image_idx + num_images_for_sample]
            sample_image_vars = image_vars_flat[current_image_idx : current_image_idx + num_images_for_sample]
            if self.image_fusion_strategy in ["mean_all", "max_all"]:
                # Pool across ALL images for the current sample
                if num_images_for_sample > 0:
                    if self.image_fusion_strategy == "mean_all":
                        aggregated_sample_embeddings.append(torch.mean(sample_images_embs, dim=0))
                    elif self.image_fusion_strategy == "max_all":
                        aggregated_sample_embeddings.append(torch.max(sample_images_embs, dim=0).values)
                else:
                    # Should not be hit if collate_fn always adds a dummy image for empty samples
                    aggregated_sample_embeddings.append(torch.zeros(self.encoder_output_dim, device=self.device))
            
            elif self.image_fusion_strategy in ["mean_by_column_concat", "max_by_column_concat"]:
                # Pool within each column, then concatenate
                column_pooled_embs = []
                for col_name in self.image_link_cols:
                    # Filter embeddings belonging to the current column for this sample
                    col_specific_embs = [
                        emb for idx, emb in enumerate(sample_images_embs)
                        if sample_image_vars[idx] == col_name
                    ]
                    
                    if col_specific_embs: # If there are images for this column
                        stacked_col_embs = torch.stack(col_specific_embs)
                        if self.image_fusion_strategy == "mean_by_column_concat":
                            column_pooled_embs.append(torch.mean(stacked_col_embs, dim=0))
                        elif self.image_fusion_strategy == "max_by_column_concat":
                            column_pooled_embs.append(torch.max(stacked_col_embs, dim=0).values)
                    else:
                        # If a column has no images for this sample, append a zero vector
                        column_pooled_embs.append(torch.zeros(self.encoder_output_dim, device=self.device))
                
                # Concatenate the pooled embeddings from each column
                aggregated_sample_embeddings.append(torch.cat(column_pooled_embs, dim=0))
            else:
                raise ValueError(f"Invalid image fusion strategy: {self.image_fusion_strategy}")
            
            current_image_idx += num_images_for_sample
        
        final_image_features = torch.stack(aggregated_sample_embeddings) # (batch_size, total_input_image_dim)

        # Project to the desired fusion_embedding_dim
        projected_image_features = self.image_fusion_projection(final_image_features)
        
        return projected_image_features
    
class TabularEmbedder(torch.nn.Module):
    def __init__(self, config: Any, device: torch.device, tabular_input_dim:int):
        super().__init__()
        self.device = device
        params = config.get("params", {})

        # This input_dim corresponds to the number of numerical tabular columns
        self.tabular_input_dim = tabular_input_dim

        # Define the final fusion embedding dimension
        self.fusion_embedding_dim = params.get("fusion_embedding_dim", 128)

        # Instantiate TabularEncoder
        self.tabular_encoder = TabularEncoder(
            input_dim=self.tabular_input_dim,
            output_dim=self.fusion_embedding_dim
        ).to(self.device)

    def forward(self, batched_tabular_data: torch.Tensor) -> torch.Tensor:
        """
        Encodes tabular data and projects it to the common fusion embedding dimension.

        Args:
            batched_tabular_data (torch.Tensor): A tensor of tabular data for the batch.
                                                 Expected shape: (batch_size, tabular_input_dim).

        Returns:
            torch.Tensor: The projected tabular features (batch_size, fusion_embedding_dim).
        """
        # Ensure input data is on the correct device
        tabular_data_on_device = batched_tabular_data.to(self.device)

        # Encode tabular data using the TabularEncoder
        final_tabular_features = self.tabular_encoder(tabular_data_on_device)
        # Shape: (batch_size, fusion_embedding_dim)

        return final_tabular_features
    
class NaverBlogModel(torch.nn.Module):
    def __init__(self, config, device, tabular_input_dim:int):
        """Initialise the NaverBlogModel
        Args:
            config: Configuration object containing model parameters
            device: device to run the model on
            tabular_input_dim: number of input features for the tabular features; obtained from dataset
        """
        super().__init__()
        self.device() = device
        params = config.get("params", {})

        # Ensure consistent fusion embedding dim across all modalities
        self.fusion_embedding_dim = params.get("fusion_embedding_dim", 128)

        # Modality embedders
        self.text_embedder = TextEmbedder(config, device).to(device)
        self.image_embedder = ImageEmbedder(config, device).to(device)
        self.tabular_embedder = TabularEmbedder(config, device, tabular_input_dim).to(device)

        # Fusion layer 
        self.fusion_strategy = params.get("fusion_strategy", "concat_proj")

        if self.fusion_strategy == "concat_proj":
            self.fusion_layer = torch.nn.Linear(self.fusion_embedding_dim * 3, self.fusion_embedding_dim).to(device)
        elif self.fusion_strategy in ["mean_pool","max_pool"]:
            self.fusion_layer = torch.nnIdentity().to(device)
        else:
            raise ValueError(f"Unknown fusion_strategy: {self.fusion_strategy}")
        self.classifier_input_dim = self.fusion_embedding_dim

        # Classifier head
        self.classifier_hidden_dim = params.get("classifier_hidden_dim", 64)
        self.classifier_output_dim = 1 # binary classification
        self.classifier_dropout = params.get("classifier_dropout", 0.1)

        self.classifier_head = ClassifierHead(
            input_dim = self.classifier_input_dim,
            hidden_dim = self.classifier_hidden_dim,
            output_dim = self.classifier_output_dim,
            dropout = self.classifier_dropout
        ).to(device)
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass through NaverBlogModel
        """
        # Modality embeddings
        text_features = self.text_embedder(batch["text"])
        image_features = self.image_embedder(
            batch["images_flat"],
            batch["image_counts"],
            batch["image_vars_flat"]
        )
        tabular_features = self.tabular_embedder(batch["tabular"])

        # Apply fusion strategy
        if self.fusion_strategy == "concat_proj":
            fused_features_raw = torch.cat(
                (text_features, image_features, tabular_features),
                dim=1
            )
            fused_features = self.fusion_layer(fused_features_raw)
        elif self.fusion_strategy == "mean_pool":
            stacked_features = torch.stack(
                (text_features, image_features, tabular_features),
                dim=0
            )
            fused_features = torch.mean(stacked_features, dim=0)
        elif self.fusion_strategy == "max_pool":
            stacked_features = torch.stack(
                (text_features, image_features, tabular_features),
                dim=0
            )
            fused_features = torch.max(stacked_features, dim=0).values
        else:
            raise ValueError(f"Invalid fusion strategy: {self.fusion_strategy}")
        
        # Classifier
        logits = self.classifier_head(fused_features)
        return logits
    