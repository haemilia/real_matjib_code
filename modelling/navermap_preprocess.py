import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import numpy as np
import os
import ast

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

class BaseKcELECTRAEncoder(torch.nn.Module):
    def __init__(self, model_name="beomi/KcELECTRA-base"):
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
    def __init__(self, model_name="openai/clip-vit-base-patch32", max_images=10):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name).vision_model
        self.max_images = max_images
        self.embedding_dim = self.model.config.hidden_size
        # Make placeholder learnable
        self.no_image_embedding = nn.Parameter(torch.zeros(self.embedding_dim))

    def forward(self, pixel_values, attention_mask):
        batch_size, num_images_in_batch, C, H, W = pixel_values.shape
        embedding_dim = self.embedding_dim

        flat_pixel_values = pixel_values.view(-1, C, H, W)
        # Image encoding
        flat_image_features = self.model(pixel_values=flat_pixel_values).pooler_output
        # Reshape into original batch structure
        image_features_batched = flat_image_features.view(batch_size, num_images_in_batch, embedding_dim)

        # Masking the padded areas
        masked_image_features = image_features_batched * attention_mask.unsqueeze(-1).float()

        # Sum over the masked image -> sum of all real image features
        sum_image_features = masked_image_features.sum(dim=1)
        # Counts the number of real images
        num_real_images = attention_mask.sum(dim=1, keepdim=True).float()
        # Calculates mean of real image features
        aggregated_image_features = sum_image_features / (num_real_images + 1e-9)

        # Replace with no_image_embedding where num_real_images is zero
        # Ensure actual 0 padding where there's no image input.
        aggregated_image_features = torch.where(num_real_images == 0,
                                                self.no_image_embedding.unsqueeze(0).expand(batch_size, -1),
                                                aggregated_image_features)
        return aggregated_image_features

class TabularEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.embedding_dim = output_dim

    def forward(self, tabular_data):
        return self.relu(self.fc(tabular_data))



#### DATASET & DATALOADER ###############################################################################
class ReviewDataset(Dataset):
    """
    A PyTorch Dataset for handling the diverse data types in the review DataFrame,
    preparing them for the respective feature encoders (text, image, tabular).
    Updated to reflect new column usage and 'is_advert' as label.
    """
    def __init__(self, dataframe, tokenizer, clip_processor, max_text_len=128, max_tags=5, max_tag_len=32, max_images=5):
        """
        Initializes the dataset with the DataFrame and necessary processors.

        Args:
            dataframe (pd.DataFrame): The input pandas DataFrame containing review data.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for text data (e.g., from KcELECTRA).
            clip_processor (transformers.CLIPProcessor): Processor for image data (from CLIP).
            max_text_len (int): Maximum sequence length for the main review text.
            max_tags (int): Maximum number of keyword tags to process per review.
            max_tag_len (int): Maximum sequence length for each individual keyword tag.
            max_images (int): Maximum number of images to process per review.
        """
        self.dataframe = dataframe.copy() # Operate on a copy to prevent modifying the original DF
        self.tokenizer = tokenizer
        self.clip_processor = clip_processor
        self.max_text_len = max_text_len
        self.max_tags = max_tags
        self.max_tag_len = max_tag_len
        self.max_images = max_images

        # --- Preprocessing applied to the entire DataFrame at initialization ---

        # Text data preprocessing
        self.dataframe['review_text'] = self.dataframe['review_text'].fillna('')
        self.dataframe['store_naver_name'] = self.dataframe['store_naver_name'].fillna('')
        self.dataframe['category'] = self.dataframe['category'].fillna('') # Now an aux text, not label

        # 'keyword_tags_hangul' and 'visit_keywords' can be string representations of lists, or NaN.
        # Fill NaN with '[]' so ast.literal_eval doesn't fail.
        self.dataframe['keyword_tags_hangul'] = self.dataframe['keyword_tags_hangul'].fillna('[]')
        self.dataframe['visit_keywords'] = self.dataframe['visit_keywords'].fillna('[]')


        # Numerical data preprocessing (Imputation for NaNs and type conversion)
        # Separate imputation for 'total_reviews'/'total_images' (fillna with 0.0)
        # and 'rating' (fillna with median)
        count_cols_to_zero_impute = [
            'author_total_reviews',
            'author_total_images'
        ]
        for col in count_cols_to_zero_impute:
            if col in self.dataframe.columns:
                self.dataframe[col] = self.dataframe[col].fillna(0.0)

        # Impute 'rating' with median
        if 'rating' in self.dataframe.columns:
            median_rating_val = self.dataframe['rating'].median()
            self.dataframe['rating'] = self.dataframe['rating'].fillna(median_rating_val)

        # Convert boolean 'is_advert' to float (0.0 or 1.0)
        # This is now the target label
        if 'is_advert' in self.dataframe.columns:
            self.dataframe['is_advert'] = self.dataframe['is_advert'].astype(float)
        else:
            # Fallback if 'is_advert' is missing, for robustness
            self.dataframe['is_advert'] = 0.0 # Or raise error, depending on desired behavior

        # Ensure all other specified numerical columns are indeed numeric.
        other_numerical_cols = [
            'num_of_media', 'visit_count', 'reactions_fun', 'reactions_helpful',
            'reactions_wannago', 'reactions_cool', 'review_year'
        ]
        for col in other_numerical_cols:
            if col in self.dataframe.columns:
                self.dataframe[col] = pd.to_numeric(self.dataframe[col], errors='coerce').fillna(0.0)

        # Store the list of tabular columns actually used
        # Exclude 'review_id', 'store_name', 'store_id', 'store_reply', 'author_nickname'
        # and also 'is_advert' (now the label), and text columns.
        all_possible_tabular_cols = [
            'num_of_media', 'visit_count', 'author_total_reviews',
            'author_total_images', 'reactions_fun', 'reactions_helpful',
            'reactions_wannago', 'reactions_cool', 'review_year', 'rating'
        ]
        self.actual_tabular_cols = [col for col in all_possible_tabular_cols if col in self.dataframe.columns]


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieves and preprocesses a single sample at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing preprocessed tensors for main text,
                  auxiliary texts, images, tabular data, and the binary label.
        """
        row = self.dataframe.iloc[idx]

        # --- 1. Main Text Features (review_text) ---
        review_text = str(row['review_text'])
        encoded_review_text = self.tokenizer(
            review_text,
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        review_input_ids = encoded_review_text['input_ids'].squeeze(0)
        review_attention_mask = encoded_review_text['attention_mask'].squeeze(0)

        # --- 2. Auxiliary Text Features ---

        # Store Naver Name
        store_naver_name = str(row['store_naver_name'])
        encoded_store_naver_name = self.tokenizer(
            store_naver_name,
            max_length=self.max_tag_len, # Using max_tag_len for auxiliary single texts for consistency
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        store_naver_name_input_ids = encoded_store_naver_name['input_ids'].squeeze(0)
        store_naver_name_attention_mask = encoded_store_naver_name['attention_mask'].squeeze(0)

        # Visit Keywords (can be multiple)
        try:
            visit_keywords = ast.literal_eval(row['visit_keywords'])
            if not isinstance(visit_keywords, list):
                visit_keywords = []
        except (ValueError, SyntaxError):
            visit_keywords = []

        tokenized_visit_keywords_input_ids = []
        tokenized_visit_keywords_attention_mask = []
        for kw_idx, keyword in enumerate(visit_keywords[:self.max_tags]):
            encoded_kw = self.tokenizer(
                str(keyword),
                max_length=self.max_tag_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            tokenized_visit_keywords_input_ids.append(encoded_kw['input_ids'].squeeze(0))
            tokenized_visit_keywords_attention_mask.append(encoded_kw['attention_mask'].squeeze(0))

        num_actual_visit_keywords = len(tokenized_visit_keywords_input_ids)
        if num_actual_visit_keywords < self.max_tags:
            pad_shape = (self.max_tag_len,)
            padding_ids = torch.full(pad_shape, self.tokenizer.pad_token_id, dtype=torch.long)
            padding_mask = torch.full(pad_shape, 0, dtype=torch.long)
            for _ in range(self.max_tags - num_actual_visit_keywords):
                tokenized_visit_keywords_input_ids.append(padding_ids)
                tokenized_visit_keywords_attention_mask.append(padding_mask)
        visit_keywords_input_ids = torch.stack(tokenized_visit_keywords_input_ids)
        visit_keywords_attention_mask = torch.stack(tokenized_visit_keywords_attention_mask)

        # Keyword Tags Hangul (can be multiple) - already implemented similarly
        try:
            keyword_tags = ast.literal_eval(row['keyword_tags_hangul'])
            if not isinstance(keyword_tags, list):
                keyword_tags = []
        except (ValueError, SyntaxError):
            keyword_tags = []

        tokenized_tags_input_ids = []
        tokenized_tags_attention_mask = []
        for tag_idx, tag in enumerate(keyword_tags[:self.max_tags]):
            encoded_tag = self.tokenizer(
                str(tag),
                max_length=self.max_tag_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            tokenized_tags_input_ids.append(encoded_tag['input_ids'].squeeze(0))
            tokenized_tags_attention_mask.append(encoded_tag['attention_mask'].squeeze(0))

        num_actual_tags = len(tokenized_tags_input_ids)
        if num_actual_tags < self.max_tags:
            pad_shape = (self.max_tag_len,)
            padding_ids = torch.full(pad_shape, self.tokenizer.pad_token_id, dtype=torch.long)
            padding_mask = torch.full(pad_shape, 0, dtype=torch.long)
            for _ in range(self.max_tags - num_actual_tags):
                tokenized_tags_input_ids.append(padding_ids)
                tokenized_tags_attention_mask.append(padding_mask)
        tags_input_ids = torch.stack(tokenized_tags_input_ids)
        tags_attention_mask = torch.stack(tokenized_tags_attention_mask)

        # Category
        category_text = str(row['category'])
        encoded_category = self.tokenizer(
            category_text,
            max_length=self.max_tag_len, # Using max_tag_len for auxiliary single texts
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        category_input_ids = encoded_category['input_ids'].squeeze(0)
        category_attention_mask = encoded_category['attention_mask'].squeeze(0)


        # --- 3. Image Features (for CLIPImageEncoder) ---
        try:
            image_links = ast.literal_eval(row['image_links'])
            if not isinstance(image_links, list):
                image_links = []
        except (ValueError, SyntaxError):
            image_links = []

        processed_images = []
        image_attention_mask = torch.zeros(self.max_images, dtype=torch.long)

        for i, link in enumerate(image_links[:self.max_images]):
            dummy_image = Image.new('RGB', (224, 224), color = 'black')
            processed_input = self.clip_processor(images=[dummy_image], return_tensors="pt")
            processed_images.append(processed_input['pixel_values'].squeeze(0))
            image_attention_mask[i] = 1

        num_actual_images = len(processed_images)
        if num_actual_images < self.max_images:
            dummy_pad_image = torch.zeros(3, 224, 224, dtype=torch.float)
            for _ in range(self.max_images - num_actual_images):
                processed_images.append(dummy_pad_image)

        image_pixel_values = torch.stack(processed_images)


        # --- 4. Tabular Features (for TabularEncoder) ---
        tabular_data = torch.tensor([row[col] for col in self.actual_tabular_cols], dtype=torch.float)


        # --- 5. Target Label (is_advert) ---
        is_advert_label = torch.tensor(row['is_advert'], dtype=torch.float)


        return {
            'review_input_ids': review_input_ids,
            'review_attention_mask': review_attention_mask,
            'store_naver_name_input_ids': store_naver_name_input_ids,
            'store_naver_name_attention_mask': store_naver_name_attention_mask,
            'visit_keywords_input_ids': visit_keywords_input_ids,
            'visit_keywords_attention_mask': visit_keywords_attention_mask,
            'keyword_tags_hangul_input_ids': tags_input_ids, # Renamed for clarity in model
            'keyword_tags_hangul_attention_mask': tags_attention_mask, # Renamed
            'category_input_ids': category_input_ids,
            'category_attention_mask': category_attention_mask,
            'image_pixel_values': image_pixel_values,
            'image_attention_mask': image_attention_mask,
            'tabular_data': tabular_data,
            'labels': is_advert_label
        }

    def get_tabular_columns(self):
        """Helper to get the list of tabular columns actually used, for TabularEncoder input_dim."""
        return self.actual_tabular_cols


def create_dataloader(dataframe, batch_size=32, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, shuffle=True, num_workers=0, random_state=42):
    """
    Creates PyTorch DataLoaders for train, validation, and test sets.

    Args:
        dataframe (pd.DataFrame): The input pandas DataFrame.
        batch_size (int): Number of samples per batch.
        train_ratio (float): Proportion of the dataset to include in the training split.
        val_ratio (float): Proportion of the dataset to include in the validation split.
        test_ratio (float): Proportion of the dataset to include in the test split.
        shuffle (bool): Whether to shuffle the data at each epoch for training and validation.
        num_workers (int): How many subprocesses to use for data loading. 0 means main process.
        random_state (int): Seed for random splitting for reproducibility.

    Returns:
        tuple: A tuple containing:
            - torch.utils.data.DataLoader: Training DataLoader.
            - torch.utils.data.DataLoader: Validation DataLoader.
            - torch.utils.data.DataLoader: Test DataLoader.
            - int: The dimension of the tabular input features.
    """
    # Ensure ratios sum to 1 (or close enough due to floating point precision)
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Train, validation, and test ratios must sum to 1.0. Current sum: {train_ratio + val_ratio + test_ratio}")

    # First, split into training and temp (validation + test) sets
    train_df, temp_df = train_test_split(dataframe, test_size=(val_ratio + test_ratio), random_state=random_state, stratify=dataframe['is_advert'] if 'is_advert' in dataframe.columns else None)

    # Then, split the temp set into validation and test sets
    # Adjust test_size for the second split: it's relative to temp_df size
    if (val_ratio + test_ratio) > 0:
        val_df, test_df = train_test_split(temp_df, test_size=test_ratio / (val_ratio + test_ratio), random_state=random_state, stratify=temp_df['is_advert'] if 'is_advert' in temp_df.columns else None)
    else: # If no validation or test split is desired
        val_df = pd.DataFrame()
        test_df = pd.DataFrame()


    try:
        tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
    except Exception:
        print("Warning: Could not load 'beomi/KcELECTRA-base' tokenizer. Using 'bert-base-uncased' as fallback.")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    try:
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    except Exception:
        print("Warning: Could not load 'openai/clip-vit-base-patch32' processor. Using a mock processor.")
        class MockCLIPProcessor:
            def __init__(self):
                pass
            def __call__(self, images, return_tensors="pt"):
                if isinstance(images, Image.Image):
                    images = [images]
                dummy_pixels = torch.zeros(len(images), 3, 224, 224, dtype=torch.float)
                return {'pixel_values': dummy_pixels}
        clip_processor = MockCLIPProcessor()

    # Create dataset instances for each split
    train_dataset = ReviewDataset(
        train_df, tokenizer, clip_processor,
        max_text_len=128, max_tags=5, max_tag_len=32, max_images=5
    )
    val_dataset = ReviewDataset(
        val_df, tokenizer, clip_processor,
        max_text_len=128, max_tags=5, max_tag_len=32, max_images=5
    )
    test_dataset = ReviewDataset(
        test_df, tokenizer, clip_processor,
        max_text_len=128, max_tags=5, max_tag_len=32, max_images=5
    )

    # Create DataLoader instances for each split
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True # Typically no shuffle for validation/test
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True # Typically no shuffle for validation/test
    )

    # Get tabular input dimension from one of the datasets (they should be consistent)
    tabular_input_dim = len(train_dataset.get_tabular_columns())

    return train_dataloader, val_dataloader, test_dataloader, tabular_input_dim