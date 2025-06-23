import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor
import os
from PIL import Image
from navermap_utils import KcELECTRATextEncoder, CLIPImageEncoder, SimpleCrossAttention, clean_text
##########################################################################################################################################
# --- The TextPreprocessor Module ---
class TextPreprocessor(torch.nn.Module):
    """
    Module to handle text cleaning, tokenization, chunking, and feature extraction
    for all text inputs in a batch. It then applies a specified fusion strategy.
    """
    def __init__(self, config):
        super().__init__()
        self.strategy = config['strategy']
        params = config['params']

        # Tokenizer setup
        tokenizer_model_name = params.get("tokenizer_model_name", "beomi/KcELECTRA-base")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        except Exception:
            print(f"Warning: Could not load '{tokenizer_model_name}' tokenizer. Using 'bert-base-uncased' as fallback.")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
        # Text dimensions and chunking parameters
        self.text_encoder_max_len = params.get("text_encoder_max_len", 512)
        self.aux_text_max_len = params.get("aux_text_max_len", 32)
        self.max_tags = params.get("max_tags", 5)
        self.review_text_chunk_overlap = params.get("review_text_chunk_overlap", 50)
        
        # Output embedding dimension for this preprocessor's fused text output
        self.fusion_embed_dim = params.get("fusion_embed_dim", 768)
        self.attention_heads = params.get("attention_heads", 4)
        self.attention_layers = params.get("attention_layers", 1) # Not used by current options, but good to have.
        self.dropout = params.get("dropout", 0.1)

        # Initialize the core text feature extractor
        self.text_encoder = KcELECTRATextEncoder(tokenizer_model_name)
        self.encoder_output_dim = self.text_encoder.embedding_dim # e.g., 768

        # Projection layers to bring all encoder outputs to fusion_embed_dim
        self.proj_review_text = nn.Linear(self.encoder_output_dim, self.fusion_embed_dim)
        self.proj_store_naver_name = nn.Linear(self.encoder_output_dim, self.fusion_embed_dim)
        self.proj_visit_keywords = nn.Linear(self.encoder_output_dim, self.fusion_embed_dim)
        self.proj_keyword_tags_hangul = nn.Linear(self.encoder_output_dim, self.fusion_embed_dim)
        self.proj_category = nn.Linear(self.encoder_output_dim, self.fusion_embed_dim)

        # Initialize attention modules based on strategy, if needed
        if self.strategy in ["option1_cross_attention_results", "option3_cross_attention_mean_pool"]:
            self.attention_store_naver_name = SimpleCrossAttention(self.fusion_embed_dim, num_heads=self.attention_heads)
            self.attention_visit_keywords = SimpleCrossAttention(self.fusion_embed_dim, num_heads=self.attention_heads)
            self.attention_keyword_tags_hangul = SimpleCrossAttention(self.fusion_embed_dim, num_heads=self.attention_heads)
            self.attention_category = SimpleCrossAttention(self.fusion_embed_dim, num_heads=self.attention_heads)
            
            if self.strategy == "option1_cross_attention_results":
                self.proj_option1_final = nn.Linear(5 * self.fusion_embed_dim, self.fusion_embed_dim)
            elif self.strategy == "option3_cross_attention_mean_pool":
                self.proj_option3_final = nn.Linear(2 * self.fusion_embed_dim, self.fusion_embed_dim)
        
        elif self.strategy == "option2_mean_pooling_others":
            self.proj_option2_final = nn.Linear(2 * self.fusion_embed_dim, self.fusion_embed_dim)


    def _tokenize_and_chunk_long_text(self, texts: list[str], max_length: int, chunk_overlap: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch_input_ids = []
        batch_attention_masks = []
        
        for text in texts:
            cleaned_text = clean_text(text)
            tokenized = self.tokenizer(
                cleaned_text,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt'
            )
            # Ensure these are long before further processing
            input_ids = tokenized['input_ids'].squeeze(0).to(torch.long) # Explicit conversion
            attention_mask = tokenized['attention_mask'].squeeze(0).to(torch.long) # Explicit conversion
            
            if input_ids.size(0) <= max_length:
                padded_input_ids = F.pad(input_ids, (0, max_length - input_ids.size(0)), value=self.tokenizer.pad_token_id)
                padded_attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.size(0)), value=0)
                batch_input_ids.append(padded_input_ids.unsqueeze(0))
                batch_attention_masks.append(padded_attention_mask.unsqueeze(0))
            else:
                chunks_input_ids = []
                chunks_attention_masks = []
                stride = max_length - chunk_overlap
                
                for i in range(0, input_ids.size(0), stride):
                    chunk_input_ids = input_ids[i:i + max_length]
                    chunk_attention_mask = attention_mask[i:i + max_length]
                    if chunk_input_ids.size(0) < max_length:
                        chunk_input_ids = F.pad(chunk_input_ids, (0, max_length - chunk_input_ids.size(0)), value=self.tokenizer.pad_token_id)
                        chunk_attention_mask = F.pad(chunk_attention_mask, (0, max_length - chunk_attention_mask.size(0)), value=0)
                    chunks_input_ids.append(chunk_input_ids)
                    chunks_attention_masks.append(chunk_attention_mask) # <-- This was `chunks_attention_masks.append(torch.stack(chunks_attention_masks))` in a previous version, but now it's correctly `chunk_attention_mask`
                
                batch_input_ids.append(torch.stack(chunks_input_ids))
                batch_attention_masks.append(torch.stack(chunks_attention_masks))

        max_num_chunks = max([item.size(0) for item in batch_input_ids])
        
        final_batch_input_ids = []
        final_batch_attention_masks = []
        
        for input_ids_item, attention_mask_item in zip(batch_input_ids, batch_attention_masks):
            num_chunks_item = input_ids_item.size(0)
            if num_chunks_item < max_num_chunks:
                padding_chunk_input_ids = torch.full((max_num_chunks - num_chunks_item, max_length), self.tokenizer.pad_token_id, dtype=torch.long)
                padding_chunk_attention_mask = torch.full((max_num_chunks - num_chunks_item, max_length), 0, dtype=torch.long)
                input_ids_item = torch.cat([input_ids_item, padding_chunk_input_ids], dim=0)
                attention_mask_item = torch.cat([attention_mask_item, padding_chunk_attention_mask], dim=0)
            
            final_batch_input_ids.append(input_ids_item)
            final_batch_attention_masks.append(attention_mask_item)

        return torch.stack(final_batch_input_ids), torch.stack(final_batch_attention_masks)


    def _tokenize_single_text(self, texts: list[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes a list of single text strings, applies cleaning."""
        cleaned_texts = [clean_text(text) for text in texts]
        encoded = self.tokenizer(
            cleaned_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # Ensure outputs are long integers
        return encoded['input_ids'].to(torch.long), encoded['attention_mask'].to(torch.long)

    def _tokenize_multi_tag_text(self, list_of_tag_lists: list[list[str]], max_tags: int, max_tag_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenizes lists of lists of tags, applies cleaning.
        Outputs (batch_size, max_tags, seq_len) tensors.
        """
        batch_input_ids = []
        batch_attention_masks = []
        
        for tags_for_single_item in list_of_tag_lists:
            item_input_ids = []
            item_attention_masks = []
            
            # Handle empty tag lists for a sample
            if not tags_for_single_item:
                # Append a single padding tag to ensure a dimension of 1 for that sample
                # This makes subsequent stacking easier without breaking shape
                pad_shape = (max_tag_length,)
                padding_ids = torch.full(pad_shape, self.tokenizer.pad_token_id, dtype=torch.long)
                padding_mask = torch.full(pad_shape, 0, dtype=torch.long)
                item_input_ids.append(padding_ids)
                item_attention_masks.append(padding_mask)
            else:
                for tag in tags_for_single_item[:max_tags]:
                    # Ensure tag is treated as a string, especially if it's not always
                    cleaned_tag = clean_text(str(tag)) 
                    encoded_tag = self.tokenizer(
                        cleaned_tag,
                        max_length=max_tag_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    # Ensure outputs are long integers
                    item_input_ids.append(encoded_tag['input_ids'].squeeze(0).to(torch.long))
                    item_attention_masks.append(encoded_tag['attention_mask'].squeeze(0).to(torch.long))

            num_actual_tags = len(item_input_ids)
            if num_actual_tags < max_tags:
                pad_shape = (max_tag_length,)
                # Ensure dtype=torch.long for these full tensors
                padding_ids = torch.full(pad_shape, self.tokenizer.pad_token_id, dtype=torch.long)
                padding_mask = torch.full(pad_shape, 0, dtype=torch.long)

                for _ in range(max_tags - num_actual_tags):
                    item_input_ids.append(padding_ids)
                    item_attention_masks.append(padding_mask)

            batch_input_ids.append(torch.stack(item_input_ids).to(torch.long)) # Explicit conversion
            batch_attention_masks.append(torch.stack(item_attention_masks).to(torch.long)) # Explicit conversion
            
        return torch.stack(batch_input_ids).to(torch.long), torch.stack(batch_attention_masks).to(torch.long)


    def forward(self, raw_batch_data: dict) -> torch.Tensor|None:
        # --- Phase 1: Cleaning, Tokenization, and Initial Feature Extraction ---
        review_input_ids, review_attention_mask = self._tokenize_and_chunk_long_text(
            raw_batch_data['review_text'], self.text_encoder_max_len, self.review_text_chunk_overlap
        )
        store_naver_name_input_ids, store_naver_name_attention_mask = self._tokenize_single_text(
            raw_batch_data['store_naver_name'], self.aux_text_max_len
        )
        # FIX: Category should be treated as multi-tag if NavermapReviewDataset defines it as 'list_of_strings'
        category_input_ids, category_attention_mask = self._tokenize_multi_tag_text( # <-- CHANGED from _tokenize_single_text
            raw_batch_data['category'], 1, self.aux_text_max_len # Assuming 'category' has 1 tag usually
        )
        visit_keywords_input_ids, visit_keywords_attention_mask = self._tokenize_multi_tag_text(
            raw_batch_data['visit_keywords'], self.max_tags, self.aux_text_max_len
        )
        keyword_tags_hangul_input_ids, keyword_tags_hangul_attention_mask = self._tokenize_multi_tag_text(
            raw_batch_data['keyword_tags_hangul'], self.max_tags, self.aux_text_max_len
        )

        # Get embeddings from the text encoder
        batch_size, num_chunks, _ = review_input_ids.shape
        review_emb_all_chunks = self.text_encoder(
            review_input_ids.view(-1, self.text_encoder_max_len),
            review_attention_mask.view(-1, self.text_encoder_max_len)
        ).view(batch_size, num_chunks, self.encoder_output_dim)
        review_emb = torch.mean(review_emb_all_chunks, dim=1) # (batch_size, encoder_output_dim)


        store_naver_name_emb = self.text_encoder(store_naver_name_input_ids, store_naver_name_attention_mask)
        
        # FIX: Category embedding always needs mean pooling after encoding, since it's now multi-tag
        category_emb_raw = self.text_encoder(category_input_ids, category_attention_mask)
        category_emb = torch.mean(category_emb_raw, dim=1) # (batch_size, encoder_output_dim)


        visit_keywords_emb = self.text_encoder(visit_keywords_input_ids, visit_keywords_attention_mask)
        keyword_tags_hangul_emb = self.text_encoder(keyword_tags_hangul_input_ids, keyword_tags_hangul_attention_mask)

        # --- Phase 2: Project embeddings to common fusion_embed_dim ---
        review_emb_proj = self.proj_review_text(review_emb)
        store_naver_name_emb_proj = self.proj_store_naver_name(store_naver_name_emb)
        category_emb_proj = self.proj_category(category_emb) # category_emb is now always (batch_size, embed_dim)
        
        # Mean pool multi-tag embeddings across the max_tags dimension BEFORE projection
        visit_keywords_pooled_emb = torch.mean(visit_keywords_emb, dim=1)
        visit_keywords_emb_proj = self.proj_visit_keywords(visit_keywords_pooled_emb)

        keyword_tags_hangul_pooled_emb = torch.mean(keyword_tags_hangul_emb, dim=1)
        keyword_tags_hangul_emb_proj = self.proj_keyword_tags_hangul(keyword_tags_hangul_pooled_emb)


        # --- Phase 3: Apply text fusion strategy ---
        fused_text_features = None

        if self.strategy == "option1_cross_attention_results":
            # These inputs for SimpleCrossAttention's key_value_emb MUST be (batch_size, N, embed_dim)
            attended_store_naver_name_emb = self.attention_store_naver_name(review_emb_proj, store_naver_name_emb_proj.unsqueeze(1))
            attended_visit_keywords_emb = self.attention_visit_keywords(review_emb_proj, visit_keywords_emb_proj.unsqueeze(1)) # <-- FIX: Added .unsqueeze(1)
            attended_keyword_tags_hangul_emb = self.attention_keyword_tags_hangul(review_emb_proj, keyword_tags_hangul_emb_proj.unsqueeze(1)) # <-- FIX: Added .unsqueeze(1)
            attended_category_emb = self.attention_category(review_emb_proj, category_emb_proj.unsqueeze(1)) # This was already correct as category_emb_proj is 2D


            fused_text_features = torch.cat([
                review_emb_proj,
                attended_store_naver_name_emb,
                attended_visit_keywords_emb,
                attended_keyword_tags_hangul_emb,
                attended_category_emb
            ], dim=-1) # Shape: (batch_size, 5 * fusion_embed_dim)
            
            fused_text_features = self.proj_option1_final(fused_text_features) # Project back to fusion_embed_dim


        elif self.strategy == "option2_mean_pooling_others":
            aux_texts_to_pool = torch.stack([
                store_naver_name_emb_proj,
                visit_keywords_emb_proj,
                keyword_tags_hangul_emb_proj,
                category_emb_proj
            ], dim=1) # Shape: (batch_size, 4, fusion_embed_dim)
            
            pooled_aux_features = torch.mean(aux_texts_to_pool, dim=1) # Shape: (batch_size, fusion_embed_dim)

            fused_text_features = torch.cat([
                review_emb_proj,
                pooled_aux_features
            ], dim=-1) # Shape: (batch_size, 2 * fusion_embed_dim)
            
            fused_text_features = self.proj_option2_final(fused_text_features) # Project back to fusion_embed_dim


        elif self.strategy == "option3_cross_attention_mean_pool":
            attended_store_naver_name_emb = self.attention_store_naver_name(review_emb_proj, store_naver_name_emb_proj.unsqueeze(1))
            attended_visit_keywords_emb = self.attention_visit_keywords(review_emb_proj, visit_keywords_emb_proj.unsqueeze(1)) # <-- FIX: Added .unsqueeze(1)
            attended_keyword_tags_hangul_emb = self.attention_keyword_tags_hangul(review_emb_proj, keyword_tags_hangul_emb_proj.unsqueeze(1)) # <-- FIX: Added .unsqueeze(1)
            attended_category_emb = self.attention_category(review_emb_proj, category_emb_proj.unsqueeze(1)) # This was already correct


            attended_aux_texts_to_pool = torch.stack([
                attended_store_naver_name_emb,
                attended_visit_keywords_emb,
                attended_keyword_tags_hangul_emb,
                attended_category_emb
            ], dim=1) # Shape: (batch_size, 4, fusion_embed_dim)
            
            pooled_attended_aux_features = torch.mean(attended_aux_texts_to_pool, dim=1) # Shape: (batch_size, fusion_embed_dim)

            fused_text_features = torch.cat([
                review_emb_proj,
                pooled_attended_aux_features
            ], dim=-1) # Shape: (batch_size, 2 * fusion_embed_dim)
            
            fused_text_features = self.proj_option3_final(fused_text_features) # Project back to fusion_embed_dim
            

        else:
            raise ValueError(f"Unknown text preprocessing strategy: {self.strategy}")
        
        return fused_text_features
    

class ImagePreprocessor(torch.nn.Module):
    """
    Module to handle image loading, preprocessing with CLIP's processor,
    feature extraction with CLIPImageEncoder, and aggregation of multiple images
    based on a configurable strategy.
    """
    def __init__(self, config):
        super().__init__()
        self.strategy = config['strategy']
        params = config['params']

        # Parameters for image processing
        self.max_images_per_sample = params.get("max_images_per_sample", 5)
        self.target_image_size = tuple(params.get("target_image_size", [224, 224]))
        
        # Output embedding dimension for this preprocessor's fused image output
        self.fusion_embed_dim = params.get("fusion_embed_dim", 768)

        # Load CLIP's AutoProcessor (handles resizing, normalization, tensor conversion)
        clip_model_name = params.get("clip_model_name", "openai/clip-vit-base-patch32")
        try:
            self.processor = AutoProcessor.from_pretrained(clip_model_name)
        except Exception:
            print(f"Warning: Could not load CLIP processor for '{clip_model_name}'. Using a dummy processor.")
            class DummyProcessor:
                def __init__(self, target_size=(224, 224)):
                    self.target_size = target_size
                def __call__(self, images, return_tensors="pt", **kwargs): 
                    if isinstance(images, Image.Image): images = [images]
                    dummy_pixel_values = torch.zeros(
                        len(images), 3, self.target_size[0], self.target_size[1]
                    )
                    del return_tensors
                    del kwargs
                    return {'pixel_values': dummy_pixel_values}
            self.processor = DummyProcessor(self.target_image_size)


        # Initialize the CLIP image feature extractor
        self.image_encoder = CLIPImageEncoder(clip_model_name)
        self.encoder_output_dim = self.image_encoder.embedding_dim # e.g., 768

        # Projection layer to map the image encoder's output to fusion_embed_dim
        # This is needed if encoder_output_dim != fusion_embed_dim
        
        # Learnable embedding to represent the case where a sample has NO real images
        # Initialized to zeros, but can be learned during training.
        self.no_sample_images_embedding = nn.Parameter(torch.zeros(self.fusion_embed_dim))


        # --- Final projection layer for each strategy ---
        # The input dimension for this projection depends on the strategy.
        if self.strategy in ["option1_global_mean_pool", "option3_global_max_pool"]:
            self.final_projection = nn.Linear(self.encoder_output_dim, self.fusion_embed_dim)
        elif self.strategy in ["option2_variable_wise_mean_pool", "option4_variable_wise_max_pool"]:
            # Assumes 'image_links' and 'video_thumbnail_links' are the two variables
            self.final_projection = nn.Linear(2 * self.encoder_output_dim, self.fusion_embed_dim)
        else:
            raise ValueError(f"Unknown image preprocessing strategy: {self.strategy}")


    def _load_image(self, image_path: str) -> Image.Image:
        """
        Loads an image from a local file path.
        Returns a PIL Image object or a dummy black image if loading fails.
        """
        if not os.path.exists(image_path) or not os.path.isfile(image_path):
            return Image.new('RGB', self.target_image_size, color = 'black')
        
        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading image from {image_path}: {e}. Creating dummy black image.")
            return Image.new('RGB', self.target_image_size, color = 'black')

    def _process_image_batch_type(self, list_of_links_for_batch: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads, preprocesses, and extracts features for all images of a *single type* in a batch,
        padding as necessary.

        Args:
            list_of_links_for_batch (list[list[str]]): A list where each sublist contains
                                                local image paths for a single sample.
                                                e.g., [['path1_s1', 'path2_s1'], ['path1_s2'], []]

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - image_features_batched: (batch_size, max_images_per_sample, encoder_output_dim)
                - image_presence_masks: (batch_size, max_images_per_sample)
        """
        batch_size = len(list_of_links_for_batch)
        all_processed_pixel_values_flat = []
        image_presence_masks = torch.zeros((batch_size, self.max_images_per_sample), dtype=torch.long)

        for i, item_image_links in enumerate(list_of_links_for_batch):
            current_sample_pixel_values = []
            
            for j, link in enumerate(item_image_links[:self.max_images_per_sample]):
                pil_image = self._load_image(link)
                processed_input = self.processor(images=pil_image, return_tensors="pt")
                current_sample_pixel_values.append(processed_input['pixel_values'].squeeze(0))
                image_presence_masks[i, j] = 1

            num_actual_images = len(current_sample_pixel_values)
            if num_actual_images < self.max_images_per_sample:
                dummy_pad_image_tensor = torch.zeros(3, self.target_image_size[0], self.target_image_size[1], dtype=torch.float)
                for _ in range(self.max_images_per_sample - num_actual_images):
                    current_sample_pixel_values.append(dummy_pad_image_tensor)
            
            all_processed_pixel_values_flat.extend(current_sample_pixel_values)

        # Handle empty `all_processed_pixel_values_flat` list if the entire batch has no images.
        if not all_processed_pixel_values_flat:
            # Return zero tensors of expected shape for a batch of this size
            return (torch.zeros(batch_size, self.max_images_per_sample, self.encoder_output_dim), 
                    image_presence_masks) # Masks will correctly be all zeros

        batched_pixel_values_for_encoder = torch.stack(all_processed_pixel_values_flat)
        image_features_flat = self.image_encoder(batched_pixel_values_for_encoder)

        image_features_batched = image_features_flat.view(
            batch_size, self.max_images_per_sample, self.encoder_output_dim
        )
        return image_features_batched, image_presence_masks

    def _aggregate_global_mean_pool(self, features_batched: torch.Tensor, presence_masks: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Applies global mean pooling over all images (masked), handling samples with no real images."""
        masked_features = features_batched * presence_masks.unsqueeze(-1).float()
        sum_features = masked_features.sum(dim=1)
        
        num_real_items = presence_masks.sum(dim=1, keepdim=True).float() # (batch_size, 1)
        
        # If a sample has no real images (num_real_items is 0 for that sample),
        # its aggregated features would be 0. We replace these with the learnable
        # 'no_sample_images_embedding'.
        
        # Calculate mean for samples with real images
        aggregated_features_with_zeros = sum_features / (num_real_items + 1e-9)

        # Use torch.where to substitute the learnable embedding where num_real_items is 0
        final_aggregated_features = torch.where(
            num_real_items == 0,
            self.no_sample_images_embedding.unsqueeze(0).expand(batch_size, -1).to(features_batched.device),
            aggregated_features_with_zeros
        )
        return final_aggregated_features

    def _aggregate_global_max_pool(self, features_batched: torch.Tensor, presence_masks: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Applies global max pooling over all images (masked), handling samples with no real images."""
        # For max pooling, if a sample has no real images, the max of zeros will be zero.
        # To make it learnable, we'll initialize masked features with a very low number if applicable,
        # or use the learnable embedding for entirely empty samples.
        
        # Set features for non-present images to a very small number to ensure real features dominate.
        # This is important for max pooling as max(0, -ve_val) would be 0.
        # If CLIP features are always >=0, then 0 is fine.
        # Assuming CLIP features are generally non-negative, masking with 0.0 is acceptable.
        
        masked_features = features_batched * presence_masks.unsqueeze(-1).float()

        # Check if any real features exist for each sample
        any_real_images = (presence_masks.sum(dim=1) > 0).unsqueeze(-1) # (batch_size, 1)
        
        # If a sample has NO real images, its max will be a vector of zeros (after masking).
        # We replace this with the learnable 'no_sample_images_embedding'.
        
        # Perform max pooling over the images dimension (dim=1)
        aggregated_features, _ = masked_features.max(dim=1) # (batch_size, encoder_output_dim)

        # Use torch.where to substitute the learnable embedding where no real images were present
        final_aggregated_features = torch.where(
            any_real_images, # Condition: True if any real images were present
            aggregated_features,
            self.no_sample_images_embedding.unsqueeze(0).expand(batch_size, -1).to(features_batched.device)
        )
        return final_aggregated_features

    def forward(self, raw_batch_data: dict) -> torch.Tensor:
        """
        Processes and aggregates image features based on the configured strategy.

        Args:
            raw_batch_data (dict): Dictionary containing lists of image links.
                                   Expected to have at least 'image_links' key.
                                   'video_thumbnail_links' is optional.

        Returns:
            torch.Tensor: A single, fused image embedding for the batch,
                          shape `(batch_size, fusion_embed_dim)`.
        """
        # Robustly get batch_size from a guaranteed input (e.g., assuming text review is always present)
        # If this Preprocessor is used standalone, a batch_size arg might be needed.
        # Assuming batch_size can be inferred from 'image_links' or other primary keys from DataLoader.
        if 'image_links' in raw_batch_data and raw_batch_data['image_links']:
            batch_size = len(raw_batch_data['image_links'])
        elif 'video_thumbnail_links' in raw_batch_data and raw_batch_data['video_thumbnail_links']:
            batch_size = len(raw_batch_data['video_thumbnail_links'])
        else:
            # Fallback for truly empty batches (should ideally not happen with DataLoader)
            # or if both are entirely absent keys. Assuming 0-batch_size means no data.
            # If both keys are missing, we can assume a batch size of 0.
            # In practice, DataLoader usually ensures batch_size > 0.
            batch_size = len(next(iter(raw_batch_data.values()))) if raw_batch_data else 0

        if batch_size == 0:
            return torch.empty(0, self.fusion_embed_dim) # Return empty tensor for empty batch

        # Robustly get image links, defaulting to empty list of lists if key missing or list empty
        image_links = raw_batch_data.get('image_links', [])
        # Ensure it's a list of lists matching batch_size if key exists but content is not list of list
        if not isinstance(image_links, list) or (image_links and not isinstance(image_links[0], list)):
            image_links = [[]] * batch_size # If it's not the expected list of lists

        video_thumbnail_links = raw_batch_data.get('video_thumbnail_links', [])
        if not isinstance(video_thumbnail_links, list) or (video_thumbnail_links and not isinstance(video_thumbnail_links[0], list)):
            video_thumbnail_links = [[]] * batch_size

        # Ensure that if image_links or video_thumbnail_links are empty lists, they match the batch size.
        # This is critical for _process_image_batch_type which expects len(list_of_links_for_batch) == batch_size
        if len(image_links) != batch_size:
            image_links = [[]] * batch_size
        if len(video_thumbnail_links) != batch_size:
            video_thumbnail_links = [[]] * batch_size


        # --- Phase 1: Load, Preprocess, and Extract Features for ALL individual images ---
        # Returns (batch_size, max_images_per_sample, encoder_output_dim) and (batch_size, max_images_per_sample) mask
        image_features, image_presence_masks = self._process_image_batch_type(image_links)
        video_features, video_presence_masks = self._process_image_batch_type(video_thumbnail_links)
        
        # --- Phase 2: Aggregate Features based on Strategy ---
        fused_image_features = None

        if self.strategy == "option1_global_mean_pool":
            # Concatenate all image features and masks into one global set per sample
            all_features_concatenated = torch.cat([image_features, video_features], dim=1)
            all_masks_concatenated = torch.cat([image_presence_masks, video_presence_masks], dim=1)
            fused_image_features = self._aggregate_global_mean_pool(
                all_features_concatenated, all_masks_concatenated, batch_size
            )

        elif self.strategy == "option2_variable_wise_mean_pool":
            # Aggregate images and video thumbnails separately
            agg_image_features = self._aggregate_global_mean_pool(image_features, image_presence_masks, batch_size)
            agg_video_features = self._aggregate_global_mean_pool(video_features, video_presence_masks, batch_size)
            
            # Concatenate the two aggregated vectors
            fused_image_features = torch.cat([agg_image_features, agg_video_features], dim=-1)

        elif self.strategy == "option3_global_max_pool":
            # Concatenate all image features and masks into one global set per sample
            all_features_concatenated = torch.cat([image_features, video_features], dim=1)
            all_masks_concatenated = torch.cat([image_presence_masks, video_presence_masks], dim=1)
            fused_image_features = self._aggregate_global_max_pool(
                all_features_concatenated, all_masks_concatenated, batch_size
            )

        elif self.strategy == "option4_variable_wise_max_pool":
            # Aggregate images and video thumbnails separately using max pooling
            agg_image_features = self._aggregate_global_max_pool(image_features, image_presence_masks, batch_size)
            agg_video_features = self._aggregate_global_max_pool(video_features, video_presence_masks, batch_size)
            
            # Concatenate the two aggregated vectors
            fused_image_features = torch.cat([agg_image_features, agg_video_features], dim=-1)
            
        else:
            raise ValueError(f"Unsupported image preprocessing strategy: {self.strategy}")

        # --- Phase 3: Final Projection ---
        # Project the fused features to the final fusion_embed_dim
        final_embedding = self.final_projection(fused_image_features)

        return final_embedding
