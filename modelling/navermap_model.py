import torch
import torch.nn as nn
import torch.nn.functional as F
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
            query_emb (torch.Tensor): The query embedding, typically `review_text_emb`,
                                      shape `(batch_size, embed_dim)`.
            key_value_emb (torch.Tensor): The key/value embeddings, e.g., an auxiliary
                                          text embedding, shape `(batch_size, N, embed_dim)`
                                          where N is 1 for single auxiliary texts, or `max_tags`
                                          for multi-tag auxiliary texts.

        Returns:
            torch.Tensor: The context vector, shape `(batch_size, embed_dim)`,
                          representing the auxiliary features attended by the query.
        """
        # Unsqueeze query_emb to (batch_size, 1, embed_dim) to act as a sequence of length 1
        query = query_emb.unsqueeze(1)

        # key_value_emb is already shaped correctly for keys and values,
        # e.g., (batch_size, 1, embed_dim) for single aux text, or (batch_size, num_tags, embed_dim)
        key = key_value_emb
        value = key_value_emb

        # No key_padding_mask is used here, assuming padding within key_value_emb
        # (e.g., for `max_tags`) is handled by the upstream encoder by outputting zero vectors,
        # or by ensuring attention weights for zero vectors become zero.
        # For simplicity in MultiheadAttention, we provide it as is.
        attn_output, _ = self.attention(query, key, value)
        
        # attn_output shape is (batch_size, 1, embed_dim). Squeeze the sequence dimension.
        return attn_output.squeeze(1)


class InterModalAttention(nn.Module):
    """
    Applies self-attention to combined embeddings from different modalities
    (text, image, tabular) using a Transformer Encoder Layer.
    A learnable CLS token is prepended to capture the fused representation.
    """
    def __init__(self, embed_dim, num_heads=4, num_layers=1, dropout=0.1):
        super().__init__()
        # Define a single TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,          # Dimension of the input features
            nhead=num_heads,            # Number of attention heads
            dim_feedforward=embed_dim * 4, # Dimension of the feedforward network model
            dropout=dropout,            # Dropout value
            batch_first=True            # Input and output tensors are (batch_size, sequence_length, feature_dimension)
        )
        # Stack multiple encoder layers if num_layers > 1
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.embed_dim = embed_dim

        # A learnable 'CLS' token-like embedding to represent the aggregated features
        # This token will be prepended to the sequence of modal embeddings before feeding to transformer
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))


    def forward(self, text_emb, image_emb, tabular_emb):
        """
        Fuses embeddings from different modalities using self-attention.

        Args:
            text_emb (torch.Tensor): Fused text embedding, shape `(batch_size, embed_dim)`.
            image_emb (torch.Tensor): Image embedding, shape `(batch_size, embed_dim)`.
            tabular_emb (torch.Tensor): Tabular embedding, shape `(batch_size, embed_dim)`.

        Returns:
            torch.Tensor: Combined and attended feature vector, shape `(batch_size, embed_dim)`.
                          This is the output corresponding to the CLS token.
        """
        batch_size = text_emb.shape[0]

        # Expand the learnable CLS token to match the batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Unsqueeze each modal embedding to add a sequence dimension (length 1)
        # Then concatenate them along the sequence dimension
        combined_embeddings = torch.cat([
            cls_tokens,            # (B, 1, E)
            text_emb.unsqueeze(1), # (B, 1, E)
            image_emb.unsqueeze(1),# (B, 1, E)
            tabular_emb.unsqueeze(1) # (B, 1, E)
        ], dim=1) # Resulting shape: (batch_size, 1 + 3, embed_dim) = (batch_size, 4, embed_dim)

        # Pass the combined sequence through the transformer encoder
        attended_output = self.transformer_encoder(combined_embeddings)

        # The output corresponding to the CLS token (the first token in the sequence)
        # is taken as the final fused representation of all modalities.
        return attended_output[:, 0, :] # Shape (batch_size, embed_dim)


class MultiModalClassifier(nn.Module):
    """
    A comprehensive multi-modal classification model for binary prediction ('is_advert').
    It integrates textual (main and auxiliary), image, and tabular features,
    applying attention mechanisms at two stages:
    1. Text-Auxiliary Text Attention
    2. Inter-Modal Attention
    Finally, a shallow MLP acts as the binary classifier head.
    """
    def __init__(self, text_model_name="beomi/KcELECTRA-base", clip_model_name="openai/clip-vit-base-patch32",
                 tabular_input_dim=10, # Example, will be determined by DataLoader
                 text_embed_dim=768,   # Default for KcELECTRA-base
                 image_embed_dim=768,  # Default for CLIP-ViT-base-patch32
                 tabular_output_dim=256, # Output dimension for tabular encoder
                 fusion_embed_dim=768, # Common dimension for inter-modal attention
                 attention_heads=4,
                 attention_layers=1,
                 dropout=0.1):
        """
        Initializes the MultiModalClassifier.

        Args:
            text_model_name (str): Name of the pre-trained text model.
            clip_model_name (str): Name of the pre-trained CLIP model.
            tabular_input_dim (int): Input dimension for the tabular encoder.
            text_embed_dim (int): Expected output dimension of the text encoder.
            image_embed_dim (int): Expected output dimension of the image encoder.
            tabular_output_dim (int): Output dimension of the tabular encoder.
            fusion_embed_dim (int): Common embedding dimension for all modalities
                                    before inter-modal attention. Projection layers
                                    will align embeddings to this size.
            attention_heads (int): Number of attention heads for MultiheadAttention.
            attention_layers (int): Number of TransformerEncoderLayers in inter-modal attention.
            dropout (float): Dropout rate.
        """
        super().__init__()

        # --- 1. Feature Encoders ---
        self.text_encoder = BaseKcELECTRAEncoder(model_name=text_model_name)
        self.image_encoder = CLIPImageEncoder(model_name=clip_model_name)
        self.tabular_encoder = TabularEncoder(input_dim=tabular_input_dim, output_dim=tabular_output_dim)

        # Ensure that the actual embedding dimensions match the expected ones,
        # otherwise, add assertion or projection layers here if they differ.
        # For this setup, we assume text_encoder.embedding_dim and image_encoder.embedding_dim
        # are consistent with their defaults (768).

        # Projection layers to align all encoder outputs to the `fusion_embed_dim`
        # for consistent input to attention layers.
        self.proj_review_text = nn.Linear(text_embed_dim, fusion_embed_dim)
        self.proj_store_naver_name = nn.Linear(text_embed_dim, fusion_embed_dim)
        self.proj_visit_keywords = nn.Linear(text_embed_dim, fusion_embed_dim)
        self.proj_keyword_tags_hangul = nn.Linear(text_embed_dim, fusion_embed_dim)
        self.proj_category = nn.Linear(text_embed_dim, fusion_embed_dim)
        self.proj_image = nn.Linear(image_embed_dim, fusion_embed_dim)
        self.proj_tabular = nn.Linear(tabular_output_dim, fusion_embed_dim) # Project tabular to fusion dim

        # --- 2. Attention Layer 1: Text-Auxiliary Text Attention ---
        # Each auxiliary text embedding will be processed by a cross-attention layer
        # where the main review text embedding acts as the query.
        self.attention_store_naver_name = SimpleCrossAttention(fusion_embed_dim, num_heads=attention_heads)
        self.attention_visit_keywords = SimpleCrossAttention(fusion_embed_dim, num_heads=attention_heads)
        self.attention_keyword_tags_hangul = SimpleCrossAttention(fusion_embed_dim, num_heads=attention_heads)
        self.attention_category = SimpleCrossAttention(fusion_embed_dim, num_heads=attention_heads)

        # A final projection layer to combine all text features into a single vector
        # 5 * fusion_embed_dim because we concatenate review_text_proj and 4 attended aux features.
        self.proj_combined_text_features = nn.Linear(5 * fusion_embed_dim, fusion_embed_dim)

        # --- 3. Attention Layer 2: Inter-Modal Attention ---
        # This layer fuses the processed text features, image features, and tabular features.
        self.inter_modal_attention = InterModalAttention(
            embed_dim=fusion_embed_dim,
            num_heads=attention_heads,
            num_layers=attention_layers,
            dropout=dropout
        )

        # --- 4. Classifier Head ---
        # A shallow MLP for binary classification on the final fused features.
        self.classifier = nn.Sequential(
            nn.Linear(fusion_embed_dim, fusion_embed_dim // 2), # Reduce dimension
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_embed_dim // 2, 1), # Output a single logit for binary classification
            nn.Sigmoid() # Apply sigmoid to get a probability between 0 and 1
        )

        self.fusion_embed_dim = fusion_embed_dim # Store for potential external use


    def forward(self,
                review_input_ids, review_attention_mask,
                store_naver_name_input_ids, store_naver_name_attention_mask,
                visit_keywords_input_ids, visit_keywords_attention_mask,
                keyword_tags_hangul_input_ids, keyword_tags_hangul_attention_mask,
                category_input_ids, category_attention_mask,
                image_pixel_values, image_attention_mask,
                tabular_data):
        """
        Forward pass through the multi-modal classifier.

        Args:
            All input tensors as prepared by the ReviewDataset and DataLoader.
            Shapes:
            - review_input_ids: (batch_size, max_text_len)
            - review_attention_mask: (batch_size, max_text_len)
            - store_naver_name_input_ids: (batch_size, max_tag_len)
            - store_naver_name_attention_mask: (batch_size, max_tag_len)
            - visit_keywords_input_ids: (batch_size, max_tags, max_tag_len)
            - visit_keywords_attention_mask: (batch_size, max_tags, max_tag_len)
            - keyword_tags_hangul_input_ids: (batch_size, max_tags, max_tag_len)
            - keyword_tags_hangul_attention_mask: (batch_size, max_tags, max_tag_len)
            - category_input_ids: (batch_size, max_tag_len)
            - category_attention_mask: (batch_size, max_tag_len)
            - image_pixel_values: (batch_size, max_images, C, H, W)
            - image_attention_mask: (batch_size, max_images)
            - tabular_data: (batch_size, tabular_input_dim)

        Returns:
            torch.Tensor: Binary classification probabilities, shape `(batch_size, 1)`.
        """
        # --- 1. Feature Extraction ---
        # Text encoders output (batch_size, embed_dim) or (batch_size, N_tags, embed_dim)
        review_emb = self.text_encoder(review_input_ids, review_attention_mask)
        store_naver_name_emb = self.text_encoder(store_naver_name_input_ids, store_naver_name_attention_mask)
        visit_keywords_emb = self.text_encoder(visit_keywords_input_ids, visit_keywords_attention_mask)
        keyword_tags_hangul_emb = self.text_encoder(keyword_tags_hangul_input_ids, keyword_tags_hangul_attention_mask)
        category_emb = self.text_encoder(category_input_ids, category_attention_mask)

        # Image encoder output (batch_size, embed_dim)
        image_emb = self.image_encoder(image_pixel_values, image_attention_mask)
        
        # Tabular encoder output (batch_size, tabular_output_dim)
        tabular_emb = self.tabular_encoder(tabular_data)

        # --- Project all embeddings to a common dimension (`fusion_embed_dim`) ---
        review_emb_proj = self.proj_review_text(review_emb)
        store_naver_name_emb_proj = self.proj_store_naver_name(store_naver_name_emb)
        visit_keywords_emb_proj = self.proj_visit_keywords(visit_keywords_emb)
        keyword_tags_hangul_emb_proj = self.proj_keyword_tags_hangul(keyword_tags_hangul_emb)
        category_emb_proj = self.proj_category(category_emb)

        image_emb_proj = self.proj_image(image_emb)
        tabular_emb_proj = self.proj_tabular(tabular_emb)


        # --- 2. Attention Layer 1: Text-Auxiliary Text Attention ---
        # `review_emb_proj` acts as the query for each auxiliary text type.
        # The output of each `SimpleCrossAttention` is the attended auxiliary embedding.
        # Note: `unsqueeze(1)` is applied for single-item auxiliary text embeddings
        # to match the (batch_size, seq_len, embed_dim) expectation of MultiheadAttention
        # where seq_len for these is 1. Multi-item auxiliary texts (keywords, tags) are already (B, N, E).
        
        attended_store_naver_name_emb = self.attention_store_naver_name(review_emb_proj, store_naver_name_emb_proj.unsqueeze(1))
        attended_visit_keywords_emb = self.attention_visit_keywords(review_emb_proj, visit_keywords_emb_proj)
        attended_keyword_tags_hangul_emb = self.attention_keyword_tags_hangul(review_emb_proj, keyword_tags_hangul_emb_proj)
        attended_category_emb = self.attention_category(review_emb_proj, category_emb_proj.unsqueeze(1))

        # Combine all processed text features (main review + 4 attended auxiliary texts)
        # Concatenate them along the feature dimension.
        combined_text_features_raw = torch.cat([
            review_emb_proj,
            attended_store_naver_name_emb,
            attended_visit_keywords_emb,
            attended_keyword_tags_hangul_emb,
            attended_category_emb
        ], dim=-1) # Shape: (batch_size, 5 * fusion_embed_dim)

        # Project the concatenated text features back to `fusion_embed_dim`
        final_text_emb = self.proj_combined_text_features(combined_text_features_raw)


        # --- 3. Attention Layer 2: Inter-Modal Attention ---
        # Fuses the `final_text_emb`, `image_emb_proj`, and `tabular_emb_proj`
        fused_features = self.inter_modal_attention(
            final_text_emb,
            image_emb_proj,
            tabular_emb_proj
        ) # Output shape: (batch_size, fusion_embed_dim)

        # --- 4. Classifier Head ---
        # Pass the final fused features through the shallow MLP classifier.
        logits = self.classifier(fused_features) # Output shape: (batch_size, 1)

        return logits