import torch
import torch.nn as nn
from navermap_preprocess import TextPreprocessor, ImagePreprocessor
from navermap_utils import TabularEncoder
# --- InterModalAttention Module ---
class InterModalAttention(nn.Module):
    """
    Applies self-attention to combined embeddings from different modalities
    (text, image, tabular) using a Transformer Encoder Layer.
    A learnable CLS token is prepended to capture the fused representation.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, num_layers: int = 1, dropout: float = 0.1):
        """
        Initializes the InterModalAttention module.

        Args:
            embed_dim (int): The common embedding dimension for all modalities,
                             and the dimension of the Transformer layers.
            num_heads (int): Number of attention heads for MultiheadAttention.
            num_layers (int): Number of TransformerEncoderLayers to stack.
            dropout (float): Dropout rate.
        """
        super().__init__()

        # Ensure embed_dim is divisible by num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        # Define a single TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4, # Common practice: 4x d_model for feedforward
            dropout=dropout,
            batch_first=True # Input/output tensors will have batch dimension first
        )

        # Stack multiple encoder layers if num_layers > 1
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.embed_dim = embed_dim

        # A learnable 'CLS' token-like embedding to represent the aggregated features
        # This token will attend to all modality embeddings and its output will be the fused representation.
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)) # Shape (1, 1, embed_dim)

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor, tabular_emb: torch.Tensor) -> torch.Tensor:
        """
        Fuses embeddings from different modalities using self-attention.

        Args:
            text_emb (torch.Tensor): Fused text embedding, shape `(batch_size, embed_dim)`.
            image_emb (torch.Tensor): Fused image embedding, shape `(batch_size, embed_dim)`.
            tabular_emb (torch.Tensor): Tabular embedding, shape `(batch_size, embed_dim)`.

        Returns:
            torch.Tensor: Combined and attended feature vector, shape `(batch_size, embed_dim)`.
                          This is the output corresponding to the CLS token.
        """
        batch_size = text_emb.shape[0]

        # Expand the learnable CLS token to match the batch size
        # Shape: (batch_size, 1, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Unsqueeze each modality embedding to add a sequence dimension (length 1)
        # Then concatenate them along the sequence dimension.
        # The order of concatenation can influence performance if positional encodings were used,
        # but for simple fusion of distinct modalities, it's less critical.
        # We concatenate in a fixed order: [CLS, Text, Image, Tabular]
        combined_embeddings_sequence = torch.cat([
            cls_tokens,            # (B, 1, E)
            text_emb.unsqueeze(1), # (B, 1, E)
            image_emb.unsqueeze(1),# (B, 1, E)
            tabular_emb.unsqueeze(1) # (B, 1, E)
        ], dim=1) # Resulting shape: (batch_size, 1 + 3, embed_dim) = (batch_size, 4, embed_dim)

        # Pass the combined sequence through the transformer encoder
        # This allows each modality and the CLS token to attend to all other elements.
        attended_output = self.transformer_encoder(combined_embeddings_sequence)

        # The final fused representation for the batch is typically taken from the
        # output corresponding to the CLS token (the first token in the sequence).
        fused_representation = attended_output[:, 0, :] # Shape (batch_size, embed_dim)

        return fused_representation

# --- Classifier Head Module ---
class ClassifierHead(nn.Module):
    """
    A simple two-layer MLP for binary classification.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, dropout: float = 0.1):
        """
        Initializes the ClassifierHead.

        Args:
            input_dim (int): The dimension of the input feature vector (e.g., fusion_embed_dim).
            hidden_dim (int): The dimension of the hidden layer.
            output_dim (int): The dimension of the output (1 for binary classification).
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim) # Output 1 for binary classification (logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier head.

        Args:
            x (torch.Tensor): The input feature vector (e.g., fused multi-modal embedding),
                              shape `(batch_size, input_dim)`.

        Returns:
            torch.Tensor: The raw logits for binary classification,
                          shape `(batch_size, output_dim)`.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x) # Raw logits, no sigmoid here. Sigmoid will be applied with BCELossWithLogits or later.
        return logits

    #################################### MAIN MODEL #################################################################################
class NaverMapModel(nn.Module):
    """
    The main multi-modal classification model for binary prediction ('is_advert').
    It orchestrates the entire data flow from raw inputs through preprocessors,
    encoders, inter-modal attention, and a classifier head.
    """
    def __init__(self, config: dict, tabular_input_dim: int, device):
        """
        Initializes the MultiModalClassifier.

        Args:
            config (dict): A dictionary containing all configuration parameters
                           for text_preprocessing, image_preprocessing, and model_architecture.
            tabular_input_dim (int): The number of features in the raw tabular data.
        """
        super().__init__()
        self.device = device

        # Extract configurations for different parts of the model
        text_config = config.get('text_preprocessing', {})
        image_config = config.get('image_preprocessing', {})
        model_arch_config = config.get('model_architecture', {})

        # Ensure fusion_embed_dim is consistent across modules that use it
        self.fusion_embed_dim = model_arch_config.get("fusion_embed_dim", 768)
        # Pass fusion_embed_dim to preprocessor configs as well, as they need it for their final projections
        text_config['params']['fusion_embed_dim'] = self.fusion_embed_dim
        image_config['params']['fusion_embed_dim'] = self.fusion_embed_dim


        # --- Instantiate Preprocessors ---
        # These handle raw data to feature vectors for each modality
        self.text_preprocessor = TextPreprocessor(text_config, self.device)
        self.image_preprocessor = ImagePreprocessor(image_config, self.device)

        # --- Instantiate Tabular Encoder ---
        # Note: tabular_output_dim should ideally be fusion_embed_dim for inter-modal attention
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_input_dim,
            output_dim=self.fusion_embed_dim # Ensure tabular output matches fusion_embed_dim
        )

        self.no_intermodal_attention = model_arch_config.get("no_intermodal_attention", False) # Store this flag
        if self.no_intermodal_attention:
            self.inter_modal_attention = None
            # If no_intermodal_attention is True, we need a projection layer
            # after concatenating the outputs of the individual encoders.
            # The input dimension to this projection will be the sum of output dimensions
            # from all encoders (text, image, tabular).
            # Assuming each preprocessor/encoder outputs self.fusion_embed_dim
            total_concat_dim = self.fusion_embed_dim * 3 # For text, image, tabular
            self.concat_projection = nn.Linear(total_concat_dim, self.fusion_embed_dim)
            self.concat_projection_activation = nn.ReLU()
        else:
            self.inter_modal_attention = InterModalAttention(
                embed_dim=self.fusion_embed_dim,
                num_heads=model_arch_config.get("attention_heads", 4),
                num_layers=model_arch_config.get("attention_layers", 1),
                dropout=model_arch_config.get("dropout", 0.1)
            )

        # --- Instantiate Classifier Head ---
        # This takes the fused multi-modal embedding and outputs logits
        self.classifier_head = ClassifierHead(
            input_dim=self.fusion_embed_dim,
            hidden_dim=model_arch_config.get("classifier_hidden_dim", self.fusion_embed_dim // 2),
            output_dim=1, # Binary classification
            dropout=model_arch_config.get("dropout", 0.1)
        )

    def forward(self, raw_batch_data: dict) -> torch.Tensor:
        """
        Forward pass through the multi-modal classifier.

        Args:
            raw_batch_data (dict): A dictionary containing raw inputs for the batch,
                                   as produced by `custom_collate_fn`.
                                   Expected keys: 'review_text', 'store_naver_name',
                                   'visit_keywords', 'keyword_tags_hangul', 'category',
                                   'image_links', 'video_thumbnail_links' (optional),
                                   'tabular_data'.

        Returns:
            torch.Tensor: Raw logits for binary classification, shape `(batch_size, 1)`.
        """
        # --- 1. Preprocessing and Feature Extraction per Modality ---
        # Text Modality
        # TextPreprocessor handles cleaning, tokenization, chunking, and fusion of all text types.
        try:
            text_fused_embedding = self.text_preprocessor(raw_batch_data) # (batch_size, fusion_embed_dim)
        except Exception as e:
            print("Something went wrong during text_preprocessor")
            raise e

        # Image Modality
        # ImagePreprocessor handles loading, CLIP processing, and fusion of all image types.
        try:
            image_fused_embedding = self.image_preprocessor(raw_batch_data) # (batch_size, fusion_embed_dim)
        except Exception as e:
            print("Something went wrong during image_preprocessor")
            raise e

        # Tabular Modality
        # TabularEncoder handles the raw tabular data
        # Ensure tabular data is on the correct device (Preprocessors handle their own device placement via models)
        tabular_data_tensor = raw_batch_data['tabular_data'].to(text_fused_embedding.device)
        tabular_embedding = self.tabular_encoder(tabular_data_tensor) # (batch_size, fusion_embed_dim)

        # --- 2. Inter-Modal Attention / Fusion ---
        # Fuses the processed embeddings from text, image, and tabular modalities
        if self.no_intermodal_attention:
            # Concatenate all embeddings
            concatenated_features = torch.cat(
                (text_fused_embedding, image_fused_embedding, tabular_embedding),
                dim=1
            ) # Shape: (batch_size, 3 * fusion_embed_dim)

            # Project concatenated features to fusion_embed_dim
            fused_multi_modal_features = self.concat_projection_activation(
                self.concat_projection(concatenated_features)
            ) # Shape: (batch_size, fusion_embed_dim)
        else:
            fused_multi_modal_features = self.inter_modal_attention(
                text_fused_embedding,
                image_fused_embedding,
                tabular_embedding
            ) # Shape: (batch_size, fusion_embed_dim)


        # --- 3. Classifier Head ---
        # Outputs raw logits for binary classification
        logits = self.classifier_head(fused_multi_modal_features) # (batch_size, 1)

        return logits