# Code run on colab
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split # type:ignore
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score# type:ignore
import wandb # For experiment tracking
from tqdm.auto import tqdm # For nice progress bars
from navermap_utils import NavermapReviewDataset, custom_collate_fn
from navermap_model import NaverMapModel

# --- Configuration for WandB ---
WANDB_PROJECT_NAME = "navermap-review-classification"
WANDB_ENTITY = "haemilia-"

# --- configuration ---
DATADIR = Path(r"G:\My Drive\Data\naver_search_results")

# --- Helper Functions ---
def re_root_path(img_paths:dict, root_dir:Path):
    new_img_paths = {}
    for review_id, img_dict in img_paths.items():
        new_img_paths[review_id] = {}
        for img_type, img_path_list in img_dict.items():
            new_img_paths[review_id][img_type] = list(map(lambda x: root_dir / x, img_path_list))
    return new_img_paths

def fbeta_score(y_true, y_pred, beta, average='binary'):
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)

    # Ensure no 0 denominator
    if p + r == 0:
        return 0.0

    # F-beta score
    fbeta = (1 + beta**2) * (p * r) / ((beta**2 * p) + r)
    return fbeta

# --- Get Data ---

def get_final_training_dataset(datadir_path=DATADIR):
    labelled = pd.read_parquet(datadir_path / "navermap_reviews_labelled_only.parquet",
                            engine="pyarrow")
    restaurants = pd.read_parquet(datadir_path / "restaurants_table.parquet",
                                engine="pyarrow",
                                columns=["naver_store_id", "category"])
    with_category = pd.merge(labelled, restaurants,
            left_on="store_id", right_on="naver_store_id", how="left")
    with_category.drop(columns=["naver_store_id"], inplace=True)
    with open(datadir_path / "navermap_reviews_labelled_only_local_image_paths.pickle", "rb") as rf:
        img_paths = pickle.load(rf)
    rerooted = re_root_path(img_paths, Path(__file__).parent.parent.parent / "images")
    img_df = pd.DataFrame(rerooted).T.reset_index(names="review_id")
    dropped_df = with_category.drop(columns=["image_links", "video_thumbnail_links"])
    result = pd.merge(dropped_df, img_df, on="review_id")

    # 1st train test split. The train data from here will go on to be split into train and val.
    train_df, test_df = train_test_split(result, test_size=0.2, random_state=68, stratify=result["is_advert"])
    for col in ['image_links', 'video_thumbnail_links']:
        if col in test_df.columns:
            # Apply a lambda function to each element in the column
            # If the element is a list, convert each item in that list to a string
            # Otherwise, convert the element itself to a string
            test_df[col] = test_df[col].apply(
                lambda x: [str(item) for item in x] if isinstance(x, list) else (str(x) if isinstance(x, Path) else x)
            )
    test_df.to_parquet(datadir_path / "navermap_reviews_test.parquet")

    return train_df

def get_sample_training_dataset(datadir_path=DATADIR):
    df = get_final_training_dataset(datadir_path)
    sampled = df.sample(10, random_state=24)
    return sampled

def load_full_dataset(datadir_path=DATADIR):
    """
    Loads the full dataset and processes image paths.
    This function now returns the entire dataframe without splitting it.
    """
    print(f"Loading data from: {datadir_path}")
    labelled = pd.read_parquet(datadir_path / "navermap_reviews_labelled_only.parquet",
                               engine="pyarrow")
    restaurants = pd.read_parquet(datadir_path / "restaurants_table.parquet",
                                  engine="pyarrow",
                                  columns=["naver_store_id", "category"])
    with_category = pd.merge(labelled, restaurants,
                             left_on="store_id", right_on="naver_store_id", how="left")
    with_category.drop(columns=["naver_store_id"], inplace=True)

    # Path to images in Colab environment (assuming they are extracted to /content)
    # This path should be where your image files themselves reside.
    colab_images_root_dir = Path("/content/navermap_reviews_labelled_only_images") # Adjust if different in your Colab setup

    with open(datadir_path / "navermap_reviews_labelled_only_local_image_paths.pickle", "rb") as rf:
        img_paths = pickle.load(rf)

    # Reroot image paths assuming images are extracted/symlinked to colab_images_root_dir
    rerooted = re_root_path(img_paths, colab_images_root_dir)
    img_df = pd.DataFrame(rerooted).T.reset_index(names="review_id")

    # Check if 'image_links' or 'video_thumbnail_links' exist in with_category before dropping
    cols_to_drop = [col for col in ["image_links", "video_thumbnail_links"] if col in with_category.columns]
    dropped_df = with_category.drop(columns=cols_to_drop)

    result = pd.merge(dropped_df, img_df, on="review_id", how="left") # Use left merge to keep all reviews

    # Ensure image_links and video_thumbnail_links are lists of strings if they exist
    for col in ['image_links', 'video_thumbnail_links']:
        if col in result.columns:
            # Fill NaN with empty lists for consistent processing in dataset
            result[col] = result[col].apply(lambda x: x if isinstance(x, list) else [])
            # Convert Path objects inside lists to strings
            result[col] = result[col].apply(
                lambda x: [str(item) for item in x] if isinstance(x, list) else x
            )
        else:
            # If column does not exist, create it as empty lists for consistency
            result[col] = [[] for _ in range(len(result))]

    print(f"Full dataset loaded. Total samples: {len(result)}")
    return result

# --- Model Training ---
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train() # Set model to training mode
    total_loss = 0
    # Use tqdm for a progress bar
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Training")):
        # Move relevant data to device
        inputs = {}
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(device)
            else: # Keep lists of strings/paths on CPU for preprocessors
                inputs[key] = value

        labels = inputs.pop('labels') # Labels must be a separate tensor

        optimizer.zero_grad() # Zero gradients for each batch

        outputs = model(inputs) # Forward pass
        loss = criterion(outputs.squeeze(1), labels) # Calculate loss
        loss.backward() # Backward pass
        optimizer.step() # Update weights

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss
# --- Model Eval ---
def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_probabilities = []
    all_labels = []

    misclassified_samples_data = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Validation")):
            inputs = {}
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(device)
                else:
                    inputs[key] = value

            labels = inputs.pop('labels')

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), labels)
            total_loss += loss.item()

            probabilities = torch.sigmoid(outputs).squeeze(1)
            predictions = (probabilities > 0.5).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            incorrect_mask = (predictions != labels).cpu().numpy()

            for i, is_incorrect in enumerate(incorrect_mask):
                if is_incorrect:
                    sample_info = {
                        "batch_idx_in_epoch": batch_idx,
                        "sample_idx_in_batch": i,
                        "review_text": batch_data.get('review_text', ['N/A'])[i],
                        "store_naver_name": batch_data.get('store_naver_name', ['N/A'])[i],
                        "category": batch_data.get('category', [[]])[i], # Assuming category can be a list
                        "image_links": batch_data.get('image_links', [[]])[i], # List of paths/strings
                        "actual_label": int(labels[i].item()),
                        "predicted_label": int(predictions[i].item()),
                        "predicted_probability": float(probabilities[i].item())
                    }
                    misclassified_samples_data.append(sample_info)

    avg_loss = total_loss / len(dataloader)

    all_predictions_np = np.array(all_predictions)
    all_probabilities_np = np.array(all_probabilities)
    all_labels_np = np.array(all_labels)

    accuracy = np.mean(all_predictions_np == all_labels_np)

    if len(np.unique(all_labels_np)) < 2:
        precision = np.nan
        recall = np.nan
        f1 = np.nan
        roc_auc = np.nan
        pr_auc = np.nan

        print("Warning: Only one class present in validation labels. Precision, Recall, F1, ROC-AUC, PR-AUC will be NaN.")
    else:
        precision = precision_score(all_labels_np, all_predictions_np)
        recall = recall_score(all_labels_np, all_predictions_np)
        f1 = f1_score(all_labels_np, all_predictions_np)
        roc_auc = roc_auc_score(all_labels_np, all_probabilities_np)
        pr_auc = average_precision_score(all_labels_np, all_probabilities_np)

    return avg_loss, accuracy, precision, recall, f1, roc_auc, pr_auc, misclassified_samples_data
def evaluate_on_test_set(test_df: pd.DataFrame, model_path: Path):
    """
    Evaluates a trained model on a given test DataFrame and returns detailed metrics
    and a DataFrame of misclassified samples.

    Args:
        test_df (pd.DataFrame): The DataFrame containing the test data.
                                 It should include all necessary columns
                                 (review_text, store_naver_name, category, image_links, labels).
        model_path (Path): The file path to the saved model checkpoint (.pth file).

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary of evaluation metrics (accuracy, precision, recall, f1, roc_auc, pr_auc).
            - pd.DataFrame: A DataFrame containing details of misclassified samples.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    if not model_path.suffix == '.pth':
        print(f"Warning: Model path {model_path} does not have a .pth extension. Ensure it's a PyTorch checkpoint.")


    print(f"Loading model from: {model_path}")
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu') # Load to CPU first

    # Reconstruct the model based on the config saved in the checkpoint
    # This assumes 'config' was saved in your checkpoint dictionary.
    # If not, you'll need another way to get the config (e.g., pass it as an argument).
    model_config = checkpoint.get('config')
    if model_config is None:
        raise ValueError("Model configuration not found in the checkpoint. Cannot recreate the model architecture.")

    # Determine device for evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation will run on device: {device}")

    # Initialize the model (ensure NaverMapModel is accessible here)
    # The tabular_input_dim is needed here; we can derive it from a dummy dataset if necessary,
    # or ensure it's saved in the config/checkpoint.
    # For now, let's assume NavermapReviewDataset can correctly infer it or it's in config.
    # A safer way might be:
    dummy_dataset = NavermapReviewDataset(test_df.head(1)) # Use a small part of df for structure
    tabular_input_dim = len(dummy_dataset.get_tabular_columns())

    model = NaverMapModel(model_config, tabular_input_dim=tabular_input_dim, device=device).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set model to evaluation mode

    print("Model loaded and set to evaluation mode.")

    # Create Test DataLoader
    test_dataset = NavermapReviewDataset(test_df)
    # Use a reasonable batch size for evaluation, possibly larger than training
    # or fetch from config if it's there
    batch_size = model_config['training'].get('batch_size', 4) # Re-use batch size from training config
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=0)

    criterion = nn.BCEWithLogitsLoss() # Use the same loss function for consistency

    total_loss = 0
    all_predictions = []
    all_probabilities = []
    all_labels = []
    misclassified_samples_data = []

    print("Starting evaluation on test set...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_dataloader, desc="Evaluating on Test Set")):
            inputs = {}
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(device)
                else:
                    inputs[key] = value

            labels = inputs.pop('labels').to(device)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), labels)
            total_loss += loss.item()

            probabilities = torch.sigmoid(outputs).squeeze(1)
            predictions = (probabilities > 0.5).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            incorrect_mask = (predictions != labels).cpu().numpy()

            # Get the actual batch size for this current batch (it might be smaller for the last batch)
            current_batch_size_in_loop = labels.shape[0]

            # Safely retrieve all relevant lists from batch_data for the current batch
            # Ensure they are padded with 'N/A' or empty lists if missing or shorter
            safe_review_ids = batch_data.get('review_id', ['N/A'] * current_batch_size_in_loop)
            if len(safe_review_ids) < current_batch_size_in_loop:
                safe_review_ids.extend(['N/A'] * (current_batch_size_in_loop - len(safe_review_ids)))

            safe_review_texts = batch_data.get('review_text', ['N/A'] * current_batch_size_in_loop)
            if len(safe_review_texts) < current_batch_size_in_loop:
                safe_review_texts.extend(['N/A'] * (current_batch_size_in_loop - len(safe_review_texts)))

            safe_store_naver_names = batch_data.get('store_naver_name', ['N/A'] * current_batch_size_in_loop)
            if len(safe_store_naver_names) < current_batch_size_in_loop:
                safe_store_naver_names.extend(['N/A'] * (current_batch_size_in_loop - len(safe_store_naver_names)))

            # For category and image_links, the default should be an empty list for each sample
            safe_categories = batch_data.get('category', [[] for _ in range(current_batch_size_in_loop)])
            if len(safe_categories) < current_batch_size_in_loop:
                safe_categories.extend([[] for _ in range(current_batch_size_in_loop - len(safe_categories))])

            safe_image_links = batch_data.get('image_links', [[] for _ in range(current_batch_size_in_loop)])
            if len(safe_image_links) < current_batch_size_in_loop:
                safe_image_links.extend([[] for _ in range(current_batch_size_in_loop - len(safe_image_links))])

            for i, is_incorrect in enumerate(incorrect_mask):
                if is_incorrect:
                    sample_info = {
                        "review_id": safe_review_ids[i],
                        "review_text": safe_review_texts[i],
                        "store_naver_name": safe_store_naver_names[i],
                        "category":safe_categories[i],
                        "image_links": safe_image_links[i],
                        "actual_label": int(labels[i].item()),
                        "predicted_label": int(predictions[i].item()),
                        "predicted_probability": float(probabilities[i].item())
                    }
                    misclassified_samples_data.append(sample_info)

    avg_loss = total_loss / len(test_dataloader)

    all_predictions_np = np.array(all_predictions)
    all_probabilities_np = np.array(all_probabilities)
    all_labels_np = np.array(all_labels)

    accuracy = np.mean(all_predictions_np == all_labels_np)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy
    }

    if len(np.unique(all_labels_np)) < 2:
        metrics["precision"] = np.nan
        metrics["recall"] = np.nan
        metrics["f1"] = np.nan
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
        print("Warning: Only one class present in test labels. Some metrics will be NaN.")
    else:
        metrics["precision"] = precision_score(all_labels_np, all_predictions_np)
        metrics["recall"] = recall_score(all_labels_np, all_predictions_np)
        metrics["f1"] = f1_score(all_labels_np, all_predictions_np)
        metrics["roc_auc"] = roc_auc_score(all_labels_np, all_probabilities_np)
        metrics["pr_auc"] = average_precision_score(all_labels_np, all_probabilities_np)

    misclassified_df = pd.DataFrame(misclassified_samples_data)

    print("\n--- Test Set Evaluation Complete ---")
    print("Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.replace('_', ' ').capitalize()}: {value:.4f}")
    print(f"Total misclassified samples: {len(misclassified_df)}")

    return metrics, misclassified_df, all_predictions_np, all_probabilities_np, all_labels_np

# Somewhere to save the resulting model
MODEL_SAVE_DIR = Path(__file__).parent.parent.parent / "models"
MODEL_SAVE_DIR.mkdir(exist_ok=True)
# Version of training loop to be called by sweep
def train_model_with_config(config=None, train_df=None, val_df=None, test_df=None):
    """
    Runs a single training and evaluation trial based on the provided config.
    This function is designed to be called by wandb.agent for sweeps.

    Args:
        config (wandb.config or dict): Hyperparameters and model configuration.
        train_df (pd.DataFrame): Training data for the current trial.
        val_df (pd.DataFrame): Validation data for the current trial.
        test_df (pd.DataFrame): Final unseen test data for evaluation.
    """
    # 1. Initialize WandB run
    # If called by wandb.agent, config will be automatically passed.
    # For standalone test, pass a dict or None for default.
    with wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, config=config) as run:
        # Access config parameters (either from wandb.config or a passed dictionary)
        config = run.config # Ensure we're using the wandb.config object for consistent access

        current_text_strategy = config['text_preprocessing.strategy']
        current_image_strategy = config['image_preprocessing.strategy']
        current_no_intermodal_attention = config['model_architecture.no_intermodal_attention']

        current_combination = (current_text_strategy, current_image_strategy, current_no_intermodal_attention)

        print(f"Starting sweep trial for combination: {current_combination}")
        print(f"Current config: {config}")

        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")

        # Initialize Dataset Loaders
        # Ensure train_df, val_df, test_df are not None
        if train_df is None or val_df is None or test_df is None:
            raise ValueError("train_df, val_df, and test_df must be provided to train_model_with_config.")

        train_dataset = NavermapReviewDataset(train_df)
        val_dataset = NavermapReviewDataset(val_df)

        tabular_input_dim = len(train_dataset.get_tabular_columns())

        batch_size = config['training.batch_size']
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=0)

        # Reconstruct the model config dictionary from flattened sweep config values
        # This is crucial because NaverMapModel expects a nested config dict.
        model_config_for_NaverMapModel = {
            "text_preprocessing": {
                "strategy": current_text_strategy,
                "params": {
                    # Assuming these params are constant or pulled from a default config if not in sweep
                    "tokenizer_model_name": "monologg/koelectra-small-discriminator",
                    "text_encoder_max_len": 512,
                    "aux_text_max_len": 32,
                    "max_tags": 5,
                    "review_text_chunk_overlap": 50,
                    # Accessing flattened parameters
                    "fusion_embed_dim": config['model_architecture.fusion_embed_dim'],
                    "attention_heads": config['model_architecture.attention_heads'],
                    "dropout": config['model_architecture.dropout']
                }
            },
            "image_preprocessing": {
                "strategy": current_image_strategy,
                "params": {
                    # Assuming these params are constant or pulled from a default config
                    "clip_model_name": "openai/clip-vit-base-patch32",
                    "max_images_per_sample": 5,
                    "target_image_size": [224, 224],
                    # Accessing flattened parameters
                    "fusion_embed_dim": config['model_architecture.fusion_embed_dim']
                }
            },
            "model_architecture": {
                # Accessing flattened parameters
                "no_intermodal_attention": current_no_intermodal_attention,
                "fusion_embed_dim": config['model_architecture.fusion_embed_dim'],
                "attention_heads": config['model_architecture.attention_heads'],
                "attention_layers": config['model_architecture.attention_layers'],
                # Calculate classifier_hidden_dim based on ratio
                "classifier_hidden_dim": int(config['model_architecture.fusion_embed_dim'] * config['model_architecture.classifier_hidden_dim_ratio']),
                "dropout": config['model_architecture.dropout']
            },
            "training": {
                # Accessing flattened parameters
                "batch_size": config['training.batch_size'],
                "num_epochs": config['training.num_epochs'],
                "learning_rate": config['training.learning_rate']
            }
        }

        model = NaverMapModel(model_config_for_NaverMapModel, tabular_input_dim=tabular_input_dim, device=device).to(device)
        print("\nNaverMapmodel initialized successfully.")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_config_for_NaverMapModel['training']['learning_rate'])
        criterion = nn.BCEWithLogitsLoss()
        run.watch(model, criterion, log="all", log_freq=100)
        print("WandB is watching model parameters and gradients.")
        
        num_epochs = model_config_for_NaverMapModel['training']['num_epochs']
        best_val_fbeta = -1.0
        best_epoch_idx = -1
        best_val_loss_at_best_fbeta = float('inf')
        best_model_filename = f"model_sweep_{run.id}_best_fbeta.pth"
        best_model_save_path_for_config = MODEL_SAVE_DIR / best_model_filename
        print(f"\n--- Starting Training for {num_epochs} Epochs ---")
        for epoch in range(1, num_epochs + 1):
            train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
            val_loss, val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_pr_auc, _, all_val_predictions_arr, all_val_labels_arr = \
                evaluate_epoch(model, val_dataloader, criterion, device)
            val_fbeta_0_5 = fbeta_score(all_val_labels_arr, all_val_predictions_arr, beta=0.5)
            print(f"Epoch {epoch}/{num_epochs}:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Validation Loss: {val_loss:.4f}")
            print(f"   Validation Accuracy: {val_accuracy:.4f}")
            print(f"   Validation Precision: {val_precision:.4f}")
            print(f"   Validation Recall: {val_recall:.4f}")
            print(f"   Validation F1-score: {val_f1:.4f}")
            print(f"   Validation Fbeta-0.5: {val_fbeta_0_5:.4f}")
            print(f"   Validation ROC-AUC: {val_roc_auc:.4f}")
            print(f"   Validation PR-AUC: {val_pr_auc:.4f}")
            run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "val_roc_auc": val_roc_auc,
                "val_pr_auc": val_pr_auc,
                "val_fbeta_score": val_fbeta_0_5
            })
            if val_fbeta_0_5 > best_val_fbeta:
                best_val_fbeta = val_fbeta_0_5
                best_epoch_idx = epoch
                best_val_loss_at_best_fbeta = val_loss
                checkpoint = {
                    'epoch': best_epoch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_fbeta': best_val_fbeta,
                    'val_loss_at_best_fbeta': val_loss,
                    'config': model_config_for_NaverMapModel
                }
                torch.save(checkpoint, best_model_save_path_for_config)
                print(f"*** New best model saved! Fbeta: {best_val_fbeta:.4f} at Epoch {best_epoch_idx}. Overwriting {best_model_filename} ***")
                run.summary["best_val_fbeta_achieved"] = best_val_fbeta
                run.summary["best_val_epoch"] = best_epoch_idx
                run.summary["best_val_loss_at_best_fbeta"] = best_val_loss_at_best_fbeta
        print("\n--- Training Complete ---")
        print(f"Best Validation F-beta Score for this run: {best_val_fbeta:.4f} at Epoch {best_epoch_idx}")
        print(f"\n--- Evaluating the best model ({best_model_save_path_for_config.name}) on the Final Test Set ---")
        test_metrics, test_misclassified_df, all_predictions_np, all_probabilities_np, all_labels_np = \
            evaluate_on_test_set(test_df, best_model_save_path_for_config)
        print("\nFinal Test Set Metrics (from best model):")
        for metric_name, value in test_metrics.items(): print(f"   Test {metric_name.replace('_', ' ').capitalize()}: {value:.4f}")
        test_fbeta_0_5 = fbeta_score(all_labels_np, all_predictions_np, beta=0.5)
        run.summary["test_loss"] = test_metrics["loss"]
        run.summary["test_accuracy"] = test_metrics["accuracy"]
        run.summary["test_precision"] = test_metrics["precision"]
        run.summary["test_recall"] = test_metrics["recall"]
        run.summary["test_f1"] = test_metrics["f1"]
        run.summary["test_fbeta_0_5"] = test_fbeta_0_5
        run.summary["test_roc_auc"] = test_metrics["roc_auc"]
        run.summary["test_pr_auc"] = test_metrics["pr_auc"]
        run.summary["test_misclassified_samples_count"] = len(test_misclassified_df)
        print("\n--- Generating and Logging Test Set Graphs ---")
        y_true_test = all_labels_np
        y_pred_test = all_predictions_np
        class_names = ['Not Advert', 'Advert']
        if len(np.unique(y_true_test)) > 1:
            run.log({"test_confusion_matrix": wandb.plot.confusion_matrix(y_true=y_true_test, preds=y_pred_test, class_names=class_names)})
            print("Logged interactive test confusion matrix to WandB.")
            y_probas_combined = np.stack((1 - all_probabilities_np, all_probabilities_np), axis=1)
            run.log({"test_roc_curve": wandb.plot.roc_curve(y_true=y_true_test, y_probas=y_probas_combined, labels=class_names)})
            print("Logged interactive test ROC curve to WandB.")
            run.log({"test_pr_curve": wandb.plot.pr_curve(y_true=y_true_test, y_probas=y_probas_combined, labels=class_names)})
            print("Logged interactive test PR curve to WandB.")
        else:
            print("Skipping test plots: Only one class present in test labels.")
        if not test_misclassified_df.empty:
            if 'image_links' in test_misclassified_df.columns:
                test_misclassified_df['wandb_images'] = test_misclassified_df['image_links'].apply(
                    lambda links: [wandb.Image(str(Path(link))) for link in links if link and Path(link).exists()] if isinstance(links, list) else []
                )
            test_misclassified_table = wandb.Table(dataframe=test_misclassified_df)
            run.log({"test_misclassified_samples": test_misclassified_table})
            print(f"Logged {len(test_misclassified_df)} test set misclassified samples to WandB table.")
        else:
            print("No misclassified samples found on the test set for this run.")
        print(f"WandB run '{run.name}' finished and synced.")


# Training loop for architectural experiments
def main(test_df, path_to_configs=DATADIR / "navermap_configs"):
    config_files_to_run = []

    if path_to_configs.is_file() and path_to_configs.suffix in ['.yaml', '.yml']:
        config_files_to_run.append(path_to_configs)
    elif path_to_configs.is_dir():
        for f_path in path_to_configs.glob('*.yaml'):
            config_files_to_run.append(f_path)
        for f_path in path_to_configs.glob('*.yml'): # Also check for .yml extension
            config_files_to_run.append(f_path)
        config_files_to_run = sorted(list(set(config_files_to_run))) # Remove duplicates and sort for consistency
    else:
        raise ValueError(f"Invalid path_to_configs: {path_to_configs}. Must be a .yaml file or a directory containing .yaml files.")

    if not config_files_to_run:
        print(f"No .yaml or .yml files found in {path_to_configs}. Exiting.")
        return

    print(f"Found {len(config_files_to_run)} configuration(s) to run.")

    # Load and split dataset
    df = get_final_training_dataset() # Using sample for initial testing

    # Split data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['is_advert'])



    for config_path in config_files_to_run:
        print(f"\n--- Starting experiment with configuration: {config_path.name} ---")
        # Load configuration for the current experiment
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Initialize WandB run using 'with' statement for proper resource management
        # The run name can be derived from the config file name
        run_name = config_path.stem # Get file name without extension
        with wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, config=config, name=run_name) as run:
            if run is not None:
                print(f"WandB run initialized: {wandb.run.name}")

            train_dataset = NavermapReviewDataset(train_df)
            val_dataset = NavermapReviewDataset(val_df)

            # Get tabular input dimension (should be consistent for both datasets)
            tabular_input_dim = len(train_dataset.get_tabular_columns())

            batch_size = config['training'].get('batch_size', 4) # Get batch size from config or default

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=0)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=0)

            # Device setup
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"\nUsing device: {device}")

            # Initialize the NaverMapModel
            model = NaverMapModel(config, tabular_input_dim=tabular_input_dim, device=device).to(device)
            print("\nNaverMapmodel initialized successfully.")

        # --- Training Setup ---
            # Optimizer: AdamW is a good default for Transformers
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['training'].get('learning_rate', 1e-5))
            # Loss Function: BCEWithLogitsLoss is suitable for binary classification with raw logits
            criterion = nn.BCEWithLogitsLoss()

            # Log model architecture and gradients
            # This should be called once at the beginning of training for each run.
            # It logs the computational graph and optionally gradients.
            # Adjust 'log_freq' as needed.
            if run is not None: # Ensure wandb run is active
                run.watch(model, criterion, log="all", log_freq=100) # Logs architecture and gradient flow
                print("WandB is watching model parameters and gradients.")

            num_epochs = config['training'].get('num_epochs', 5) # Number of training epochs
            # num_epochs = 1 # for testing

            misclassified_data_for_logging = [] # To store misclassified samples from the last epoch

            # --- Initialize best F1 score for this run ---
            best_val_f1 = -1.0
            best_epoch = -1
            best_val_loss_at_best_f1 = float('inf')
            # Define the fixed filename for the best model for this config
            best_model_filename = f"{run_name}_best_model.pth"
            best_model_save_path_for_config = MODEL_SAVE_DIR / best_model_filename

            print(f"\n--- Starting Training for {num_epochs} Epochs ---")
            for epoch in range(num_epochs):
                train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
                # Only collect detailed misclassified data in the last epoch for efficiency
                if epoch == num_epochs - 1:
                    val_loss, val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_pr_auc, last_epoch_misclassified_records = \
                        evaluate_epoch(model, val_dataloader, criterion, device)
                    misclassified_data_for_logging.extend(last_epoch_misclassified_records)
                else:
                    val_loss, val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_pr_auc, _ = \
                        evaluate_epoch(model, val_dataloader, criterion, device)

                print(f"Epoch {epoch+1}/{num_epochs}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Validation Loss: {val_loss:.4f}")
                print(f"  Validation Accuracy: {val_accuracy:.4f}")
                print(f"  Validation Precision: {val_precision:.4f}")
                print(f"  Validation Recall: {val_recall:.4f}")
                print(f"  Validation F1-score: {val_f1:.4f}")
                print(f"  Validation ROC-AUC: {val_roc_auc:.4f}")
                print(f"  Validation PR-AUC: {val_pr_auc:.4f}")


                # Log metrics to WandB
                run.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "val_roc_auc": val_roc_auc,
                    "val_pr_auc": val_pr_auc
                })

                # --- Model Saving Logic (Best F1 Score - Overwriting) ---
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_epoch = epoch + 1
                    best_val_loss_at_best_f1 = val_loss
                    # Save to the fixed filename, overwriting the previous one
                    checkpoint = {
                        'epoch': best_epoch, # Still save which epoch it came from
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_f1': best_val_f1,
                        'val_loss_at_best_f1': val_loss,
                        'config': config
                    }

                    torch.save(checkpoint, best_model_save_path_for_config)
                    print(f"*** New best model saved! F1: {best_val_f1:.4f} at Epoch {best_epoch}. Overwriting {best_model_filename} ***")
                    run.log({"best_val_f1_achieved": best_val_f1, "best_epoch": best_epoch, "epoch": epoch + 1})


            print("\n--- Training Complete ---")
            print(f"Best Validation F1 Score for this run: {best_val_f1:.4f} at Epoch {best_epoch}")

            # Log final best validation metrics to WandB summary
            run.summary["final_best_val_f1"] = best_val_f1
            run.summary["final_best_val_epoch"] = best_epoch
            run.summary["final_best_val_loss_at_best_f1"] = best_val_loss_at_best_f1


            # --- Evaluate Best Model on Test Set ---
            print(f"\n--- Evaluating the best model ({best_model_save_path_for_config.name}) on the Test Set ---")
            test_metrics, test_misclassified_df, all_predictions_np, all_probabilities_np, all_labels_np = evaluate_on_test_set(test_df, best_model_save_path_for_config)

            print("\nTest Set Metrics (from best model):")
            for metric_name, value in test_metrics.items():
                print(f"   Test {metric_name.replace('_', ' ').capitalize()}: {value:.4f}")

            # Log Test Set Metrics to WandB Summary (for easy comparison across runs)
            run.summary["test_loss"] = test_metrics["loss"]
            run.summary["test_accuracy"] = test_metrics["accuracy"]
            run.summary["test_precision"] = test_metrics["precision"]
            run.summary["test_recall"] = test_metrics["recall"]
            run.summary["test_f1"] = test_metrics["f1"]
            run.summary["test_roc_auc"] = test_metrics["roc_auc"]
            run.summary["test_pr_auc"] = test_metrics["pr_auc"]
            run.summary["test_misclassified_samples_count"] = len(test_misclassified_df)

            print("\n--- Generating and Logging Test Set Graphs ---")

            # 1. Confusion Matrix (Option A: wandb.plot.confusion_matrix)
            # Requires raw predictions (0s and 1s) and true labels
            y_true_test = all_labels_np # From evaluate_on_test_set's all_labels_np
            y_pred_test = all_predictions_np # From evaluate_on_test_set's all_predictions_np
            class_names = ['Not Advert', 'Advert'] # Define your class names

            # Check if both classes are present in true labels for confusion matrix
            if len(np.unique(y_true_test)) > 1:
                run.log({"test_confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None, # Use predictions directly if available
                    y_true=y_true_test,
                    preds=y_pred_test,
                    class_names=class_names
                )})
                print("Logged interactive test confusion matrix to WandB.")
            else:
                print("Skipping test confusion matrix: Only one class present in test labels.")

            # 2. ROC Curve (Option A: wandb.plot.roc_curve)
            # Requires probabilities for both classes for wandb.plot.roc_curve when using 'labels'
            y_prob_test_class_1 = all_probabilities_np # This is probability of positive class (1)
            y_prob_test_class_0 = 1 - y_prob_test_class_1

            # Combine into a 2D array: [P(class 0), P(class 1)] for each sample
            y_probas_combined = np.stack((y_prob_test_class_0, y_prob_test_class_1), axis=1)
            y_true_test = all_labels_np

            if len(np.unique(y_true_test)) > 1:
                run.log({"test_roc_curve": wandb.plot.roc_curve(
                    y_true=y_true_test,
                    y_probas=y_probas_combined,
                    labels=class_names # Use class_names if you have multiple columns for y_probas
                )})
                print("Logged interactive test ROC curve to WandB.")
            else:
                print("Skipping test ROC curve: Only one class present in test labels.")

            # 3. Precision-Recall Curve (Option A: wandb.plot.pr_curve)
            # Requires probabilities and true labels
            if len(np.unique(y_true_test)) > 1:
                run.log({"test_pr_curve": wandb.plot.pr_curve(
                    y_true=y_true_test,
                    y_probas=y_probas_combined,
                    labels=class_names # Use class_names if you have multiple columns for y_probas
                )})
                print("Logged interactive test PR curve to WandB.")
            else:
                print("Skipping test PR curve: Only one class present in test labels.")


            # --- Log Misclassified Predictions Table to WandB (from Best Model's Test Set Evaluation) ---
            if not test_misclassified_df.empty:
                # Prepare images for WandB if 'image_links' column exists
                if 'image_links' in test_misclassified_df.columns:
                    test_misclassified_df['wandb_images'] = test_misclassified_df['image_links'].apply(
                        lambda links: [wandb.Image(str(Path(link))) for link in links if link and Path(link).exists()] if isinstance(links, list) else []
                    )

                # Log the test set misclassified samples
                test_misclassified_table = wandb.Table(dataframe=test_misclassified_df)
                run.log({"test_misclassified_samples": test_misclassified_table})
                print(f"Logged {len(test_misclassified_df)} test set misclassified samples to WandB table.")
            else:
                print("No misclassified samples found on the test set for this run.")

            # --- Log Validation Misclassified Samples (from best epoch's validation, or last epoch if no improvement) ---
            # This logic was already there, but now ensure it refers to the best epoch's misclassified data
            if misclassified_data_for_logging: # This now holds data from the best_epoch's validation run
                val_misclassified_df_to_log = pd.DataFrame(misclassified_data_for_logging)
                if 'image_links' in val_misclassified_df_to_log.columns:
                    val_misclassified_df_to_log['wandb_images'] = val_misclassified_df_to_log['image_links'].apply(
                        lambda links: [wandb.Image(str(Path(link))) for link in links if link and Path(link).exists()] if isinstance(links, list) else []
                    )
                val_misclassified_table = wandb.Table(dataframe=val_misclassified_df_to_log)
                run.log({"validation_misclassified_samples_at_best_epoch": val_misclassified_table}) # Renamed table for clarity
                print(f"Logged {len(val_misclassified_df_to_log)} validation misclassified samples (from best epoch) to WandB table.")
            else:
                print("No misclassified samples found in the best validation epoch to log for this run.")

            # At the end of the 'with wandb.init' block, the run will automatically finish and sync.
            print(f"WandB run '{run.name}' finished and synced.")
def hyperparameter_optimization():
    # --- Authenticate WandB ---
    # Option 1: Use wandb login (recommended for interactive notebooks)
    # You will be prompted to paste your API key
    wandb.login()

    # Option 2: Set API key from environment variable (if you have it configured)
    # import os
    # os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY" # Replace with your actual API key or load from .env

    # --- Data Preparation (perform once for the entire sweep) ---
    print("Loading full dataset and performing initial 80/20 train/test split...")
    original_full_df = load_full_dataset(DATADIR)

    # Perform the 80/20 split once for the entire sweep
    train_val_df, final_test_df = train_test_split(
        original_full_df, test_size=0.2, stratify=original_full_df['is_advert'], random_state=42
    )
    print(f"Data split: Train+Val ({len(train_val_df)} samples), Final Test ({len(final_test_df)} samples)")

    # Split train_val_df further for HPO trials (e.g., 80/20 of this 80%, so 64%/16% of original)
    train_df_for_sweeps, val_df_for_sweeps = train_test_split(
        train_val_df, test_size=(0.2 / 0.8), stratify=train_val_df['is_advert'], random_state=42 # 20% of train_val_df for validation
    )
    print(f"HPO Data Split: Training ({len(train_df_for_sweeps)} samples), Validation ({len(val_df_for_sweeps)} samples)")

    # --- Define the Sweep Configuration ---
    sweep_config = {
        "name": "navermap-review-hpo-exp3-final",
        "method": "bayes", # Using Bayesian Optimization
        "metric": {"name": "val_fbeta_score", "goal": "maximize"}, # Maximize F-beta score
        "parameters": {
            # Text and Image Preprocessing Strategies (categorical choices)
            "text_preprocessing.strategy": {
                "values": ["option3_cross_attention_mean_pool", "option1_cross_attention_results"]},
            "image_preprocessing.strategy": {
                "values": ["option3_global_max_pool", "option2_variable_wise_mean_pool", "option4_variable_wise_max_pool"]},

            # Model Architecture parameters
            "model_architecture.no_intermodal_attention": {"values": [True, False]},
            "model_architecture.fusion_embed_dim": {"values": [128, 256, 384, 512, 768]},
            "model_architecture.dropout": {"distribution": "uniform", "min": 0.0, "max": 0.3},
            "model_architecture.classifier_hidden_dim_ratio": {"values": [0.5, 1.0, 2.0]}, # Ratio to fusion_embed_dim
            "model_architecture.attention_heads": {"values": [2, 4, 8]}, # Only applies if no_intermodal_attention is False
            "model_architecture.attention_layers": {"values": [1, 2, 3]}, # Only applies if no_intermodal_attention is False

            # Training parameters
            "training.batch_size": {"values": [8, 16, 32]},
            "training.num_epochs": {"value": 30}, # Fixed to 30 epochs as per updated plan
            "training.learning_rate": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-4}
        }
    }

    # --- Create the Sweep ---
    print("Creating WandB Sweep...")
    sweep_id = wandb.sweep(sweep=sweep_config, project=WANDB_PROJECT_NAME)
    print(f"Sweep ID: {sweep_id}")

    # --- Run the Sweep Agent ---
    # The lambda function wraps train_model_with_config to pass the pre-loaded dataframes
    # and allows wandb.agent to provide the 'config' automatically.
    print(f"Starting WandB agent for sweep ID: {sweep_id}")
    wandb.agent(
        sweep_id,
        function=lambda: train_model_with_config(
            config=None, # config will be automatically passed by wandb.agent
            train_df=train_df_for_sweeps,
            val_df=val_df_for_sweeps,
            test_df=final_test_df
        ),
        count=20 # Set a reasonable number of trials for the sweep. You can adjust this.
                # Given 4 combinations and Bayesian search, 20-50 trials is a good start.
    )

    print("\nWandB Sweep execution complete. Check your WandB dashboard for results.")

if __name__ == "__main__":
    # test_df= pd.read_parquet(DATADIR / "navermap_reviews_test.parquet")
    # main(test_df, Path(__file__).parent / "navermap_configs") # for training experiments
    hyperparameter_optimization()

#%%