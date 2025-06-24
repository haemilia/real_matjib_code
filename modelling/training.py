#%%
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import yaml
import pickle
from navermap_utils import NavermapReviewDataset, custom_collate_fn
from navermap_model import NaverMapModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split # type:ignore
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score # type:ignore
import wandb # For experiment tracking
from tqdm.auto import tqdm # For nice progress bars

# --- Configuration for WandB ---
WANDB_PROJECT_NAME = "navermap-review-classification"
WANDB_ENTITY = "haemilia-"

# --- configuration ---
DATADIR = Path("/content/drive/MyDrive/CV_training/real_matjib_colab")

def re_root_path(img_paths:dict, root_dir:Path):
    new_img_paths:Dict[str, Dict] = {}
    for review_id, img_dict in img_paths.items():
        new_img_paths[review_id] = {}
        for img_type, img_path_list in img_dict.items():
            new_img_paths[review_id][img_type] = list(map(lambda x: root_dir / x, img_path_list))
    return new_img_paths

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
    rerooted = re_root_path(img_paths, Path("/content/navermap_reviews_labelled_only_images"))
    img_df = pd.DataFrame(rerooted).T.reset_index(names="review_id")
    dropped_df = with_category.drop(columns=["image_links", "video_thumbnail_links"])
    result = pd.merge(dropped_df, img_df, on="review_id")

    # 1st train test split. The train data from here will go on to be split into train and val.
    train_df, test_df = train_test_split(result, test_size=0.5, random_state=68, stratify=result["is_advert"])
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

# Somewhere to save the resulting model
MODEL_SAVE_DIR = DATADIR / "saved_models"
MODEL_SAVE_DIR.mkdir(exist_ok=True)

def main(path_to_configs=DATADIR / "navermap_configs"):
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
        with wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, config=config, name=run_name):
            if wandb.run is not None:
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

            num_epochs = config['training'].get('num_epochs', 5) # Number of training epochs

            misclassified_data_for_logging = [] # To store misclassified samples from the last epoch

            # --- Initialize best F1 score for this run ---
            best_val_f1 = -1.0
            best_epoch = -1
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
                wandb.log({
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
                    wandb.log({"best_val_f1_achieved": best_val_f1, "best_epoch": best_epoch, "epoch": epoch + 1})


            print("\n--- Training Complete ---")
            print(f"Best Validation F1 Score for this run: {best_val_f1:.4f} at Epoch {best_epoch}")


            # --- Create and Log Misclassified Predictions Table to WandB ---
            if misclassified_data_for_logging:
                misclassified_df = pd.DataFrame(misclassified_data_for_logging)

                # Handle image paths for WandB visualization if they are local paths
                # Assuming 'image_links' in the collected data are lists of local file paths (strings or Path objects)
                if 'image_links' in misclassified_df.columns:
                    # Create a new column with WandB Image objects
                    # Need to handle cases where 'image_links' might be empty lists or None
                    misclassified_df['wandb_images'] = misclassified_df['image_links'].apply(
                        lambda links: [wandb.Image(str(Path(link))) for link in links if link and Path(link).exists()] if isinstance(links, list) else []
                    )
                    # You might want to drop the original 'image_links' column if 'wandb_images' is preferred for display
                    # misclassified_df = misclassified_df.drop(columns=['image_links'])

                misclassified_table = wandb.Table(dataframe=misclassified_df)
                wandb.log({"validation_misclassified_samples": misclassified_table})
                print(f"Logged {len(misclassified_df)} misclassified samples to WandB table.")
            else:
                print("No misclassified samples found in the last epoch to log for this run.")