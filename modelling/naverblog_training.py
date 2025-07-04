import pandas as pd
import matplotlib.pyplot as plt
import duckdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, accuracy_score,
)
import numpy as np
import wandb
from pathlib import Path # Importing pathlib
import warnings
from naverblog_model import NaverblogReviewDataset, create_blog_collate_fn, NaverBlogModel
import os
from tqdm.auto import tqdm
import random

# Suppress UndefinedMetricWarning from sklearn if a class has no samples
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")


def get_training_data():
    # Don't worry about here yet
    # Returns a pd.DataFrame of the training set (train + val), and test set (test)
    pass
def get_image_paths_dict():
    # To be implemented later
    # Currently working on creating one, due to it taking so long to download all the photos
    pass

def train(model, dataloader, optimizer, loss_fn, device):
    """Trains NaverBlogModel.
    Args:
        model: NaverBlogModel instance
        dataloader: PyTorch DataLoader instance for the training set
        optimizer: PyTorch optimizer
        loss_fn: loss function (BCEWithLogitsLoss)
        device: computing device
    """
    # Set model to training mode
    model.train()
    total_loss = 0
    num_batches = 0
    # Iterate through batches
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        # Move input data to device
        labels = batch.pop('label').to(device)
        processed_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # Zero gradients
        optimizer.zero_grad()
        # Model forward pass
        logits = model(**processed_batch).squeeze(1)
        # Calculate Loss
        loss = loss_fn(logits, labels.float())
        # Backpropagation
        loss.backward()
        # Update model weights
        optimizer.step()  
        # Accumulate batch loss
        total_loss += loss.item()
        num_batches += 1
    # return training loss
    avg_loss = total_loss / num_batches
    return avg_loss

def validate(model, dataloader, loss_fn, device):
    """Evaluates model performance on validation set
    Args:
        model: NaverBlogModel instance
        dataloader: PyTorch DataLoader instance for the validation set
        optimizer: PyTorch optimizer
        loss_fn: loss function (BCEWithLogitsLoss)
        device: computing device
    """
    # Set model to evaluation mode 
    model.eval()
    total_loss = 0
    all_labels = []
    all_logits = []
    # torch.no_grad()
    with torch.no_grad():
        # Iterate through batches
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            labels = batch.pop('label').to(device)
            processed_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            logits = model(**processed_batch).squeeze(1)
            loss = loss_fn(logits, labels.float())

            total_loss += loss.item()
            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())
    # Calculate Average validation loss
    avg_loss = total_loss / len(dataloader)

    all_labels = torch.cat(all_labels).numpy()
    all_logits = torch.cat(all_logits).numpy()

    # Convert logits to probabilities and predictions
    all_probs = torch.sigmoid(torch.from_numpy(all_logits)).numpy() # Apply sigmoid to get probabilities
    all_preds = (all_probs >= 0.5).astype(int) # Binary predictions

    # Calculate metrics
    val_f1 = f1_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, zero_division=0) # zero_division=0 to handle cases with no true/predicted samples
    val_recall = recall_score(all_labels, all_preds, zero_division=0)
    val_accuracy = accuracy_score(all_labels, all_preds) # Added accuracy as good to know
    
    # ROC AUC and PR AUC require probabilities
    val_roc_auc = roc_auc_score(all_labels, all_probs)
    val_pr_auc = average_precision_score(all_labels, all_probs) # average_precision_score is PR AUC

    metrics = {
        "val_loss": avg_loss,
        "val_f1": val_f1,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_accuracy": val_accuracy,
        "val_roc_auc": val_roc_auc,
        "val_pr_auc": val_pr_auc,
    }

    return metrics

def test_model(model_path, config, test_dataloader, device):
    """
    Comprehensive test of performance on model
    """
    print(f"\n--- Starting Test Phase for {model_path.name} ---")
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    # Extract model state dict, tabular_input_dim, and original config
    model_state_dict = checkpoint['model_state_dict']
    loaded_tabular_input_dim = checkpoint['tabular_input_dim']
    loaded_config = checkpoint.get('config', config)
    # Initialize model using the loaded tabular_input_dim and config
    model = NaverBlogModel(loaded_config, device, loaded_tabular_input_dim).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    print(f"Model loaded with tabular_input_dim: {loaded_tabular_input_dim}")

    all_labels = []
    all_logits = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
            labels = batch.pop('label').to(device)
            processed_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            logits = model(**processed_batch).squeeze(1)
            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_logits = torch.cat(all_logits).numpy()
    all_probs = torch.sigmoid(torch.from_numpy(all_logits)).numpy()
    all_preds = (all_probs >= 0.5).astype(int)
    # Calculate metrics
    test_loss_fn = nn.BCEWithLogitsLoss()
    test_loss = test_loss_fn(torch.from_numpy(all_logits), torch.from_numpy(all_labels).float()).item()
    test_f1 = f1_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, zero_division=0)
    test_recall = recall_score(all_labels, all_preds, zero_division=0)
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_metrics = {
        "test_loss": test_loss,
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_accuracy": test_accuracy,
    }
# Only calculate AUCs if there's more than one class present
    if len(np.unique(all_labels)) > 1:
        test_roc_auc = roc_auc_score(all_labels, all_probs)
        test_pr_auc = average_precision_score(all_labels, all_probs)
        test_metrics["test_roc_auc"] = test_roc_auc
        test_metrics["test_pr_auc"] = test_pr_auc
    else:
        warnings.warn("Only one class present in test labels. ROC AUC and PR AUC cannot be calculated and will be set to NaN.")
        test_metrics["test_roc_auc"] = float('nan')
        test_metrics["test_pr_auc"] = float('nan')

    print(f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # --- Log Metrics to WandB ---
    wandb.log(test_metrics)

    # --- Plotting and Logging to WandB using wandb.plot ---
    class_names = ['Not Advert', 'Is Advert']
    
    # Confusion Matrix
    wandb.log({"test/confusion_matrix": wandb.plot.confusion_matrix(
        y_true=all_labels,
        preds=all_preds,
        class_names=class_names
    )})

    # ROC Curve
    if len(np.unique(all_labels)) > 1: # Ensure multiple classes for ROC
        wandb.log({"test/roc_curve": wandb.plot.roc_curve(
            all_labels,
            all_probs,
            labels=class_names 
        )})
    else:
        print("Skipping test/roc_curve: only one class found in test labels.")

    # Precision-Recall Curve
    if len(np.unique(all_labels)) > 1: # Ensure multiple classes for PR
        wandb.log({"test/pr_curve": wandb.plot.pr_curve(
            all_labels,
            all_probs,
            labels=class_names 
        )})
    else:
        print("Skipping test/pr_curve: only one class found in test labels.")

    print("Test phase complete. Metrics and plots logged to WandB.")

def sweep_train_function(config=None):
    """
    Main training function for a single Weights & Biases sweep run.
    Orchestrates data loading, model initialization, training, validation,
    logging, early stopping, and best model saving. Uses stratified train_test_split.
    """
    wandb.init(config=config)
    config = wandb.config 
    
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading and Preparation
    train_val_df, test_df = get_training_data()

    # Stratified Train-Validation Split using train_test_split
    # Returns DataFrames if you pass the DataFrame as the first argument
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=0.2, 
        stratify=train_val_df['is_advert'], # Crucial for stratified split
        random_state=seed
    )
    
    # Reset indices to ensure consistency with Dataset __getitem__ if it relies on sequential index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    image_paths_dict = get_image_paths_dict()

    train_dataset = NaverblogReviewDataset(config, train_df, image_paths_dict)
    val_dataset = NaverblogReviewDataset(config, val_df, image_paths_dict)
    
    tabular_input_dim = len(train_dataset.get_tabular_columns())
    print(f"Tabular input dimension detected: {tabular_input_dim}")

    collate_fn = create_blog_collate_fn(config)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # Model, Optimizer, and Loss Function Initialization
    model = NaverBlogModel(config, device, tabular_input_dim).to(device)
    print(f"Model initialized. Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    # Training and Validation Loop (with Early Stopping & Model Saving)
    best_val_f1 = -1.0 # We want to maximize F1
    patience_counter = 0
    model_save_dir = Path("./models")
    model_save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_save_dir / f"best_model_run_{wandb.run.name}.pth"

    print(f"Starting training for run: {wandb.run.name}")
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        
        train_loss = train(model, train_dataloader, optimizer, loss_fn, device)
        val_metrics = validate(model, val_dataloader, loss_fn, device)

        wandb.log({
            "train_loss": train_loss,
            **val_metrics
        }, step=epoch)

        current_val_f1 = val_metrics["val_f1"]
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['val_loss']:.4f}, Val F1: {current_val_f1:.4f}")

        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            torch.save({"model_state_dict":model.state_dict(),
                        "tabular_input_dim": tabular_input_dim,
                        "config": config}, best_model_path)
            print(f"New best model saved to {best_model_path} with F1: {best_val_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Validation F1 did not improve. Patience: {patience_counter}/{config.early_stopping.patience}")

        if patience_counter >= config.early_stopping.patience:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
            break
    
    print(f"Training finished for run: {wandb.run.name}. Best Val F1: {best_val_f1:.4f}")
    wandb.finish()

def main():
    # Implement the sweep here
    pass

if __name__ == "__main__":
    main()

