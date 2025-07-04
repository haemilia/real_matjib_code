#%%
import torch
import duckdb
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from navermap_utils import NavermapReviewDataset, custom_collate_fn
from navermap_model import NaverMapModel
from tqdm import tqdm
import numpy as np
import logging

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration for Data and Model Paths ---
DATA_PATH = Path(r"G:\My Drive\Data\naver_search_results\labelled_navermap_reviews.parquet")
# DB_PATH is no longer used.
LOCAL_DB_PATH = Path(r"C:\Users\m\2025\project\code_only\real_matjib_code\reviews_local.db")
MODEL_SAVE_DIR = Path(r"C:\Users\m\2025\project\code_only\models")

QUERY_DATA_FOR_INFERENCE = """
    SELECT
        review_id,
        review_text,
        store_id,
        store_naver_name,
        category,
        image_links,
        video_thumbnail_links,
        store_reply,
        num_of_media,
        visit_count,
        author_total_reviews,
        author_total_images,
        reactions_fun,
        reactions_helpful,
        reactions_wannago,
        reactions_cool,
        review_year,
        rating,
        review_datetime,
        visit_keywords,
        purchase_item,
        keyword_tags_hangul
    FROM navermap_reviews AS n
    LEFT JOIN restaurants AS r
    ON n.store_id = r.naver_store_id
"""
#%%
# --- Path to the best model checkpoint ---
BEST_MODEL_CHECKPOINT_PATH = MODEL_SAVE_DIR / "model_sweep_mmfh4a69_best_fbeta.pth"

# --- Global variable to hold the loaded model ---
_loaded_inference_model = None
_model_device = None

def load_inference_model(checkpoint_path: Path, device: torch.device):
    """
    Loads the best model from a checkpoint for inference.

    Args:
        checkpoint_path (Path): The file path to the saved model checkpoint (.pth).
        device (torch.device): The device to load the model onto (e.g., 'cuda' or 'cpu').

    Returns:
        NaverMapModel: The loaded model in evaluation mode.
    """
    global _loaded_inference_model, _model_device

    if _loaded_inference_model is not None and _model_device == device:
        logger.info("Model already loaded. Returning existing instance.")
        return _loaded_inference_model

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    logger.info(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = checkpoint.get('config')
    if model_config is None:
        raise ValueError("Model configuration not found in the checkpoint. Cannot recreate the model architecture.")
    
    # Reconstruct a dummy dataset to get tabular_input_dim
    # Includes all columns that NavermapReviewDataset might expect.
    dummy_df_columns = ['review_id', 'review_text', 'store_id', 'store_naver_name',
                        'category', 'image_links', 'video_thumbnail_links', 'is_advert',
                        'visit_keywords', 'keyword_tags_hangul', 'num_of_media', 'visit_count',
                        'author_total_reviews', 'author_total_images', 'reactions_fun',
                        'reactions_helpful', 'reactions_wannago', 'reactions_cool',
                        'review_year', 'rating', 'store_name', 'store_reply', 'purchase_item',
                        'keyword_tags_code', 'review_datetime']
    
    dummy_row_data = {}
    for col in dummy_df_columns:
        if col in ['image_links', 'video_thumbnail_links', 'category', 'visit_keywords', 'keyword_tags_hangul', 'keyword_tags_code']:
            dummy_row_data[col] = [] # For VARCHAR[] / list type
        elif col == 'is_advert':
            dummy_row_data[col] = 0.0 # For float label type
        elif col in ['num_of_media', 'visit_count', 'author_total_reviews', 'author_total_images', 'review_year']:
            dummy_row_data[col] = 0 # For integer type
        elif col in ['reactions_fun', 'reactions_helpful', 'reactions_wannago', 'reactions_cool', 'rating']:
            dummy_row_data[col] = 0.0 # For double/float type
        elif col == 'review_datetime':
            dummy_row_data[col] = pd.Timestamp('2000-01-01') # For TIMESTAMP_NS type
        else: # Default value for other string types
            dummy_row_data[col] = 'dummy'
    
    dummy_df = pd.DataFrame([dummy_row_data])

    dummy_dataset = NavermapReviewDataset(dummy_df)
    tabular_input_dim = len(dummy_dataset.get_tabular_columns())
    logger.info(f"Determined tabular_input_dim: {tabular_input_dim}")

    model = NaverMapModel(model_config, tabular_input_dim=tabular_input_dim, device=device).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Model loaded and set to evaluation mode.")
    
    _loaded_inference_model = model
    _model_device = device
    return _loaded_inference_model

def classify_single_review(model: NaverMapModel, review_data: dict):
    """
    Classifies a single blog review using the loaded model.
    The model is loaded once and reused for subsequent calls.

    Args:
        model (NaverMapModel): The loaded model (in evaluation mode).
        review_data (dict): A dictionary containing data for a single review.
                            Expected keys (with robust defaults if missing/invalid):
                            'review_id': str (REQUIRED)
                            'review_text': str (e.g., '')
                            'store_id': str (e.g., 'N/A')
                            'store_naver_name': str (e.g., 'N/A')
                            'category': list of str (e.g., [])
                            'image_links': list of str (file paths, e.g., [])
                            'video_thumbnail_links': list of str (file paths, e.g., [])
                            # All other columns expected by NavermapReviewDataset

    Returns:
        dict: A dictionary containing 'review_id', 'predicted_label',
              'predicted_logit', and 'predicted_probability'.

    Raises:
        ValueError: If 'review_id' is missing or empty in the input review_data.
    """
    # Convert review_id to string and strip whitespace
    review_id = str(review_data.get('review_id', '')).strip()
    if not review_id: # Raise error if it's an empty string
        raise ValueError("Input 'review_data' must contain a non-empty string 'review_id'.")
    
    processed_review_id = review_id

    # Define expected columns and their default types/values for robustness
    expected_columns = {
        'review_text': '',
        'store_id': 'N/A',
        'store_naver_name': 'N/A',
        'category': [],
        'image_links': [],
        'video_thumbnail_links': [],
        'is_advert': -1, # Dummy label for inference, NavermapReviewDataset expects it
        'visit_keywords': [],
        'keyword_tags_hangul': [],
        'num_of_media': 0,
        'visit_count': 0,
        'author_total_reviews': 0,
        'author_total_images': 0,
        'reactions_fun': 0.0,
        'reactions_helpful': 0.0,
        'reactions_wannago': 0.0,
        'reactions_cool': 0.0,
        'review_year': 0,
        'rating': 0.0,
        'store_name': 'N/A',
        'store_reply': 'N/A',
        'purchase_item': 'N/A',
        'keyword_tags_code': [],
        'review_datetime': pd.Timestamp('2000-01-01')
    }

    processed_data = {'review_id': processed_review_id}
    for col, default_val in expected_columns.items():
        val = review_data.get(col)
        
        if val is None:
            processed_data[col] = default_val
        elif isinstance(default_val, list):
            if not isinstance(val, list):
                processed_data[col] = [str(val)] if val is not None else []
            else:
                processed_data[col] = [str(item) for item in val if item is not None]
        elif isinstance(default_val, str):
            processed_data[col] = str(val)
        elif isinstance(default_val, (int, float)):
            try:
                processed_data[col] = float(val) if isinstance(default_val, float) else int(val)
            except (ValueError, TypeError):
                processed_data[col] = default_val
        elif isinstance(default_val, pd.Timestamp):
            if isinstance(val, (str, pd.Timestamp)):
                try:
                    processed_data[col] = pd.to_datetime(val)
                except (ValueError, TypeError):
                    processed_data[col] = default_val
            else:
                processed_data[col] = default_val
        else:
            processed_data[col] = val
    
    single_review_df = pd.DataFrame([processed_data])

    device = next(model.parameters()).device # Get current device of the model

    inference_dataset = NavermapReviewDataset(single_review_df)
    inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn, num_workers=0)

    with torch.no_grad():
        for batch_data in inference_dataloader:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_data.items()}
            
            if 'labels' in inputs:
                inputs.pop('labels')

            outputs = model(inputs)
            
            # --- NEW LOGGING: Check raw model outputs ---
            logger.debug(f"Review ID: {processed_review_id}, Raw Model Output (logits): {outputs.item()}")
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                logger.error(f"NaN or Inf detected in model outputs for review_id: {processed_review_id}. Output: {outputs.item()}")
                # Consider returning None or raising a specific error if you want to stop processing on NaN
                # For now, we'll let it proceed to see if it propagates.

            logits = outputs.squeeze(1) 
            probabilities = torch.sigmoid(logits) 
            predictions = (probabilities > 0.5).float() 

            predicted_label = int(predictions.item())
            predicted_logit = logits.item()
            predicted_probability = probabilities.item()

            if np.isnan(predicted_label) or np.isnan(predicted_probability):
                logger.error(f"NaN detected in final prediction values for review_id: {processed_review_id}. Label: {predicted_label}, Prob: {predicted_probability}")

            return {
                'review_id': processed_review_id,
                'predicted_label': predicted_label,
                'predicted_logit': predicted_logit,
                'predicted_probability': predicted_probability
            }
    logger.error("Inference failed to produce a result for the given review data after dataloader iteration.")
    raise RuntimeError("Inference failed to produce a result for the given review data.")

# -- Database related functions -- ##################################################################################

def get_connection_to_db(DB_path):
    """
    Establishes a connection to a DuckDB database.
    
    Args:
        DB_path (Path): DuckDB database file path.
        
    Returns:
        duckdb.DuckDBPyConnection: The DuckDB connection object.
    """
    con = duckdb.connect(str(DB_path))
    logger.info(f"Connected to DuckDB at: {DB_path}")
    return con

def create_labelled_table(con: duckdb.DuckDBPyConnection):
    """
    Creates the 'labelled_navermap_reviews' table in the local DB with a schema
    derived from the model's input query, and adds the label columns.
    It will drop the table if it already exists to ensure a fresh schema.

    Args:
        con (duckdb.DuckDBPyConnection): The DuckDB connection object (connected to LOCAL_DB_PATH).

    Returns:
        bool: True if table is successfully created/prepared, False otherwise.
    """
    try:
        table_name = 'labelled_navermap_reviews'

        # 1. Drop the table if it already exists (as per user's request "not check if table exists for now")
        logger.info(f"Attempting to drop table '{table_name}' if it exists...")
        con.execute(f"DROP TABLE IF EXISTS {table_name};")
        logger.info(f"Table '{table_name}' dropped (if it existed).")

        # 2. Get the schema of the model's input query dynamically
        # Execute query with LIMIT 0 to get column descriptions
        initial_schema_query = f"{QUERY_DATA_FOR_INFERENCE} LIMIT 0;"
        cursor = con.execute(initial_schema_query)
        input_columns_info = cursor.description

        if not input_columns_info:
            logger.critical("Failed to retrieve schema from input query. Check query validity and database connectivity.")
            return False

        column_defs = []
        for col_name, type_code, *_ in input_columns_info:
            # Map Python type codes to DuckDB SQL types based on common types
            duckdb_type = 'VARCHAR' # Default to VARCHAR for safety
            if type_code is str:
                duckdb_type = 'VARCHAR'
            elif type_code is int:
                duckdb_type = 'BIGINT' # Use BIGINT for integers from DuckDB
            elif type_code is float:
                duckdb_type = 'DOUBLE' # Use DOUBLE for floats from DuckDB
            elif type_code is pd.Timestamp or type_code is np.datetime64: # For datetime objects
                duckdb_type = 'TIMESTAMP_NS'
            # Explicitly handle list types, assuming they are VARCHAR[]
            # Note: 'keyword_tags_code' is NOT in the new QUERY_DATA_FOR_INFERENCE,
            # but it is present in the dummy_df_columns and expected_columns in classify_single_review.
            # This might cause a mismatch if the model or dataset explicitly relies on it.
            if col_name in ['category', 'image_links', 'video_thumbnail_links', 'visit_keywords', 'keyword_tags_hangul', 'keyword_tags_code']:
                duckdb_type = 'VARCHAR[]'

            column_defs.append(f"{col_name} {duckdb_type}")

        # Add the label columns
        column_defs.append("is_advert_label INTEGER")
        column_defs.append("is_advert_prob FLOAT")

        # Create table query with primary key
        create_table_query = f"CREATE TABLE {table_name} ({', '.join(column_defs)}, PRIMARY KEY (review_id));"
        
        logger.info(f"Creating table '{table_name}' with query:\n{create_table_query}")
        con.execute(create_table_query)
        logger.info(f"Table '{table_name}' created successfully.")

        con.sql("PRAGMA show_tables_expanded;").show()
        con.sql(f"DESCRIBE {table_name};").show()
        return True
    except Exception as e:
        logger.error(f"An error occurred during table creation/preparation: {e}", exc_info=True)
        return False

def update_row(con: duckdb.DuckDBPyConnection, original_row_dict: dict, new_prediction_cols: dict):
    """
    Inserts or replaces a row in the 'labelled_navermap_reviews' table with new prediction data.
    Uses INSERT OR REPLACE INTO for UPSERT functionality.
    
    Args:
        con (duckdb.DuckDBPyConnection): The DuckDB connection object.
        original_row_dict (dict): The dictionary of the original row data (including all input columns).
        new_prediction_cols (dict): A dictionary containing the new prediction columns
                                    ('is_advert_label', 'is_advert_prob').
    """
    combined_row = original_row_dict.copy()
    combined_row.update(new_prediction_cols)

    # Ensure list columns are handled correctly for DuckDB (list of strings)
    for key in ['category', 'image_links', 'video_thumbnail_links', 'visit_keywords', 'keyword_tags_hangul', 'keyword_tags_code']:
        if key in combined_row:
            val = combined_row[key]
            if isinstance(val, list):
                # Ensure all elements in the list are strings and not None
                combined_row[key] = [str(item) for item in val if item is not None]
            elif val is None:
                combined_row[key] = [] # Convert None to empty list for DuckDB VARCHAR[]
            else:
                # If it's a single value but column expects a list, wrap it
                combined_row[key] = [str(val)]
    
    # Ensure numeric columns are not None, default to 0 if they are
    numeric_cols = ['num_of_media', 'visit_count', 'author_total_reviews', 'author_total_images', 
                    'reactions_fun', 'reactions_helpful', 'reactions_wannago', 'reactions_cool', 
                    'review_year', 'rating']
    for key in numeric_cols:
        if key in combined_row and combined_row[key] is None:
            combined_row[key] = 0
    
    # Handle review_datetime conversion to ISO format string for TIMESTAMP_NS
    if 'review_datetime' in combined_row and combined_row['review_datetime'] is not None:
        if isinstance(combined_row['review_datetime'], pd.Timestamp):
            combined_row['review_datetime'] = combined_row['review_datetime'].isoformat(timespec='microseconds')
        else:
            try:
                combined_row['review_datetime'] = pd.to_datetime(combined_row['review_datetime']).isoformat(timespec='microseconds')
            except (ValueError, TypeError):
                combined_row['review_datetime'] = None # Set to None if conversion fails
    else:
        combined_row['review_datetime'] = None # Ensure it's None if originally None

    columns = list(combined_row.keys())
    placeholders = ", ".join(["?" for _ in columns])
    column_names_str = ", ".join(columns)
    values = tuple(combined_row.values())

    insert_query = f"INSERT OR REPLACE INTO labelled_navermap_reviews ({column_names_str}) VALUES ({placeholders});"

    try:
        # logger.debug(f"Attempting to insert/update review_id '{original_row_dict.get('review_id', 'N/A')}' with values: {combined_row}")
        con.execute(insert_query, values)
    except Exception as e:
        logger.error(f"Error inserting/updating row for review_id '{original_row_dict.get('review_id', 'N/A')}': {e}", exc_info=True)

def save_backup(con: duckdb.DuckDBPyConnection, parquet_file_path: Path):
    """
    Saves the 'labelled_navermap_reviews' table from DuckDB to a Parquet file.
    
    Args:
        con (duckdb.DuckDBPyConnection): The DuckDB connection object.
        parquet_file_path (Path): The path to save the Parquet file.
    """
    logger.info(f"Saving 'labelled_navermap_reviews' to '{parquet_file_path}'...")
    try:
        con.execute(f"COPY labelled_navermap_reviews TO '{str(parquet_file_path)}' (FORMAT PARQUET, COMPRESSION SNAPPY, OVERWRITE TRUE);")
        logger.info(f"Successfully saved table to '{parquet_file_path}'")
    except duckdb.Error as e:
        logger.error(f"Error saving to Parquet: {e}", exc_info=True)


def main():
    # --- Initialize DuckDB connection for local labelled data ---
    duckdb_conn_output = get_connection_to_db(LOCAL_DB_PATH)
    
    # Ensure the labelled table exists and has the correct schema
    # This function now verifies input columns and adds output columns
    if not create_labelled_table(duckdb_conn_output):
        logger.critical("Failed to prepare the labelled table in local DB. Exiting.")
        duckdb_conn_output.close()
        return

    # --- Query data from navermap_reviews and restaurants tables ---
    logger.info(f"Querying data from navermap_reviews and restaurants tables in local DB: {LOCAL_DB_PATH}")
    
    logger.info(f"Executing query on local DB:\n{QUERY_DATA_FOR_INFERENCE}")

    # Get total rows from the input query for tqdm progress bar
    # Use the global QUERY_DATA_FOR_INFERENCE as a subquery for count
    total_rows_query_count = f"SELECT COUNT(*) FROM ({QUERY_DATA_FOR_INFERENCE}) AS subquery;"
    total_rows = duckdb_conn_output.execute(total_rows_query_count).fetchone()[0]
    logger.info(f"Found {total_rows} reviews to be labelled from source tables.")

    # --- Fetch all data into a pandas DataFrame first for robustness ---
    logger.info("Fetching all data into pandas DataFrame for processing...")
    try:
        data_df = duckdb_conn_output.execute(QUERY_DATA_FOR_INFERENCE).fetchdf()
        logger.info(f"Successfully fetched {len(data_df)} rows into DataFrame.")
    except Exception as e:
        logger.critical(f"Error fetching data into DataFrame: {e}", exc_info=True)
        duckdb_conn_output.close()
        return

    # Get column names from the DataFrame
    column_names = data_df.columns.tolist()
    logger.debug(f"Columns fetched from DataFrame: {column_names}")

    # --- Set up device for inference model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device for inference: {device}")

    # --- Load model (once, outside the loop) ---
    try:
        inference_model = load_inference_model(BEST_MODEL_CHECKPOINT_PATH, device)
    except (FileNotFoundError, ValueError) as e:
        logger.critical(f"Model loading error: {e}. Cannot start inference. Exiting.")
        duckdb_conn_output.close()
        return
    except Exception as e:
        logger.critical(f"Unexpected model loading error: {e}. Cannot start inference. Exiting.", exc_info=True)
        duckdb_conn_output.close()
        return

    # --- Start classification and incremental saving ---
    logger.info("\nStarting classification and incremental saving...")
    
    # Loop through all rows from the DataFrame
    for i, row_series in enumerate(tqdm(data_df.iterrows(), total=len(data_df), desc="Classifying and Saving")):
        
        # row_series is a pandas Series, convert to dict for consistent processing
        row_dict = row_series[1].to_dict() # row_series[0] is index, row_series[1] is the Series data
        
        # Log the review_id being processed for every 1000 samples and the first 5
        if i < 5 or i % 1000 == 0:
            logger.info(f"Processing sample {i+1}/{total_rows}. Fetched Review ID: {row_dict.get('review_id', 'N/A')} (Type: {type(row_dict.get('review_id'))})")
            logger.debug(f"Full row_dict for review_id {row_dict.get('review_id', 'N/A')}: {row_dict}")

        # Data type handling for NavermapReviewDataset input
        # Ensure these conversions are robust to None/NaN values from the database
        for key in ['category', 'image_links', 'video_thumbnail_links', 'visit_keywords', 'keyword_tags_hangul', 'keyword_tags_code']:
            if key in row_dict and row_dict[key] is not None:
                if isinstance(row_dict[key], list):
                    # Ensure all elements are strings
                    row_dict[key] = [str(item) for item in row_dict[key] if item is not None]
                else:
                    # If it's a single value, wrap it in a list
                    row_dict[key] = [str(row_dict[key])]
            elif key in row_dict and row_dict[key] is None:
                row_dict[key] = [] # Ensure it's an empty list if None
        
        numeric_cols = ['num_of_media', 'visit_count', 'author_total_reviews', 'author_total_images', 
                        'reactions_fun', 'reactions_helpful', 'reactions_wannago', 'reactions_cool', 
                        'review_year', 'rating']
        for key in numeric_cols:
            if key in row_dict and row_dict[key] is None:
                row_dict[key] = 0 # Default numeric to 0 if None
            elif key in row_dict and pd.isna(row_dict[key]): # Handle pandas NaN for numeric columns
                row_dict[key] = 0

        # Handle review_datetime conversion to pandas Timestamp
        if 'review_datetime' in row_dict and row_dict['review_datetime'] is not None:
            if not isinstance(row_dict['review_datetime'], pd.Timestamp):
                try:
                    row_dict['review_datetime'] = pd.to_datetime(row_dict['review_datetime'])
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert review_datetime '{row_dict['review_datetime']}' to Timestamp for review_id {row_dict.get('review_id', 'N/A')}. Setting to default.")
                    row_dict['review_datetime'] = pd.Timestamp('2000-01-01')
        else:
            row_dict['review_datetime'] = pd.Timestamp('2000-01-01') # Default if None

        try:
            # Pass the already loaded model object to classify_single_review function
            label_result = classify_single_review(
                model=inference_model, # Pass model object directly
                review_data=row_dict
            )

            if label_result:
                new_prediction_cols = {
                    'is_advert_label': label_result['predicted_label'],
                    'is_advert_prob': label_result['predicted_probability']
                }
                if i < 5 or i % 1000 == 0:
                    # Log the prediction results before updating the row
                    logger.info(f"Review ID: {row_dict.get('review_id', 'N/A')}, Predicted: {new_prediction_cols['is_advert_label']}, Prob: {new_prediction_cols['is_advert_prob']:.4f}")
                # Update the row in the database with the new predictions
                update_row(duckdb_conn_output, original_row_dict=row_dict, new_prediction_cols=new_prediction_cols)
            else:
                logger.warning(f"Classification returned no result for review_id: {row_dict.get('review_id', 'N/A')}. Skipping save.")
        except Exception as e:
            logger.error(f"An error occurred processing review_id {row_dict.get('review_id', 'N/A')}: {e}", exc_info=True)

    # --- Save final backup to Parquet ---
    save_backup(duckdb_conn_output, DATA_PATH)

    # --- Close DuckDB connection ---
    duckdb_conn_output.close()
    logger.info("DuckDB connection closed. Inference process finished.")

if __name__ == "__main__":
    main()

#%%
