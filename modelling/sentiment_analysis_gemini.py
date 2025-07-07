#%%
import os
from dotenv import load_dotenv
from google.genai import Client, types
from pydantic import BaseModel
from enum import Enum
import duckdb
import pandas as pd
import csv
import io
from pathlib import Path
from typing import List, Dict, Any, Tuple
import datetime
import time
import uuid
from tqdm import tqdm
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

NUM_RESTAURANTS = 10
DATA_PATH = Path(r"H:\My Drive\reviews.db") #Input data parquet # If DB mode, insert path to DB
SENTIMENT_SAVE_PATH = Path(r"G:\My Drive\Data\naver_search_results") / "naver_sentiment_classified.parquet"
RESTAURANT_SAVE_PATH = Path(r"G:\My Drive\Data\naver_search_results") / "naver_sentiment_classified_restaurants.csv"
UNPARSEABLE_JSON_DIR = Path(r"G:\My Drive\Data\naver_search_results")
GEMINI_MODEL_NAME = "gemini-1.5-flash-8b"
TOKEN_LIMIT_PER_REQUEST = 250000
REQUEST_LIMIT_PER_DAY = 50

class SentimentEnum(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ERROR = "error"

class ConfidenceEnum(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ERROR = "error"

class SentimentClassification(BaseModel):
    """Represents the sentiment classification of a given text."""
    sentiment: SentimentEnum
    confidence: ConfidenceEnum



# def robust_json_parse(json_string: str, error_save_dir: Path | None = None,
#                       error_filename: str | None = None) -> List[Dict[str, Any]]:
#     """
#     Attempts to parse a JSON string, robustly handling common errors like truncation
#     or malformation by trying to salvage valid JSON segments.

#     Args:
#         json_string: The raw JSON string to parse.
#         error_save_dir: Optional Path object to a directory where problematic JSON
#                         strings will be saved if parsing/salvage fails.
#         error_filename: Optional string for the filename to use when saving errors.
#                         If provided, error_save_dir must also be provided.

#     Returns:
#         A list of dictionaries representing the parsed JSON objects.
#         Returns an empty list if parsing or salvage fails completely.
#     """
#     if not json_string or not json_string.strip():
#         print("Warning: Empty or whitespace-only JSON string provided for parsing.")
#         return []

#     processed_json_string = json_string.strip()

#     # Helper for saving errors
#     def _save_error_file(content_to_save: str, specific_reason: str):
#         if error_save_dir and error_filename:
#             save_unparseable_json_to_file(error_save_dir, error_filename, content_to_save)
#             print(f"Saved unparseable JSON to {error_save_dir / error_filename} due to: {specific_reason}")
#         else:
#             print(f"Error: Cannot save unparseable JSON. error_save_dir or error_filename not provided for reason: {specific_reason}")


#     try:
#         # Attempt direct parsing first
#         parsed_data = json.loads(processed_json_string)
#         if isinstance(parsed_data, list):
#             return parsed_data
#         else:
#             print(f"Warning: JSON content is not a list (got {type(parsed_data)}). Attempting salvage.")
#             if isinstance(parsed_data, dict):
#                 return [parsed_data]
#             _save_error_file(json_string, "initial_unexpected_type")
#             return []

#     except json.JSONDecodeError as e:
#         print(f"Direct JSON parsing failed: {e}. Attempting aggressive salvage.")
        
#         salvaged_json = processed_json_string

#         last_brace_idx = salvaged_json.rfind("}")
#         if last_brace_idx != -1:
#             salvaged_json = salvaged_json[:last_brace_idx + 1] + "]"
#         else:
#             if salvaged_json.startswith("[") and not salvaged_json.endswith("]"):
#                 if salvaged_json.endswith(","):
#                     salvaged_json = salvaged_json[:-1]
#                 salvaged_json += "]"

#         if not salvaged_json.startswith("[") and salvaged_json.startswith("{"):
#             salvaged_json = "[" + salvaged_json + "]"
#         elif not salvaged_json.startswith("["):
#             _save_error_file(json_string, "start_malformed")
#             return []

#         if not salvaged_json.endswith("]"):
#             salvaged_json += "]"

#         salvaged_json = salvaged_json.replace(",,", ",")
#         salvaged_json = salvaged_json.replace("},]", "}]")

#         try:
#             print(f"Attempting to parse salvaged JSON (first 200 chars): {salvaged_json[:200]}...")
#             parsed_data = json.loads(salvaged_json)
#             if isinstance(parsed_data, list):
#                 return parsed_data
#             else:
#                 if isinstance(parsed_data, dict):
#                     print("Salvaged JSON resulted in a single object. Wrapping in a list.")
#                     return [parsed_data]
#                 print(f"Salvaged JSON did not result in a list or single object (got {type(parsed_data)}).")
#                 _save_error_file(json_string, "salvage_unexpected_type")
#                 return []
#         except Exception as e_inner:
#             print(f"Salvage failed completely after aggressive attempts: {e_inner}")
#             _save_error_file(json_string, "salvage_failed")
#             return []

def save_unparseable_json_to_file(file_dir: Path, file_name: str, file_content: str):
    """
    Saves the given file_content to a text file.

    Args:
        file_dir: The directory where the file should be saved.
        file_name: The name of the file (e.g., "error_output.txt").
        file_content: The content to write to the file.
    """
    if not file_dir.is_dir():
        file_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {file_dir}")

    file_path = file_dir / file_name

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file_content)
    except Exception as e:
        print(f"Error saving unparseable JSON to file {file_path}: {e}")
# def classify_sentiment_batched(texts_to_classify: List[str]) -> List[SentimentClassification]:
#     """
#     Classifies sentiment for texts by batching them into sizes of 500 for API requests.
#     """
#     client = Client(api_key=GEMINI_API_KEY)
#     all_classified_results = []
    
#     # Define the batch size
#     BATCH_SIZE = 200 

#     system_instruction_text = "You are a highly accurate sentiment analysis model that classifies text as positive, neutral, or negative, providing a confidence level (high, medium, low)."
    
#     user_prompt_header = (
#         "Classify the sentiment of the following texts. "
#         "For each text, provide the sentiment ('positive', 'neutral', 'negative') "
#         "and confidence ('high', 'medium', 'low'). "
#         "Respond with a JSON array where each element is an object like "
#         '{"sentiment": "value", "confidence": "value"}. '
#         "Maintain the order of the input texts in your output.\n\n"
#         "Texts to classify:\n"
#     )
    
#     # Iterate through batches
#     for i in tqdm(range(0, len(texts_to_classify), BATCH_SIZE), desc="Processing review batches"):
#         current_batch_texts = texts_to_classify[i:i + BATCH_SIZE]
        
#         if not current_batch_texts:
#             continue

#         # Generate a unique filename for this batch's potential error log
#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         unique_id = uuid.uuid4().hex[:8]
#         # Filename includes batch start index for better organization
#         error_filename = f"batch_{i}_model_response_error_{timestamp}_{unique_id}.txt"
        
#         raw_json_response_array = "" # Initialize outside try for scope

#         try:
#             full_user_input = user_prompt_header + "\n".join([f"- {text}" for text in current_batch_texts])

#             response = client.models.generate_content(
#                 model=GEMINI_MODEL_NAME, 
#                 contents=[full_user_input],
#                 config=types.GenerateContentConfig( 
#                     response_mime_type="application/json",
#                     temperature=0.01, 
#                     system_instruction=system_instruction_text 
#                 )
#             )
            
#             raw_json_response_array = response.text
#             print(f"Batch {i//BATCH_SIZE}: Raw JSON Response Array (start): {raw_json_response_array[:500]}...")

#             parsed_list = robust_json_parse(raw_json_response_array, error_save_dir=UNPARSEABLE_JSON_DIR, error_filename=error_filename)
            
#             # Check if model returned fewer results than inputs for this specific batch
#             if len(parsed_list) < len(current_batch_texts):
#                 print(f"Batch {i//BATCH_SIZE}: Warning! Model returned fewer results ({len(parsed_list)}) than inputs ({len(current_batch_texts)}). Saving raw response for manual review.")
#                 save_unparseable_json_to_file(UNPARSEABLE_JSON_DIR, error_filename, raw_json_response_array)

#             # Process results for the current batch
#             batch_classified_results = []
#             for j in range(len(current_batch_texts)):
#                 if j < len(parsed_list):
#                     parsed_element = parsed_list[j]
#                     try:
#                         batch_classified_results.append(SentimentClassification(**parsed_element))
#                     except Exception as e:
#                         print(f"Batch {i//BATCH_SIZE}, Item {j}: Warning: Failed to parse item from parsed_list: {parsed_element}. Error: {e}")
#                         batch_classified_results.append(SentimentClassification(sentiment=SentimentEnum.ERROR, confidence=ConfidenceEnum.ERROR)) 
#                 else:
#                     # This branch is hit if parsed_list is shorter than current_batch_texts
#                     print(f"Batch {i//BATCH_SIZE}, Item {j}: No corresponding classification. Padding with ERROR.")
#                     batch_classified_results.append(SentimentClassification(sentiment=SentimentEnum.ERROR, confidence=ConfidenceEnum.ERROR)) 
            
#             all_classified_results.extend(batch_classified_results)
        
#         except Exception as e:
#             # This catches general API errors (network issues, rate limits, etc.) for the current batch
#             print(f"Batch {i//BATCH_SIZE}: An API call or unexpected error occurred: {e}")
#             if raw_json_response_array: 
#                 # Use a different filename prefix for API exceptions vs. parsing/truncation
#                 error_filename_api_exception = f"batch_{i}_api_exception_raw_output_{timestamp}_{unique_id}.txt"
#                 save_unparseable_json_to_file(UNPARSEABLE_JSON_DIR, error_filename_api_exception, raw_json_response_array)
            
#             # Pad the rest of the current batch with errors if an exception occurred
#             all_classified_results.extend([SentimentClassification(sentiment=SentimentEnum.ERROR, confidence=ConfidenceEnum.ERROR) for _ in current_batch_texts])
            
#     return all_classified_results
  
def robust_csv_parse(csv_string: str, expected_columns: List[str], error_save_dir: Path | None = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Attempts to parse a CSV string robustly, handling common errors like
    truncation, malformed rows, or incorrect column counts.

    Args:
        csv_string (str): The CSV content as a string.
        expected_columns (List[str]): A list of expected column names in order.
        error_save_dir (Path | None): Directory to save unparseable CSV content.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            A tuple containing:
            - A list of successfully parsed rows (dictionaries).
            - A list of dictionaries representing error rows, including 'raw_line' and 'error_reason'.
    """
    parsed_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []
    
    if not csv_string or not csv_string.strip():
        print("Warning: Empty or whitespace-only CSV string provided for parsing.")
        return [], [{'raw_line': '', 'error_reason': 'Empty or whitespace-only input'}]

    # Normalize line endings to Unix-style for consistent parsing
    processed_csv_string = csv_string.strip().replace('\r\n', '\n')

    # Use io.StringIO to treat the string as a file
    csv_file = io.StringIO(processed_csv_string)
    
    # csv.reader handles quoting correctly
    reader = csv.reader(csv_file)
    
    header = None
    first_row_skipped = False

    for i, row in enumerate(reader):
        raw_line = ','.join(row) # Reconstruct raw line for error reporting

        if not row: # Skip empty lines
            continue

        if not first_row_skipped:
            # Assume the first non-empty row is the header.
            # We can try to validate it or just use it.
            # For this scenario, we assume the model *should* output "sentiment,confidence" as header.
            if row == expected_columns:
                header = row
                first_row_skipped = True
                continue
            else:
                # If the first line doesn't match expected headers,
                # it's either missing or malformed header, or the model
                # started directly with data without header.
                # We'll try to parse it as data if it has the right number of columns.
                print(f"Warning: Expected header {expected_columns}, but got {row}. Attempting to parse as data.")
                # We don't set header, and proceed to parse this row as data if it has correct length.
                header = expected_columns # Force header for Dict-like output
                first_row_skipped = True # Prevent trying to find header again
                # Do NOT continue; process this row as data
        
        # If header is still None after checking the first row, something is very wrong
        if header is None:
            error_rows.append({'raw_line': raw_line, 'error_reason': 'No valid header found or implied, cannot parse data rows.'})
            continue

        if len(row) != len(expected_columns):
            error_rows.append({
                'raw_line': raw_line,
                'error_reason': f"Incorrect number of columns. Expected {len(expected_columns)}, got {len(row)}."
            })
            continue

        try:
            # Create a dictionary from the row using the expected_columns as keys
            # This handles cases where the model might generate values in a slightly different order
            # (though we prompt it to maintain order, it's good to be robust).
            parsed_dict = dict(zip(expected_columns, row))
            
            # Optional: Basic validation of sentiment/confidence values
            if parsed_dict.get('sentiment') not in [e.value for e in SentimentEnum]:
                raise ValueError(f"Invalid sentiment value: {parsed_dict.get('sentiment')}")
            if parsed_dict.get('confidence') not in [e.value for e in ConfidenceEnum]:
                raise ValueError(f"Invalid confidence value: {parsed_dict.get('confidence')}")

            parsed_rows.append(parsed_dict)

        except Exception as e:
            error_rows.append({
                'raw_line': raw_line,
                'error_reason': f"Data parsing/validation error: {e}"
            })

    # If any error rows were found, save the original CSV string for debugging
    if error_rows and error_save_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        error_filename = f"malformed_csv_response_{timestamp}_{unique_id}.txt"
        save_unparseable_json_to_file(error_save_dir, error_filename, csv_string) # Reusing the helper

    return parsed_rows, error_rows

# --- Update classify_sentiment_batched to use the new CSV parsing ---

def classify_sentiment_batched(texts_to_classify: List[str]) -> List[SentimentClassification]:
    client = Client(api_key=GEMINI_API_KEY)
    all_classified_results = []
    
    BATCH_SIZE = 250 # Or even lower if needed
    MAX_RETRIES = 5
    INITIAL_DELAY = 1

    system_instruction_text = "You are a highly accurate sentiment analysis model that classifies text as positive, neutral, or negative, providing a confidence level (high, medium, low)."
    
    user_prompt_header = (
        "Classify the sentiment of the following texts. "
        "For each text, provide the sentiment ('positive', 'neutral', 'negative') "
        "and confidence ('high', 'medium', 'low'). "
        "Respond with a CSV format. The first row should be headers: sentiment,confidence. "
        "Each subsequent row should contain only the sentiment and confidence for a text, in the order provided. "
        "Do not include any other text or preamble, just the CSV content.\n\n"
        "Texts to classify:\n"
    )
    
    # Expected columns for our CSV output
    EXPECTED_CSV_COLUMNS = ["sentiment", "confidence"]

    for i in tqdm(range(0, len(texts_to_classify), BATCH_SIZE), desc="Processing review batches"):
        current_batch_texts = texts_to_classify[i:i + BATCH_SIZE]
        
        if not current_batch_texts:
            continue

        retries = 0
        successful_batch = False
        while retries < MAX_RETRIES and not successful_batch:
            raw_csv_response_array = "" 
            try:
                full_user_input = user_prompt_header + "\n".join([f"- {text}" for text in current_batch_texts])

                response = client.models.generate_content(
                    model=GEMINI_MODEL_NAME, 
                    contents=[full_user_input],
                    config=types.GenerateContentConfig( 
                        # response_mime_type="text/csv", # Changed to CSV
                        temperature=0.01, 
                        system_instruction=system_instruction_text 
                    )
                )
                
                raw_csv_response_array = response.text
                print(f"Batch {i//BATCH_SIZE} (Attempt {retries + 1}): Raw CSV Response Array (start): {raw_csv_response_array[:500]}...")

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = uuid.uuid4().hex[:8]
                # Filename includes batch start index for better organization
                error_filename_prefix = f"batch_{i}_" # Prefix for potential error logs from this batch

                # Use the new robust_csv_parse function
                parsed_list_of_dicts, batch_error_rows = robust_csv_parse(
                    raw_csv_response_array, 
                    expected_columns=EXPECTED_CSV_COLUMNS,
                    error_save_dir=UNPARSEABLE_JSON_DIR # Reusing the directory
                )
                
                # Convert list of dicts to SentimentClassification Pydantic models
                parsed_list = []
                for item_dict in parsed_list_of_dicts:
                    try:
                        parsed_list.append(SentimentClassification(**item_dict))
                    except Exception as e:
                        # This catches Pydantic validation issues even after CSV parsing
                        print(f"Batch {i//BATCH_SIZE}, Item (post-CSV parse): Warning: Failed to convert dictionary to SentimentClassification: {item_dict}. Error: {e}")
                        parsed_list.append(SentimentClassification(sentiment=SentimentEnum.ERROR, confidence=ConfidenceEnum.ERROR))
                
                # Handle items that were marked as errors during CSV parsing
                # These will be appended to parsed_list as ERRORs
                for err_row_info in batch_error_rows:
                     print(f"Batch {i//BATCH_SIZE}: CSV parsing error for row: '{err_row_info.get('raw_line', 'N/A')}' Reason: {err_row_info.get('error_reason', 'Unknown')}")
                     # Append an error classification for each problematic row
                     parsed_list.append(SentimentClassification(sentiment=SentimentEnum.ERROR, confidence=ConfidenceEnum.ERROR))


                # Check if model returned fewer results than inputs for this specific batch
                # Now, len(parsed_list) should ideally include the ERROR classifications from malformed rows
                # The crucial check is still against the original current_batch_texts length.
                if len(parsed_list_of_dicts) < len(current_batch_texts):
                    print(f"Batch {i//BATCH_SIZE} (Attempt {retries + 1}): Warning! Model returned fewer *validly parsed* results ({len(parsed_list_of_dicts)}) than inputs ({len(current_batch_texts)}).")
                    # The robust_csv_parse already saves raw content if errors, so no extra save here unless
                    # there's a complete truncation after all parsing attempts.
                    if not parsed_list_of_dicts and raw_csv_response_array: # Check if completely failed to parse anything
                        print(f"Batch {i//BATCH_SIZE}: Completely failed to parse CSV. Saving raw response for manual review.")
                        save_unparseable_json_to_file(UNPARSEABLE_JSON_DIR, f"{error_filename_prefix}complete_parse_fail_{timestamp}_{unique_id}.txt", raw_csv_response_array)


                # Process results for the current batch
                batch_classified_results = []
                # Ensure we have a classification for every input text, even if it's an ERROR
                # We need to re-align parsed_list back to current_batch_texts, 
                # taking into account parsing errors that might have added ERROR items.
                
                # A more robust approach here: assume model_outputs from API maps to input texts
                # If there are fewer outputs than inputs, fill the rest with errors.
                for j in range(len(current_batch_texts)):
                    if j < len(parsed_list): # parsed_list now contains both good and ERROR items
                        batch_classified_results.append(parsed_list[j])
                    else:
                        # This branch is hit if parsed_list is shorter than current_batch_texts
                        # after all parsing attempts (including error rows).
                        print(f"Batch {i//BATCH_SIZE}, Item {j}: No corresponding classification (due to truncation/severe error). Padding with ERROR.")
                        batch_classified_results.append(SentimentClassification(sentiment=SentimentEnum.ERROR, confidence=ConfidenceEnum.ERROR)) 
                
                all_classified_results.extend(batch_classified_results)
                successful_batch = True 

            except Exception as e:
                retries += 1
                current_delay = INITIAL_DELAY * (2 ** (retries - 1)) 
                print(f"Batch {i//BATCH_SIZE} (Attempt {retries}/{MAX_RETRIES}): An API call or unexpected error occurred: {e}. Retrying in {current_delay} seconds...")
                if raw_csv_response_array: 
                    error_filename_api_exception = f"{error_filename_prefix}api_exception_raw_output_{timestamp}_{unique_id}.txt"
                    save_unparseable_json_to_file(UNPARSEABLE_JSON_DIR, error_filename_api_exception, raw_csv_response_array)
                
                time.sleep(current_delay) 

        if not successful_batch:
            print(f"Batch {i//BATCH_SIZE}: All retries failed. Padding remaining items in this batch with ERROR.")
            all_classified_results.extend([SentimentClassification(sentiment=SentimentEnum.ERROR, confidence=ConfidenceEnum.ERROR) for _ in current_batch_texts])
            
    return all_classified_results
def sample_restaurants_reviews(restaurant_num:int, db_path:Path=DATA_PATH, random_state:int|None=None):
    with duckdb.connect(db_path) as conn:
        restaurants = conn.table("restaurants").fetchdf()
        reviews = conn.table("navermap_reviews").fetchdf()
    sampled_restaurants = restaurants.sample(restaurant_num, random_state=random_state)
    sampled_reviews = reviews[reviews["store_id"].isin(sampled_restaurants["naver_store_id"])]
    print(f"Sampled {len(sampled_reviews)} reviews.")
    return sampled_restaurants, sampled_reviews

def token_limit_reached(review_list:List[str], 
                        token_limit:int=TOKEN_LIMIT_PER_REQUEST, 
                        allowance_tokens:int=100):
    total_string_len = sum(map(len, review_list)) * 0.7 # Rough estimate based on empirical evidence
    if total_string_len >= token_limit - allowance_tokens:
        return True
    return False
def process_model_io(input_data: pd.DataFrame,
                     text_column_name:str,
                     id_column_name:str,
                     model_outputs:List[SentimentClassification]) -> pd.DataFrame:
    to_merge = input_data.copy()
    if len(model_outputs) != len(to_merge):
        print(f"Warning! Mismatch in original data and output data. Input: {len(to_merge)} vs Output: {len(model_outputs)}")
    # container for new df
    all_classified_results = []
    # original ids
    og_ids = to_merge[id_column_name].to_list()

    for i, mod_out in enumerate(model_outputs):
        record_id = og_ids[i]
        all_classified_results.append({
            id_column_name: record_id,
            "sentiment" : mod_out.sentiment.value,
            "confidence": mod_out.confidence.value
        })
    classified_results_df = pd.DataFrame(all_classified_results)

    final_df = pd.merge(
        left=to_merge,
        right= classified_results_df,
        on=id_column_name,
        how="left"
    )
    return final_df


def main():
    sampled_restaurants, sampled_reviews = sample_restaurants_reviews(NUM_RESTAURANTS, DATA_PATH, 72)
    print("These restaurants were sampled:")
    print(sampled_restaurants["store_name"])
    if token_limit_reached(list(sampled_reviews["review_text"])):
        print("Cannot handle this with one batch")
        print("Terminating...")
        return
    df_cleaned = sampled_reviews.copy() 
    df_cleaned['review_text'] = df_cleaned['review_text'].replace('', pd.NA)
    df_cleaned = df_cleaned.dropna(subset=['review_text'])
    sampled_reviews=df_cleaned
    model_inputs = sampled_reviews['review_text'].to_list()
    model_outputs = classify_sentiment_batched(model_inputs)
    final_classified_df = process_model_io(input_data=sampled_reviews,
                                           text_column_name="review_text",
                                           id_column_name="review_id",
                                           model_outputs=model_outputs)
    final_classified_df.to_parquet(SENTIMENT_SAVE_PATH)
    sampled_restaurants.to_csv(RESTAURANT_SAVE_PATH)
#%%
if __name__ == "__main__":
    main()


# %%
