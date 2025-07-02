import os
import subprocess
import sys
import getpass # For secure input of the API key
from google import genai
from pydantic import BaseModel, Field
import duckdb
import pandas as pd
import json
from tqdm import tqdm

ENV_VAR_NAME = "GEMINI_API_KEY"

DATA_PATH = "" #Input data parquet # If DB mode, insert path to DB
DATA_NAME = "" # DB Table name
COLUMN_NAME = "" # Column name of the text column
SAVE_PATH = "" # Output data parquet  # If DB mode, insert new table name


def set_powershell_env_var(var_name, var_value):
    """
    Sets a session-level environment variable in PowerShell.
    This uses a subprocess call to execute a PowerShell command.
    """
    #NOTE: This command sets the variable for the *current* PowerShell session
    # from which the Python script was launched. It does NOT persist.
    powershell_command = f'[Environment]::SetEnvironmentVariable("{var_name}", "{var_value}", "Process")'
    
    try:
        # Use subprocess.run for cleaner handling.
        # shell=True | needed to execute the command string directly in the shell
        # capture_output=True | to potentially see errors
        # text=True | to decode output
        result = subprocess.run(
            ["powershell.exe", "-Command", powershell_command],
            capture_output=True,
            text=True,
            check=True # Raise an exception for non-zero exit codes
        )
        print(f"PowerShell command output: {result.stdout.strip()}")
        if result.stderr:
            print(f"PowerShell command error: {result.stderr.strip()}", file=sys.stderr)
        print(f"Successfully set {var_name} as session environment variable.")
    except subprocess.CalledProcessError as e:
        print(f"Error setting PowerShell environment variable: {e}", file=sys.stderr)
        print(f"STDOUT: {e.stdout}", file=sys.stderr)
        print(f"STDERR: {e.stderr}", file=sys.stderr)
        sys.exit(1) # Exit if we can't set the variable
    except Exception as e:
        print(f"An unexpected error occurred while setting env var: {e}", file=sys.stderr)
        sys.exit(1)

def get_or_set_api_key():
    """
    Checks for the API key in environment, prompts user if not found,
    and sets it as a session environment variable if needed.
    """
    api_key = os.getenv(ENV_VAR_NAME)

    if api_key:
        print(f"API Key '{ENV_VAR_NAME}' found in environment. Using existing key.")
        return api_key
    else:
        print(f"API Key '{ENV_VAR_NAME}' not found in environment.")
        print("\n--- API Key Input ---")
        # Use getpass for secure input
        print("Your input will not be seen for security reasons!")
        user_input_key = getpass.getpass("Please enter your API Key: ")
        print("---------------------\n")

        if user_input_key:
            # Set it in the current Python process's environment
            os.environ[ENV_VAR_NAME] = user_input_key
            print(f"API Key '{ENV_VAR_NAME}' set in current Python process.")

            # Also set it in the parent PowerShell session
            # This makes it available to other commands/scripts run in that same PowerShell window
            set_powershell_env_var(ENV_VAR_NAME, user_input_key)
            return user_input_key
        else:
            print("No API Key was entered. Cannot proceed.")
            sys.exit(1) # Exit if no key 

class SentimentClassification(BaseModel):
    """Represents the sentiment classification of a given text."""
    sentiment: str = Field(description="The classified sentiment of the text.", examples=["positive", "neutral", "negative"])
    confidence: str = Field(description="The confidence level of the sentiment classification.", examples=["high", "medium", "low"])


def classify_sentiment(text_to_classify:str):
    try:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY")) # Just in case...
        # The client gets the API key from the environment variable `GEMINI_API_KEY`.

        system_instruction = ("You are a very proficient sentiment analysis model."
                            "Your job is to classify the user's input into one of the three sentiment categories."
                            "You will also provide how confident you are about each classification."
                            "The sentiment categories are: ['positive', 'neutral', 'negative']"
                            "The confidence levels are: ['high', 'medium', 'low']"
                            "Your response for the classification MUST be ONE of the sentiment categories ('positive', 'neutral', 'negative')."
                            "Your response will include the sentiment classification, and the confidence level.")
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=text_to_classify,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=SentimentClassification,
                temperature=0.01 # low, deterministic model output
                )

        )
        raw_json_response = response.text
        parsed = json.loads(raw_json_response)
        classified_sentiment = SentimentClassification(**parsed)
        return classified_sentiment
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}. Raw response: {response.text}")
        return {"sentiment": "Error", "details": "Invalid JSON from model"}
    except Exception as e:
        print("An error occurred: ", e)
        return {"sentiment": "Error", "details":str(e)}

def get_text_from_dataset(dataset_path:str, text_column:str)->pd.Series:
    try:
        # Dataset is a parquet file
        df = pd.read_parquet(dataset_path)
        return df[text_column]
    except Exception as e:
        print("There was an error loading the text data: ", e)
        raise Exception # Because this is a significant error
    
def get_text_from_DB(DB_path:str, table_name:str, text_column:str) -> pd.Series:
    try:
        with duckdb.connect(DB_path) as con:
            query = f"SELECT {text_column} FROM {table_name}"
            df = con.execute(query).fetchdf()
            return df[text_column]
    except Exception as e:
        print("There was an error loading the text data: ", e)
        raise Exception # Because this is a significant error

def loop_through_text(text_data:pd.Series) -> pd.DataFrame:
    collected_results = []
    for one_text in tqdm(text_data):
        result = {}
        result ["text"] = one_text
        sentiment_result = classify_sentiment(one_text)
        result["sentiment"] = sentiment_result.sentiment
        result["confidence"] = sentiment_result.confidence
        collected_results.append(result)
    return pd.DataFrame(collected_results)

def save_table_to_DB(table_df:pd.DataFrame, DB_path:str, new_table_name:str):
    try:
        with duckdb.connect(DB_path) as con:
            query = f"CREATE OR REPLACE TABLE {new_table_name} AS SELECT * FROM table_df"
            con.execute(query)
    except Exception as e:
        print("There was an error saving the text data to the database: ", e)
        raise Exception # Fatal error

### MAIN USER INTERFACE
def main():
    api_key = get_or_set_api_key()

    if api_key:
        # Mask the key for display, showing only the last few chars
        masked_key = '*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else api_key
        print(f"Using API Key: {masked_key}")

        mode = input("Will you use DB or parquet? Press 'd' for DB, 'p' for Parquet. (d/p): ")
        if mode == "d":
            if not DATA_PATH:
                DATA_PATH = input("Please enter path to database: ")
            if not DATA_NAME:
                DATA_NAME = input("Please enter the name of the table: ")
            if not COLUMN_NAME:
                COLUMN_NAME = input("Please enter the column name of the column that contains the text to be classified: ")

            text_data = get_text_from_DB(DB_path=DATA_PATH, 
                                         table_name=DATA_NAME,
                                            text_column=COLUMN_NAME)
            classified_results = loop_through_text(text_data)
            if not SAVE_PATH:
                SAVE_PATH = input("Please enter name of the new table that will be saved to the database: ")
            save_table_to_DB(classified_results,
                             DB_path=DATA_PATH,
                             new_table_name=SAVE_PATH)
        elif mode == "p":
            if not DATA_PATH:
                DATA_PATH = input("Please enter the path to the dataset parquet file: ")
            if not COLUMN_NAME:
                COLUMN_NAME = input("Please enter the column name of the column that contains the text to be classified: ")
            text_data = get_text_from_dataset(dataset_path=DATA_PATH, 
                                            text_column=COLUMN_NAME)
            classified_results = loop_through_text(text_data)
            if not SAVE_PATH:
                SAVE_PATH = input("Please enter the path to save the classified results.")
            classified_results.to_parquet(SAVE_PATH)
        else:
            print("Wrong input. Terminating...")
    else:
        print("Failed to obtain API key. Exiting.")

if __name__ == "__main__":
    main()