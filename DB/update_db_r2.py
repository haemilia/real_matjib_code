import os
import boto3
from dotenv import load_dotenv
#### DO NOT RUN THIS CODE WITHOUT PROVIDING A .env FILE!!
def get_local_r2_env_vars(local_db_file_path:str) -> dict:
    # --- Configuration ---
    # Load environment variables from .env file for local development
    try:
        load_dotenv()
        local_r2_env_vars = {}

        local_r2_env_vars["LOCAL_DB_FILE_PATH"] = local_db_file_path 
        local_r2_env_vars["R2_BUCKET_NAME"] = os.getenv("R2_BUCKET_NAME")
        local_r2_env_vars["R2_KEY_PATH"] = "reviews.db" # DO NOT CHANGE!!

        # R2 Credentials from environment variables
        local_r2_env_vars["R2_ACCESS_KEY_ID"] = os.getenv("R2_ACCESS_KEY_ID")
        local_r2_env_vars["R2_SECRET_ACCESS_KEY"] = os.getenv("R2_SECRET_ACCESS_KEY")
        local_r2_env_vars["R2_ENDPOINT_URL"] = os.getenv("R2_ENDPOINT_URL") # E.g., "https://<ACCOUNT_ID>.r2.cloudflarestorage.com"
        return local_r2_env_vars
    except Exception as e:
        print("Error while getting credentials: ", e)
        raise e
    

# --- Main Upload Function ---
def upload_db_to_r2(env_vars:dict):
    LOCAL_DB_FILE_PATH = env_vars["LOCAL_DB_FILE_PATH"]
    R2_BUCKET_NAME = env_vars["R2_BUCKET_NAME"]
    R2_KEY_PATH = env_vars["R2_KEY_PATH"]
    R2_ACCESS_KEY_ID = env_vars["R2_ACCESS_KEY_ID"]
    R2_SECRET_ACCESS_KEY = env_vars["R2_SECRET_ACCESS_KEY"]
    R2_ENDPOINT_URL = env_vars["R2_ENDPOINT_URL"]
    print(f"Starting upload of '{LOCAL_DB_FILE_PATH}' to R2 bucket '{R2_BUCKET_NAME}' at key '{R2_KEY_PATH}'...")

    if not os.path.exists(LOCAL_DB_FILE_PATH):
        print(f"Error: Local DB file not found at '{LOCAL_DB_FILE_PATH}'. Aborting.")
        return False
        
    session = boto3.Session(
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY
    )

    s3_client = session.client(
        's3',
        endpoint_url=R2_ENDPOINT_URL,
        region_name='auto' # R2 doesn't have regions like AWS, 'auto' is common or any placeholder
    )

    try:
        # Upload the file
        s3_client.upload_file(
            LOCAL_DB_FILE_PATH,
            R2_BUCKET_NAME,
            R2_KEY_PATH,
            # You can add callback for progress if needed for very large files
            # Callback=lambda bytes_transferred: print(f"{bytes_transferred} bytes transferred...")
        )
        print(f"Successfully uploaded '{LOCAL_DB_FILE_PATH}' to R2 at '{R2_BUCKET_NAME}/{R2_KEY_PATH}'.")
        return True
    except Exception as e:
        print(f"Error uploading file to R2: {e}")
        return False

# --- Execute if script is run directly ---
if __name__ == "__main__":
    r2_env_vars = get_local_r2_env_vars(local_db_file_path= r"H:\My Drive\reviews.db")
    if upload_db_to_r2(env_vars=r2_env_vars): # Use raw string for Windows paths
        print("DB update process completed.")
    else:
        print("DB update process failed.")