#%%
import streamlit as st
import duckdb
from dotenv import load_dotenv
import os
import tempfile
import requests

@st.cache_resource(ttl="1h")
def get_duckdb_connection():
    print("Getting duckdb connection")
    load_dotenv()
    print("Loaded dotenv")
    db_type = os.getenv("DB_TYPE", "R2").upper()
    print(f"Got db type: {db_type}")
    if db_type == "R2":
        print("Attempting R2")
        r2_access_key_id = st.secrets["R2_ACCESS_KEY_ID"]
        r2_secret_access_key = st.secrets["R2_SECRET_ACCESS_KEY"]
        r2_bucket_name = st.secrets["R2_BUCKET_NAME"]
        r2_account_id = st.secrets["R2_ACCOUNT_ID"]

        db_path_r2_protocol = f"r2://{r2_bucket_name}/reviews.db"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Connect to an in-memory database initially
                con = duckdb.connect(
                    database=':memory:',
                    read_only=False, # Temporarily allow write for secret creation
                    config={'allow_unsigned_extensions': 'true'}
                )
                con.install_extension('httpfs')
                con.load_extension('httpfs')

                # Create the R2 secret for native R2 access
                con.execute(f"""
                    CREATE SECRET my_r2_secret (
                        TYPE r2,
                        KEY_ID '{r2_access_key_id}',
                        SECRET '{r2_secret_access_key}',
                        ACCOUNT_ID '{r2_account_id}'
                    );
                """)

                # Attach the remote DuckDB database
                con.execute(f"ATTACH '{db_path_r2_protocol}' AS reviews (READ_ONLY TRUE);")

                # Perform a quick test query to ensure connection is live and attached DB is accessible
                con.execute("SELECT 1 FROM reviews.restaurants LIMIT 1;").fetchone()
                # st.success(f"Successfully connected to DuckDB file on R2 on attempt {attempt + 1}.")
                return con # Return the live connection

            except Exception as e:
                st.warning(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    # Add a small delay before retrying
                    import time
                    time.sleep(1)
                else:
                    st.error(f"FATAL ERROR: Could not establish DuckDB connection to R2 after {max_retries} attempts.")
                    st.info("Please ensure your R2 bucket name, account ID, access keys are correct and the file exists at the specified path in Cloudflare R2 secrets.")
                    st.stop() # Stop the app execution if all retries fail
    elif db_type == "LOCAL":
        raw_db_path = os.getenv("DB_PATH", "H:/My Drive/reviews.db")
        db_path = os.path.normpath(os.path.abspath(raw_db_path))
        try:
            con = duckdb.connect(db_path, read_only=True)
            return con
        except Exception as e:
            print("Failed local db connection")
            st.warning(f"Connection to Google Drive DuckDB failed!:{e}")
            print("Program terminating...")
            st.stop()
    return None

@st.cache_resource
def download_font_for_wordcloud(font_url: str, font_filename: str) -> str:
    """
    Downloads a font file from a URL to a temporary local path on the server.
    This function is cached to avoid re-downloading the font on every rerun.
    Logs progress to the terminal (stdout/stderr).

    Args:
        font_url (str): The public URL of the font file (e.g., from R2).
        font_filename (str): The desired filename for the downloaded font (e.g., "NanumGothic.ttf").

    Returns:
        str: The local file path to the downloaded font, or None if download fails.
    """
    temp_dir = tempfile.gettempdir()
    local_font_path = os.path.join(temp_dir, font_filename)

    if os.path.exists(local_font_path):
        print(f"INFO: Font already exists at: `{local_font_path}` (cached).")
        return local_font_path

    print(f"INFO: Downloading font from R2: `{font_url}` to `{local_font_path}`...")
    try:
        with requests.get(font_url, stream=True) as response:
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            with open(local_font_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("SUCCESS: Font downloaded successfully! ✅")
        return local_font_path
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Error downloading font from R2: {e} 😞. Please check the URL and R2 bucket access.")
        return None