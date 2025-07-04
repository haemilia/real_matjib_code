#%%
import streamlit as st
import duckdb
from viz.main_map_view import main_map_view
from viz.restaurant_detail_view import restaurant_detail_view
# from viz.restaurants_map_plot import plot_restaurants_on_map, get_map_data
# Set layout as wide
st.set_page_config(
    page_title="진품명품: 진짜 맛집 찾기",
    layout="wide",
    initial_sidebar_state="collapsed")

# --- DuckDB Connection to R2 ---
@st.cache_resource(ttl="1h") # Cache the DuckDB connection for up to 1 hour
def get_r2_duckdb_connection():
    # This function is designed to ALWAYS return a fresh, open connection,
    # or stop the app if connection cannot be established.

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

    # This line should theoretically not be reached due to st.stop()
    return None

@st.cache_resource(ttl="1h")
def get_gdrive_duckdb_connection():
    try:
        con = duckdb.connect(r"H:\My Drive\reviews.db")
    except Exception as e:
        st.warning("Connection to Google Drive DuckDB failed!")
        st.stop()
    else:
        return con
    return None

#%%
#%%

# --- Main Streamlit App Logic ---
st.header("진품명품 : 진짜 맛집 찾기")

if 'current_page' not in st.session_state:
    st.session_state.current_page = "main_map" # "main_map" / "detail_view"
if 'selected_restaurant' not in st.session_state:
    st.session_state.selected_restaurant = None
if 'map_df' not in st.session_state:
    st.session_state.map_df = None
# Get the cached DuckDB connection
con = get_r2_duckdb_connection()
# con = get_gdrive_duckdb_connection() # For development

if con: # Only proceed if connection was successful
    if st.session_state.current_page == "main_map":
        map_df = main_map_view(con)
        st.session_state.map_df = map_df
    elif st.session_state.current_page == "detail_view":
        restaurant_detail_view(con, st.session_state.map_df)
