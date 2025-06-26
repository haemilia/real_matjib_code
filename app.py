import streamlit as st
import duckdb
import plotly.express as px
import os
import pandas as pd

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
            con.execute(f"ATTACH '{db_path_r2_protocol}' AS reviews_db (READ_ONLY TRUE);")

            # Perform a quick test query to ensure connection is live and attached DB is accessible
            con.execute("SELECT 1 FROM reviews_db.restaurants LIMIT 1;").fetchone()
            st.success(f"Successfully connected to DuckDB file on R2 on attempt {attempt + 1}.")
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

# --- Function to get map data from DuckDB ---
def get_map_data(duckdb_connection):
    """
    Queries the first 10 rows of store_name, jibun_address, X_naver_WGS_84 (longitude),
    and Y_naver_WGS_84 (latitude) from the 'restaurants' table,
    and ensures coordinate columns are numeric.
    """
    table_name = "restaurants"

    try:
        query = f"""
        SELECT
            store_name,
            jibun_address,
            X_naver_WGS_84, -- Longitude
            Y_naver_WGS_84  -- Latitude
        FROM reviews_db.{table_name}
        LIMIT 10;
        """
        df = duckdb_connection.execute(query).fetchdf()

        # Convert coordinate columns to numeric
        df['X_naver_WGS_84'] = pd.to_numeric(df['X_naver_WGS_84'], errors='coerce')
        df['Y_naver_WGS_84'] = pd.to_numeric(df['Y_naver_WGS_84'], errors='coerce')

        # Drop rows where coordinates might have become NaN due to coercion
        df.dropna(subset=['X_naver_WGS_84', 'Y_naver_WGS_84'], inplace=True)

        return df
    except Exception as e:
        st.error(f"Error querying data for map from table '{table_name}': {e}")
        return pd.DataFrame() # Return empty DataFrame on error

# --- Function to plot the map ---
def plot_stores_on_map(df):
    """
    Plots store locations on a Plotly map.
    Args:
        df (pd.DataFrame): DataFrame containing store_name, jibun_address,
                           X_naver_WGS_84 (lon), Y_naver_WGS_84 (lat).
    Returns:
        plotly.graph_objects.Figure: The Plotly map figure.
    """
    if df.empty:
        st.warning("No data available to plot on the map.")
        return None

    center_lat = df['Y_naver_WGS_84'].mean()
    center_lon = df['X_naver_WGS_84'].mean()

    # Use px.scatter_map as recommended (replaces scatter_mapbox)
    fig = px.scatter_map( # Changed to scatter_map
        df,
        lat="Y_naver_WGS_84",
        lon="X_naver_WGS_84",
        hover_name="store_name",
        hover_data={"jibun_address": True, "store_name": False},
        zoom=12,
        center={"lat": center_lat, "lon": center_lon},
        title="Top 10 Stores by Location"
    )

    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})

    return fig

# --- Main Streamlit App Logic ---
st.header("Real Matjib Data - Powered by DuckDB on Cloudflare R2")

# Get the cached DuckDB connection
con = get_r2_duckdb_connection()

if con: # Only proceed if connection was successful
    st.subheader("Data from R2 DuckDB")


    # --- Fetch and Plot Map Data ---
    st.subheader("Store Locations Map")
    map_data_df = get_map_data(con) # Use the 'con' guaranteed to be live

    if not map_data_df.empty:
        map_figure = plot_stores_on_map(map_data_df)
        if map_figure:
            st.plotly_chart(map_figure, use_container_width=True) # THIS IS WHERE THE MAP IS SHOWN
    else:
        st.error("Could not retrieve data to plot the map. Please check the table name 'restaurants' and data availability.")