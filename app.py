import streamlit as st
import pandas as pd
import plotly.express as px
import duckdb

# --- DuckDB Connection to R2 ---
@st.cache_resource(ttl="1h") # Cache the DuckDB connection for 1 hour
def get_r2_duckdb_connection():
    # Retrieve R2 credentials from Streamlit secrets
    r2_access_key_id = st.secrets["R2_ACCESS_KEY_ID"]
    r2_secret_access_key = st.secrets["R2_SECRET_ACCESS_KEY"]
    r2_bucket_name = st.secrets["R2_BUCKET_NAME"]
    r2_account_id = st.secrets["R2_ACCOUNT_ID"]

    db_path_r2_protocol = f"r2://{r2_bucket_name}/reviews.db"

    try:
        con = duckdb.connect(
            database=':memory:',
            read_only=False,
            config={'allow_unsigned_extensions': 'true'}
        )
        con.install_extension('httpfs')
        con.load_extension('httpfs')

        # Create the R2 secret
        con.execute(f"""
            CREATE SECRET my_r2_secret (
                TYPE r2,
                KEY_ID '{r2_access_key_id}',
                SECRET '{r2_secret_access_key}',
                ACCOUNT_ID '{r2_account_id}'
            );
        """)

        # Attach the remote DuckDB database using the R2 protocol and the named secret
        con.execute(f"ATTACH '{db_path_r2_protocol}' AS reviews_db (READ_ONLY TRUE);")

        st.success(f"Successfully connected to DuckDB file on R2: {db_path_r2_protocol}")
        return con

    except Exception as e:
        st.error(f"Error connecting to DuckDB on R2: {e}")
        st.info("Please ensure your R2 bucket name, account ID, access keys are correct and the file exists at the specified path.")
        st.stop() # Stop the app if connection fails to prevent further errors
        return None
    
# --- Function to get map data from DuckDB ---
def get_map_data(duckdb_connection):
    """
    Queries the first 10 rows of store_name, jibun_address, X_naver_WGS_84 (longitude),
    and Y_naver_WGS_84 (latitude) from the 'restaurants' table,
    and ensures coordinate columns are numeric.
    """
    table_name = "restaurants" # Corrected table name based on user feedback

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

        # --- CRUCIAL FIX: Convert coordinate columns to numeric ---
        # Using pd.to_numeric with errors='coerce' is robust.
        # It will convert valid numbers to float and any non-numeric strings to NaN.
        df['X_naver_WGS_84'] = pd.to_numeric(df['X_naver_WGS_84'], errors='coerce')
        df['Y_naver_WGS_84'] = pd.to_numeric(df['Y_naver_WGS_84'], errors='coerce')

        # Drop rows where coordinates might have become NaN due to coercion (if any invalid data)
        df.dropna(subset=['X_naver_WGS_84', 'Y_naver_WGS_84'], inplace=True)
        # --- END CRUCIAL FIX ---

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

    # Calculate approximate center for initial map view (using the provided 10 rows)
    center_lat = df['Y_naver_WGS_84'].mean()
    center_lon = df['X_naver_WGS_84'].mean()

    # Create the scatter mapbox plot
    fig = px.scatter_mapbox(
        df,
        lat="Y_naver_WGS_84", # Latitude column
        lon="X_naver_WGS_84", # Longitude column
        hover_name="store_name", # Column for primary hover text
        hover_data={"jibun_address": True, "store_name": False}, # Additional data to show on hover (hide store_name as it's hover_name)
        zoom=12, # Adjust initial zoom level as needed
        center={"lat": center_lat, "lon": center_lon}, # Center the map on the data
        mapbox_style="carto-positron", # A clean, light map style
        title="Top 10 Stores by Location"
    )

    # Update layout to remove margin and make it fill the space better
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})

    return fig

# --- Main Streamlit App Logic ---
st.header("Real Matjib Data - Powered by DuckDB on Cloudflare R2")

# Get the cached DuckDB connection
con = get_r2_duckdb_connection()

if con: # Only proceed if connection was successful
    st.subheader("Data from R2 DuckDB")

    # Option 1: Show all tables from all attached databases, then filter
    all_tables_df = con.execute("SHOW ALL TABLES;").fetchdf()
    # Filter to only show tables from 'reviews_db'
    r2_tables = all_tables_df[all_tables_df['database'] == 'reviews_db']

        # This block now correctly uses 'restaurants' as the known_table_name
    if not r2_tables.empty:
        # This will now correctly pick 'restaurants' if it's the first table, or any other if it's not.
        first_table_name = r2_tables['name'].iloc[0]
        st.info(f"Showing first 10 rows from table: '{first_table_name}'")
        query_data = f"SELECT * FROM reviews_db.{first_table_name} LIMIT 10;"
        df = con.execute(query_data).fetchdf()
        st.dataframe(df)
    else:
        st.warning("No tables found in your DuckDB file on R2 via metadata. Please ensure your DuckDB file contains tables.")
        # Fallback to the *correct* hardcoded table name
        known_table_name = "restaurants" # Corrected hardcoded table name
        st.info(f"As a fallback, trying to show first 10 rows from hardcoded table: '{known_table_name}'")
        try:
            query_data_known = f"SELECT * FROM reviews_db.{known_table_name} LIMIT 10;"
            df_known = con.execute(query_data_known).fetchdf()
            st.dataframe(df_known)
            st.success(f"Successfully queried hardcoded table '{known_table_name}'!")
        except Exception as e_known:
            st.error(f"Error querying hardcoded table '{known_table_name}': {e_known}")
            st.info("Please ensure the hardcoded table name is correct and exists in your DuckDB file.")

# --- Fetch and Plot Map Data ---
    st.subheader("Store Locations Map")
    map_data_df = get_map_data(con)

    if not map_data_df.empty:
        # Plot the map if data is available
        map_figure = plot_stores_on_map(map_data_df)
        if map_figure: # Ensure figure was successfully created
            st.plotly_chart(map_figure, use_container_width=True)
    else:
        st.error("Could not retrieve data to plot the map. Please check the table name 'restaurants' and data availability.")

    con.close() # Close the connection when done