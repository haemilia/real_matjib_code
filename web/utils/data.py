import streamlit as st
import duckdb
import pandas as pd
from typing import Tuple, Any
from pyproj import CRS, Transformer
from web.utils.database import get_duckdb_connection

@st.cache_data(ttl="1h")
def _execute_cached_query_to_df(query:str) -> pd.DataFrame:
    con = get_duckdb_connection()
    if con is None:
        raise ConnectionError("Failed to connect to duckdb")
    try:
        df = con.execute(query).fetchdf()
        return df
    except Exception as e:
        print(f"Error while executing query: {query}; Error: {e}")
        st.error(f"Error while executing query to DB: {e}")
        st.stop
    return pd.DataFrame([])

#### Helper Functions ######################################################################################
def convert_epsg5174_to_wgs84(x:pd.Series, y:pd.Series) -> Tuple[Any, Any]:
    """
    Converts coordinates from EPSG:5174 to WGS 84 (EPSG:4326).
    Not super accurate...

    Args:
        x (float): The Easting coordinate (x) in EPSG:5174.
        y (float): The Northing coordinate (y) in EPSG:5174.

    Returns:
        tuple: A tuple containing (longitude(x), latitude(y)) in WGS 84.
               Returns None if the transformation fails.
    """
    try:
        # Define the source coordinate system (EPSG:5174)
        crs_from = CRS.from_proj4("+proj=tmerc +lat_0=38 +lon_0=127.002890277778 +k=1 +x_0=200000 +y_0=500000 +ellps=bessel +towgs84=-145.907,505.034,685.756,-1.162,2.347,1.592,6.342 +units=m +no_defs")

        # Define the target coordinate system (WGS 84 - EPSG:4326)
        crs_to = CRS.from_epsg(4326)

        # Create a transformer
        transformer = Transformer.from_crs(crs_from, crs_to)

        # Perform the transformation
        latitude, longitude = transformer.transform(x, y)

        return longitude, latitude
    except Exception as e:
        print(f"An error occurred during the transformation: {e}")
        return None, None
def get_map_data() -> pd.DataFrame:
    """
    Queries X_EPSG_5174(longitude), Y_EPSG_5174(latitude), store_name from table `restaurants`.
    Returns columns as a DataFrame. Returns None if there's an error.
    """
    print("Entered get_map_data")
    table_name = "restaurants"

    try:
        query = f"""
        SELECT
            X_EPSG_5174,
            Y_EPSG_5174,
            store_name
        FROM reviews.{table_name}
        """
        df = _execute_cached_query_to_df(query)

        # Convert coordinate columns to numeric
        df['X_EPSG_5174'] = pd.to_numeric(df['X_EPSG_5174'], errors='coerce')
        df['Y_EPSG_5174'] = pd.to_numeric(df['Y_EPSG_5174'], errors='coerce')

        # Drop rows where coordinates might have become NaN due to coercion
        df.dropna(subset=['X_EPSG_5174', 'Y_EPSG_5174'], inplace=True)
        converted_x, converted_y = convert_epsg5174_to_wgs84(df['X_EPSG_5174'], df['Y_EPSG_5174'])
        df["restaurant_long"] = converted_x
        df["restaurant_lat"] = converted_y
        result_df = df[["restaurant_long", "restaurant_lat", "store_name"]]

        return result_df
    except Exception as e:
        st.error(f"Error querying data for map from table '{table_name}': {e}")
        return pd.DataFrame([])