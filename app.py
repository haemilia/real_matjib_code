#%%
import streamlit as st
import duckdb
from web.pages.main_map_view import main_map_view
from web.pages.restaurant_detail_view import restaurant_detail_view
from web.utils.database import get_duckdb_connection
# from viz.restaurants_map_plot import plot_restaurants_on_map, get_map_data
# Set layout as wide
st.set_page_config(
    page_title="진품명품: 진짜 맛집 찾기",
    layout="wide",
    initial_sidebar_state="collapsed")

# --- Main Streamlit App Logic ---
st.header("진품명품 : 진짜 맛집 찾기")

if 'current_page' not in st.session_state:
    st.session_state.current_page = "main_map" # "main_map" / "detail_view"
if 'selected_restaurant' not in st.session_state:
    st.session_state.selected_restaurant = None
# if 'map_df' not in st.session_state:
#     st.session_state.map_df = None
# Get the cached DuckDB connection
# con = get_duckdb_connection()
# con = get_gdrive_duckdb_connection() # For development

# if con: # Only proceed if connection was successful
if st.session_state.current_page == "main_map":
    print("Showing main view")
    main_map_view()
    # st.session_state.map_df = map_df
elif st.session_state.current_page == "detail_view":
    restaurant_detail_view()


