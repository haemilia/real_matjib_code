#%%
import streamlit as st
import duckdb
import plotly.graph_objects as go
import pandas as pd
from typing import Tuple, Any

# --- Function to get map data from DuckDB ---
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
    from pyproj import CRS, Transformer
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
    
def get_map_data(con:duckdb.DuckDBPyConnection) -> Tuple[pd.Series|None]:
    """
    Queries X_EPSG_5174(longitude), Y_EPSG_5174(latitude), store_name from table `restaurants`.
    Returns columns as tuple of pd.Series.
    """
    table_name = "restaurants"

    try:
        query = f"""
        SELECT
            X_EPSG_5174,
            Y_EPSG_5174,
            store_name
        FROM reviews.{table_name}
        """
        df = con.execute(query).fetchdf()

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
        return (None, None, None) # Return tuple of Nones if there's an error
    

def plot_restaurants_on_map(
        #음식점 위도/경도, 음식점명, 타이틀
        df, title='연남동 일반 음식점 지도 시각화',
        #연남동 위도/경도
        yeonnam_lat=37.5628, yeonnam_lon=126.9222, zoom=15,
        active_restaurant = None,# Name of selected restaurant
        ):

    fig = go.Figure()   #fig 객체 생성
    if active_restaurant:
        # 선택된 음식점과 아닌 음식점 분리
        active_df = df[df["store_name"] == active_restaurant]
        other_df = df[df["store_name"] != active_restaurant]
        if not active_df.empty:
            # 선택된 음식점 표시
            fig.add_trace(
                go.Scattermap(
                    lat=active_df["restaurant_lat"],    #음식점 위도
                    lon=active_df["restaurant_long"],    #음식점 경도
                    mode='markers',
                    #마커 스타일 지정
                    marker=dict(
                        size=20,
                        color='yellow',
                        opacity=1.0,
                        symbol="circle"),
                    text=active_df["store_name"],    #음식점명
                    hoverinfo='text'    #마커에 마우스 hover시, 음식점명 출력,
                )
            )
            fig.add_trace(
                go.Scattermap(
                    lat=active_df["restaurant_lat"],    #음식점 위도
                    lon=active_df["restaurant_long"],    #음식점 경도
                    mode='markers+text',
                    textposition="top center",
                    #마커 스타일 지정
                    marker=dict(
                        # size=20,
                        color='yellow',
                        opacity=1.0,
                        symbol="restaurant"),
                    text=active_df["store_name"],    #음식점명
                    hoverinfo='text'    #마커에 마우스 hover시, 음식점명 출력,
                )
            )
        if not other_df.empty:
            # 선택 안 된 음식점 표시
            fig.add_trace(
                go.Scattermap(
                    lat=other_df["restaurant_lat"],    #음식점 위도
                    lon=other_df["restaurant_long"],    #음식점 경도
                    mode='markers',
                    #마커 스타일 지정
                    marker=dict(
                        size=5,
                        color='red',
                        opacity=0.4,
                        symbol='circle'),
                    text=other_df["store_name"],    #음식점명
                    hoverinfo='text'    #마커에 마우스 hover시, 음식점명 출력,
                )
            )
        #그래프의 레이아웃과 시각적 속성 설정
        title_text = dict(text=active_restaurant)
        map_width = 400
        map_height = None
        center_lat = active_df["restaurant_lat"].iloc[0]
        center_lon = active_df["restaurant_long"].iloc[0]
        margin = dict(b=20, l=20, r=20)
        zoom = 17

    else:
        restaurants_long = df["restaurant_long"]
        restaurants_lat = df["restaurant_lat"]
        store_name = df["store_name"]
        fig.add_trace(
            go.Scattermap(
                lat=restaurants_lat,    #음식점 위도
                lon=restaurants_long,    #음식점 경도
                mode='markers',
                #마커 스타일 지정
                marker=dict(
                    size=10,
                    color='red',
                    opacity=0.4,
                    symbol='circle'),
                text=store_name,    #음식점명
                hoverinfo='text'    #마커에 마우스 hover시, 음식점명 출력,
            )
        )
        title_text = dict(text=title)
        map_height = 700
        map_width = None
        center_lat = yeonnam_lat
        center_lon = yeonnam_lon
        margin = None
        zoom = zoom
    fig.update_layout(
    title=title_text, #음식점 이름을 제목으로
    autosize=True,  #창 크기에 따라 자동으로 그래프 크기 조절
    width = map_width,
    height = map_height,
    hovermode='closest',    #마우스를 가장 가까운 점에만 반응하도록 설정
    showlegend=False,   #범례 표시 여부
    margin=margin,
    map=dict(
        bearing=0,  #지도 회전 각도
        #지도 중심 좌표(선택된 음식점)
        center=dict(
            lat=center_lat,
            lon=center_lon
            ),
        zoom=zoom,
        style='outdoors'    #지도 배경 스타일
        )
    )

    return fig   #plotly.fig 객체 리턴

