import streamlit as st
import pandas as pd
from typing import Tuple, Any
from pyproj import CRS, Transformer
from web.utils.database import get_duckdb_connection
import re
import json

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

    try:
        query = """
        SELECT
            X_EPSG_5174,
            Y_EPSG_5174,
            store_name
        FROM reviews.kakaomap_restaurants
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
        st.error(f"Error querying data for map: {e}")
        return pd.DataFrame([])


def get_kakaomap_data(click_store):
    query = """
        SELECT
            r.store_name, r.road_address,
            l.predicted_label, l.kakaomap_id, l.rating, l.reviewer_name, l.review_text, l.photo_url, l.processed_cleaned, l.realreview_prob, l.review_date
        FROM
            kakaomap_reviews_labelled l
        JOIN
            kakaomap_restaurants r
        ON
            l.kakaomap_id = r.kakaomap_id
    """
    df_kakaomap = _execute_cached_query_to_df(query)
    df_kakaomap['predicted_label'] = df_kakaomap['predicted_label'].map({0:'홍보성', 1:'진정성'})
    df_kakaomap['review_date'] = pd.to_datetime(df_kakaomap['review_date']).dt.strftime('%Y-%m-%d')
    
    #'스시정인'이라는 음식점으로 test
    store_name = click_store
    #print(df_kakaomap)
    df_store = df_kakaomap.query("store_name == @store_name").reset_index()
    #print(df_store)
    store_name = re.sub(r'\(.*?\)', '', store_name)  #괄호와 그 안의 내용 제거

    ## -------------------------------------------------------------------------------------------------------------

    if not df_store.empty:
        
        #파이차트에 쓰일 변수
        pre_label_name = list(df_store.predicted_label.unique()) #홍보성/진정성
        pre_label_value = df_store['predicted_label'].value_counts() #라벨링값
        pie_label_list = [pre_label_name, pre_label_value]

        #리뷰가 '진정성'이 있는 것들만 필터링
        df_store_real = df_store.query("predicted_label == '진정성'")

        if df_store_real.empty: #'진정성' 리뷰가 없을 떄
            bar_rating_list = []    #막대그래프 변수
            wordcloud_text = '' #워드클라우드 변수
            real_rating = ''    #진정성 리뷰 평점 변수
        
        else:
            #막대그래프에 쓰일 변수
            rating_xlabel = sorted(list(df_store_real.rating.unique()))  #x축 라벨
            rating_ylabel = sorted(df_store_real.rating.value_counts())  #y축 값
            bar_rating_list = [rating_xlabel, rating_ylabel]

            #워드클라우드
            wordcloud_review = df_store_real.processed_cleaned

            #모든 리뷰의 토큰을 하나의 리스트로 합침
            all_words = []
            for review in wordcloud_review:
                if isinstance(review, list):
                    all_words.extend(review)
                elif isinstance(review, str):
                    review_json = review.replace("'", '"')
                    word_list = json.loads(review_json)
                    all_words.extend(word_list)

            #하나의 텍스트로 합치기
            wordcloud_text = ' '.join(all_words)

            #상세페이지 칸에 쓰일 것
            real_rating = round(df_store_real.rating.mean(), 1)  #진정성 리뷰의 평점
        
        #상세페이지 칸에 쓰일 것들
        kakaomap_id = df_store.kakaomap_id[0]   #카카오맵 id
        road_address = df_store.road_address[0] #도로명 주소
        all_rating = round(df_store.rating.mean(), 1)    #전체 리뷰의 평점

        df_store_detail = df_store.iloc[:, 4:].sort_values(
            by=['realreview_prob', 'photo_url', 'review_date'], #정렬 기준: '진정성 리뷰일 확률' ＞ '이미지 링크' ＞ '리뷰 작성일'
            ascending=[False, False, False]
        ).reset_index(drop=True).iloc[:2]

        reviewer_name = reviewer_name = list(df_store_detail.reviewer_name) #리뷰어 네임

        #리뷰 내용
        review_text = []
        for review in df_store_detail.review_text:
            if not review:  #리뷰 내용이 없을 경우
                review_text.append('')
            else:
                review_text.append(review)

        reviewer_rating = df_store_detail.rating #리뷰어 평점
        review_date = df_store_detail.review_date   #리뷰 작성일

        #리뷰 이미지 링크
        photo_url = []
        for url in df_store_detail.photo_url:
            if url:
                urls = url.split(',')   #쉼표로 url 분리
                urls = ['https:' + url for url in urls] #각 url에 https: 붙이기
                urls = urls[:2] #2개만 가져오기
                photo_url.append(urls)
            else:   #이미지 링크가 없을 경우
                photo_url.append('')

        detail_list = [kakaomap_id, road_address, all_rating, real_rating, reviewer_name, review_text, reviewer_rating, review_date, photo_url]
    
    else:
        pie_label_list = []
        bar_rating_list = []
        wordcloud_text = ''
        detail_list = []

    return pie_label_list, bar_rating_list, wordcloud_text, store_name, detail_list