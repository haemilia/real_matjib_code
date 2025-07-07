import pandas as pd
import re
import numpy as np
import streamlit as st
import duckdb
from pathlib import Path

new_db_path = Path("G:\내 드라이브") / "reviews.db"
conn = duckdb.connect(database=new_db_path, read_only=False)

def get_navermap(conn):    #db, 클릭한 스토어명
    query = """
        SELECT
            n.naver_jibun_address, n.naver_store_id,
            m.store_id, m.store_naver_name, m.review_text, m.review_datetime, m.visit_count, m.image_links, m.review_datetime, m.is_advert_prob, m.sentiment, m.confidence,
        FROM
            navermap_reviews m
        JOIN
            naver_restaurants n
        ON
            n.naver_store_id = m.store_id
    """
    df_navermap = conn.execute(query).df()

    df_navermap['sentiment'] = df_navermap['sentiment'].map({'positive':'긍정 리뷰', 'neutral':'중립 리뷰', 'negative':'부정 리뷰'})
    df_navermap['review_datetime'] = pd.to_datetime(df_navermap['review_datetime']).dt.strftime('%Y-%m-%d')

    # click_store = '하타네 연남점'
    click_store = '향미'

    df_store = df_navermap.query("store_naver_name == @click_store").sort_values(
        by=['is_advert_prob', 'review_datetime'], #홍보성 리뷰일 가능성, 작성일자
        ascending=[True, False] #내림차순(진정성일 확률이 높은 리뷰부터), 오름차순(최신순)
    ).reset_index()

    return df_store, click_store

def get_detail(dataframe, store_name):  #query한 데이터프레임, 음식점명
    df_store = dataframe
    naver_store_name = store_name
    
    store_id = df_store.store_id[0]
    store_url = f'https://pcmap.place.naver.com/restaurant/{store_id}/home'

    store_location = df_store.naver_jibun_address[0]

    store_sentiment = df_store['sentiment']
    store_sentiment_cnt = ''

    if not store_sentiment.empty:
        store_sentiment_cnt = store_sentiment.value_counts()

    store_review_cnt = len(df_store['sentiment'])

    store_dict = {
        'name': naver_store_name,
        'url': store_url,
        'location': store_location,
        'sentiment': store_sentiment,
        'sentiment_cnt': store_sentiment_cnt,
        'review_cnt': store_review_cnt
    }

    review_text = df_store['review_text'][:2]
    review_datetime = df_store['review_datetime'][:2]

    img_url = df_store['image_links'][:2]
    review_img_url = []
    for x in img_url:
        if x is None or (isinstance(x, (list, tuple, np.ndarray)) and len(x) == 0): #이미지 링크가 없을 시
            review_img_url.append(None)
        else:
            match = re.search(r"https?://[^\']+", str(x))
            if match:
                review_img_url.append(match.group(0))
            else:
                review_img_url.append(None)

    review_visit_count = df_store['visit_count'][:2]

    review_dict = {
        'review': review_text,
        'datetime': review_datetime,
        'img_url': review_img_url,
        'visit_count': review_visit_count
    }

    return store_dict, review_dict

def get_detail_html_css(store_dict, review_dict):
    href_url = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"

    store_name = store_dict['name']
    store_url = store_dict['url']

    store_name_css = f'<style> #store_name{{font-size: 25px; color: #000; font-weight: bold; text-decoration: None;}} #pointer{{color: #2DB400; margin-left:10px;}} </style>'
    store_name_html = f'<link rel="stylesheet" href="{href_url}"><a href="{store_url}" id=store_name>{store_name} 네이버 리뷰</a><i class="fa-solid fa-arrow-pointer fa-2xl" id="pointer"></i>'

    store_name_structure = {
        'html': store_name_html,
        'css': store_name_css
    }

    store_location = store_dict['location']
    store_sentiment_cnt = store_dict['sentiment_cnt']
    store_review_cnt = store_dict['review_cnt']

    store_box_css = f'<style> #store_box{{margin-top:5px; padding:10px; border: 1px solid black; border-radius: 5px}} span{{margin-right: 10px;}} </style>'
    store_box_html = f"<div id=store_box><div>{store_location}</div>"
    if (store_sentiment_cnt is not None and not store_sentiment_cnt.empty):
            store_box_html += f"<div id=sentiment_box><span>전체 리뷰: {store_review_cnt}개</span>"
            for sentiment, count in store_sentiment_cnt.items():
                store_box_html += f"<span>{sentiment}: {count}개</span>"
            store_box_html += f"</div>"
    else:
        store_box_html += f"</div>"

    store_box_structure = {
        'html': store_box_html,
        'css': store_box_css
    }

    reviewer_cnt = 0
    review_sentiment = store_dict['sentiment']
    review_datetime = review_dict['datetime']
    review_visit_count = review_dict['visit_count']
    review_text = review_dict['review']
    review_img_url = review_dict['img_url']

    review_css = """
        <style>
            .container{margin: 5px 0 30px;}
            .more_box{display: flex; justify-content: space-between;}
            #visit_count{padding: 0 10px;}
            #img_box{margin-top:5px}
        </style>
    """

    review_html = ""
    n = len(review_text)
    for i in range(n):
        review_html += (
            f"<div class='container'><div class=more_box><div class=left_box><b>리뷰어 {reviewer_cnt + 1}</b></div>"
        )
        reviewer_cnt += 1

        if pd.notnull(review_sentiment[i]):
            review_html += f"<div class=right_box><span>{review_sentiment[i]}</span><span id=visit_count>{review_visit_count[i]}번 방문</span><span id=datetime>{review_datetime[i]}</span></div></div><div>{review_text[i]}</div>"
        else:
            review_html += f"<div class=right_box><span id=visit_count>{review_visit_count[i]}번 방문</span><span id=datetime>{review_datetime[i]}</span></div></div><div>{review_text[i]}</div>"

        if review_img_url[i] not in [None, '', np.nan] and pd.notnull(review_img_url[i]):
            review_html += f"<a href='{review_img_url[i]}' target='_blank'><img src='{review_img_url[i]}' id=img_box></a></div>"
        else:
            review_html  += ''  # 아무것도 출력하지 않음
    
    review_structure = {
        'html': review_html,
        'css': review_css
    }

    return store_name_structure, store_box_structure, review_structure


# streamlit design
df_store, click_store = get_navermap(conn)
store_dict, review_dict = get_detail(df_store, click_store)
store_name_structure, store_box_structure, review_structure = get_detail_html_css(store_dict, review_dict)
st.set_page_config(page_title=f'{click_store}', layout="wide")

plot_col, detail_col = st.columns([.7, .3])

with detail_col:
    store_name_html = store_name_structure.get('html')
    store_name_css = store_name_structure.get('css')
    store_box_html = store_box_structure.get('html')
    store_box_css = store_box_structure.get('css')
    review_html = review_structure.get('html')
    review_css = review_structure.get('css')

    st.markdown(store_name_html, unsafe_allow_html=True)
    st.markdown(store_name_css, unsafe_allow_html=True)
    st.markdown(store_box_html, unsafe_allow_html=True)
    st.markdown(store_box_css, unsafe_allow_html=True)
    
    st.markdown(review_html, unsafe_allow_html=True)
    st.markdown(review_css, unsafe_allow_html=True)

conn.close()

# print('db 종료')


