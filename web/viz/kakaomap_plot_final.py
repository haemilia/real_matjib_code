import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import ast
import pandas as pd
import re
from web.utils.database import download_font_for_wordcloud

#1. duckdb 파일 연결, query문 함수
# def get_kakaomap(conn, click_store): # web.utils.data 로 옮김
#     query = """y
#         SELECT
#             r.store_name, r.road_address,
#             l.predicted_label, l.kakaomap_id, l.rating, l.reviewer_name, l.review_text, l.photo_url, l.processed_cleaned, l.realreview_prob, l.review_date
#         FROM
#             kakaomap_reviews_labelled l
#         JOIN
#             kakaomap_restaurants r
#         ON
#             l.kakaomap_id = r.kakaomap_id
#     """
#     df_kakaomap = conn.execute(query).df()
#     df_kakaomap['predicted_label'] = df_kakaomap['predicted_label'].map({0:'홍보성', 1:'진정성'})
#     df_kakaomap['review_date'] = pd.to_datetime(df_kakaomap['review_date']).dt.strftime('%Y-%m-%d')
    
#     #'스시정인'이라는 음식점으로 test
#     store_name = click_store
#     #print(df_kakaomap)
#     df_store = df_kakaomap.query("store_name == @store_name").reset_index()
#     #print(df_store)
#     store_name = re.sub(r'\(.*?\)', '', store_name)  #괄호와 그 안의 내용 제거

#     ## -------------------------------------------------------------------------------------------------------------

#     if not df_store.empty:
        
#         #파이차트에 쓰일 변수
#         pre_label_name = list(df_store.predicted_label.unique()) #홍보성/진정성
#         pre_label_value = df_store['predicted_label'].value_counts() #라벨링값
#         pie_label_list = [pre_label_name, pre_label_value]

#         #리뷰가 '진정성'이 있는 것들만 필터링
#         df_store_real = df_store.query("predicted_label == '진정성'")

#         if df_store_real.empty: #'진정성' 리뷰가 없을 떄
#             bar_rating_list = []    #막대그래프 변수
#             wordcloud_text = '' #워드클라우드 변수
#             real_rating = ''    #진정성 리뷰 평점 변수
        
#         else:
#             #막대그래프에 쓰일 변수
#             rating_xlabel = sorted(list(df_store_real.rating.unique()))  #x축 라벨
#             rating_ylabel = sorted(df_store_real.rating.value_counts())  #y축 값
#             bar_rating_list = [rating_xlabel, rating_ylabel]

#             #워드클라우드
#             wordcloud_review = df_store_real.processed_cleaned

#             #모든 리뷰의 토큰을 하나의 리스트로 합침
#             all_words = []
#             for review in wordcloud_review:
#                 if isinstance(review, str):
#                     word_list = ast.literal_eval(review)
#                     all_words.extend(word_list)
#                 elif isinstance(review, list):
#                     all_words.extend(review)

#             #하나의 텍스트로 합치기
#             wordcloud_text = ' '.join(all_words)

#             #상세페이지 칸에 쓰일 것
#             real_rating = round(df_store_real.rating.mean(), 1)  #진정성 리뷰의 평점
        
#         #상세페이지 칸에 쓰일 것들
#         kakaomap_id = df_store.kakaomap_id[0]   #카카오맵 id
#         road_address = df_store.road_address[0] #도로명 주소
#         all_rating = round(df_store.rating.mean(), 1)    #전체 리뷰의 평점

#         df_store_detail = df_store.iloc[:, 4:].sort_values(
#             by=['realreview_prob', 'photo_url', 'review_date'], #정렬 기준: '진정성 리뷰일 확률' ＞ '이미지 링크' ＞ '리뷰 작성일'
#             ascending=[False, False, False]
#         ).reset_index(drop=True).iloc[:2]

#         reviewer_name = reviewer_name = list(df_store_detail.reviewer_name) #리뷰어 네임

#         #리뷰 내용
#         review_text = []
#         for review in df_store_detail.review_text:
#             if not review:  #리뷰 내용이 없을 경우
#                 review_text.append('')
#             else:
#                 review_text.append(review)

#         reviewer_rating = df_store_detail.rating #리뷰어 평점
#         review_date = df_store_detail.review_date   #리뷰 작성일

#         #리뷰 이미지 링크
#         photo_url = []
#         for url in df_store_detail.photo_url:
#             if url:
#                 urls = url.split(',')   #쉼표로 url 분리
#                 urls = ['https:' + url for url in urls] #각 url에 https: 붙이기
#                 urls = urls[:2] #2개만 가져오기
#                 photo_url.append(urls)
#             else:   #이미지 링크가 없을 경우
#                 photo_url.append('')

#         detail_list = [kakaomap_id, road_address, all_rating, real_rating, reviewer_name, review_text, reviewer_rating, review_date, photo_url]
    
#     else:
#         pie_label_list = []
#         bar_rating_list = []
#         wordcloud_text = ''
#         detail_list = []

#     return pie_label_list, bar_rating_list, wordcloud_text, store_name, detail_list

#2. 차트 함수
def make_chart(pie_label_list, bar_rating_list, wordcloud_text):
    #파이차트
    pie_fig = go.Figure()
    pie_fig.add_trace(go.Pie(
        labels=pie_label_list[0],
        values=pie_label_list[1],
        marker=dict(colors=['#fee500', "#fbef85"])
    ))
    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    pie_fig.update_layout(
        title=dict(text='<b>홍보성과 진정성 여부</b>', xanchor='center', x=.5, font=dict(size=20)),
        legend=dict(orientation='h', xanchor='center', x=.5, yanchor='bottom', y=-.2)
    )

    #'진정성' 리뷰가 존재하여 막대그래프 변수랑 워드클라우드 변수가 있을 경우
    if bar_rating_list and wordcloud_text:
        #막대그래프
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=bar_rating_list[0],
            y=bar_rating_list[1],
            marker_color="#402424",
        ))
        bar_fig.update_layout(
            title=dict(text='<b>진정성 리뷰 평점 분포도</b>', xanchor='center', x=.5, font=dict(size=20))
        )
        bar_fig.update_xaxes(
            tickmode='array',
            tickvals=[1, 2, 3, 4, 5],
            ticktext=['1', '2', '3', '4', '5'],
            range=[0.5, 5.5]
        )

        #폰트 path
        font_path = download_font_for_wordcloud(font_url = "https://pub-2781d44b6c7d49598d357b65966c4a68.r2.dev/NanumGothic.ttf", # 폰트 저장해둔 R2 공개 버킷
                                                font_filename="NanumGothic.ttf")
        wordcloud = WordCloud(
            font_path=font_path,
            background_color='white',
            random_state=42
        ).generate(wordcloud_text)
        wordcloud_fig = plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

    else:   #없을 경우
        bar_fig = ''
        wordcloud_fig = ''

    return pie_fig, bar_fig, wordcloud_fig 

#3. 리뷰 평점 별 아이콘 함수
def rating_stars(rating_avg):
    #평점값만큼 별 출력(.5~.7은 덜 꽉찬 별 추가, 나머지는 반올림)
    full_stars = int(rating_avg)
    sub = round(rating_avg - full_stars, 2)

    if sub >= 0.8 and full_stars < 5:
        full_stars += 1
        half_star = False
    elif 0.5 <= sub < 0.8:
        half_star =  True
    else:
        half_star = False

    stars = []
    for _ in range(full_stars):
        stars.append('<i class="fa-solid fa-star stars"></i>')
    if half_star and len(stars) < 5:
        stars.append('<i class="fa-solid fa-star-half-stroke stars"></i>')
    for _ in range(5 -len(stars)):
        stars.append('<i class="fa-regular fa-star stars"></i>')

    results = ''.join(stars)

    return results

#4. 상세컬럼에 작성할 내용 함수
def unpack_detail_list(detail_list):
    """
    반환값: (kakaomap_id, road_address, all_rating_avg, real_rating_avg,
           reviewer_name, review_text, reviewer_rating, review_date, img_url_list, url,
           all_rating_avg_stars, real_rating_avg_stars)
    """

    url = None

    if detail_list: 

        kakaomap_id = detail_list[0]    #음식점 카카오맵 id
        road_address = detail_list[1]   #도로명 주소
        all_rating_avg = detail_list[2] #전체 리뷰 평점
        real_rating_avg = detail_list[3]    #'진정성' 리뷰 평점
        reviewer_name = detail_list[4]  #리뷰어 닉네임
        review_text = detail_list[5]    #리뷰 내용
        reviewer_rating = detail_list[6]    #리뷰어가 남긴 평점
        review_date = detail_list[7]    #리뷰 작성일
        img_url_list = detail_list[8]   #리뷰 이미지 url

        url = f'https://place.map.kakao.com/{kakaomap_id}'  #음식점 카카오맵 리뷰 url
        all_rating_avg_stars = rating_stars(all_rating_avg) #전체 리뷰 평점 평균의 별

        # real_rating_avg가 None일 수도 있으니 예외 처리
        real_rating_avg_stars = rating_stars(real_rating_avg) if real_rating_avg else ''
        
    else:
        kakaomap_id = ''
        road_address = ''
        all_rating_avg = ''
        real_rating_avg = ''
        reviewer_name = ''
        review_text = ''
        reviewer_rating = ''
        review_date = ''
        img_url_list = ''
        url = ''
        all_rating_avg_stars = ''
        real_rating_avg_stars = ''

    return (kakaomap_id, road_address, all_rating_avg, real_rating_avg,
            reviewer_name, review_text, reviewer_rating, review_date, img_url_list,
            url, all_rating_avg_stars, real_rating_avg_stars) 

#5. 상세페이지에서 음식점 간략 소개? html이랑 css 함수
def make_store_html_and_css(store_name, url, road_address, all_rating_avg, all_rating_avg_stars, real_rating_avg, real_rating_avg_stars):
    rating_html = f'<div><span id="rating_text"><b>전체 리뷰 평점 {all_rating_avg}</b></span><span>{all_rating_avg_stars}</span></div>'

    #'진정성' 리뷰 평점이 있을 경우
    if real_rating_avg:
        rating_html += f'<div><span id="rating_text"><b>진정성 리뷰 평점 {real_rating_avg}</b></span><span>{real_rating_avg_stars}</span></div>'

    href_url = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"  #별 아이콘 fontawesome url

    store_css = """
        <style>
            #map-link{
                font-size: 25px;
                color: #000;
                font-weight:bold;
                text-decoration: None;
            }
            #pointer{
                color: #fee500;
                margin-left: 10px;
            }
            .container{
                margin-top:5px;
                padding: 10px;
                border: 1px solid black;
                border-radius: 5px;
            }
            .stars{
                color: #fee500;
            }
        </style>
    """

    store_html = f"""
    <div class="header">
        <link rel="stylesheet" href="{href_url}">
        <a href="{url}" id="map-link">{store_name} 상세페이지</a>
        <i class="fa-solid fa-arrow-pointer fa-2xl" id="pointer"></i>
    </div>
    <div class="container">
        <div>{road_address}</div>
        <div>{rating_html}</div>
    </div>
    """

    return store_css, store_html

#6. 상세페이지에서 리뷰 출력 html이랑 css 함수
def make_review_html_and_css(name, stars, date, review):
    review_html = (
        f"<div class='review'>"
        f"<span class='reviewer_name'><b>{name}</b></span>"
        f"<span>{stars}</span>"
        f"<span class='review_date'>{date}</span>"
        f"<div>{review}</div>"
        f"</div>"
    )

    review_css = """
    <style>
        .review{margin-bottom: 10px;}
        .reviewer_name{padding-right:10px;}
        .review_date{display: inline-block; float: right}
        .img_container{display: flex; justify-content: space-around; align-items: center;}
        a:nth-child(1){margin-bottom: 10px;}
    </style>
    """

    return review_html, review_css

#상세페이지 리뷰에서 이미지 출력 함수
def make_img_html(img_urls):
    html = "<div class='img_container'>"

    for url in img_urls[:2]:
        html += f"<a href='{url}' target='_blank'><img src='{url}'></a>"
    html += "</div>"

    return html

# run.py
# import duckdb
# import streamlit as st
# from pathlib import Path
# from test_module import (
#     get_kakaomap,
#     rating_stars,
#     make_chart,
#     unpack_detail_list,
#     make_store_html_and_css,
#     make_review_html_and_css,
#     make_img_html
# )

# #db 연결
# new_db_path = Path("G:\내 드라이브") / "reviews.db"
# conn = duckdb.connect(database=new_db_path, read_only=False)

# pie_label_list, bar_rating_list, wordcloud_text, store_name, detail_list = get_kakaomap(conn)

# #1. detail_list에서 변수 한 번에 분리
# (
#     kakaomap_id, road_address, all_rating_avg, real_rating_avg,
#     reviewer_name, review_text, reviewer_rating, review_date, img_url_list,
#     url, all_rating_avg_stars, real_rating_avg_stars
# ) = unpack_detail_list(detail_list)

# #2. 차트 객체 생성
# pie_fig, bar_fig, wordcloud_fig = make_chart(pie_label_list, bar_rating_list, wordcloud_text)

# #3. 상세페이지에서 음식점 간략 소개 함수
# store_css, store_html = make_store_html_and_css(
#     store_name, url, road_address, all_rating_avg, all_rating_avg_stars,
#     real_rating_avg, real_rating_avg_stars
# )

# #4. Streamlit 출력
# st.set_page_config(page_title=f'{store_name}', layout="wide")
# st.header(f'{store_name} 카카오맵 리뷰')

# chart_col, detail_col = st.columns([.5, .5])    #차트 컬럼/상세페이지 컬럼 분리

# #4-1. 차트 컬럼
# with chart_col:
#     if bar_rating_list and wordcloud_text and real_rating_avg:  #'진정성' 리뷰가 있는 경우
#         col1, col2 = st.columns(2)
#         with col1:
#             st.plotly_chart(pie_fig, use_container_width=True)
#         with col2:
#             st.plotly_chart(bar_fig, use_container_width=True)
#         st.pyplot(wordcloud_fig)
#     else:   #없는 경우
#         st.plotly_chart(pie_fig, use_container_width=True)

# #4-2. 상세페이지 컬럼
# with detail_col:
#     st.markdown(store_css, unsafe_allow_html=True)
#     st.markdown(store_html, unsafe_allow_html=True)
    
#     if reviewer_name:
#         stars = rating_stars(reviewer_rating[0])
#         review_html, review_css = make_review_html_and_css(
#             reviewer_name[0], stars, review_date[0], review_text[0]
#         )
#         st.markdown(review_css, unsafe_allow_html=True)
#         st.markdown(review_html, unsafe_allow_html=True)

#         if img_url_list[0]:
#             st.markdown(make_img_html(img_url_list[0]), unsafe_allow_html=True)
        
#         if real_rating_avg:
#             for i in range(1, len(reviewer_name)):
#                 stars = rating_stars(reviewer_rating[i])
#                 review_html, _ = make_review_html_and_css(
#                     reviewer_name[i], stars, review_date[i], review_text[i]
#                 )
#                 st.markdown(review_html, unsafe_allow_html=True)
#                 if img_url_list[i]:
#                     st.markdown(make_img_html(img_url_list[i]), unsafe_allow_html=True)

# conn.close()