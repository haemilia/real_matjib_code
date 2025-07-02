#module python file code
import plotly.graph_objects as go
import ast
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def get_kakaomap(conn):
    query = """
        SELECT
            r.store_name,
            l.predicted_label, l.rating, l.processed_cleaned, l.kakaomap_id
        FROM
            kakaomap_reviews_labelled l
        JOIN
            kakaomap_restaurants r
        ON
            l.kakaomap_id = r.kakaomap_id
    """
    df_kakaomap = conn.execute(query).df()
    df_kakaomap['predicted_label'] = df_kakaomap['predicted_label'].map({0:'홍보성', 1:'진정성'})

    #'스시정인'이라는 음식점으로 test
    store_name = '스시정인'
    df_test = df_kakaomap.query("store_name == @store_name")
    
    #파이차트에 쓰일 변수
    pre_label_name = list(df_test.predicted_label.unique()) #홍보성/진정성
    pre_label_value = df_test['predicted_label'].value_counts() #라벨링값
    pie_label_list = [pre_label_name, pre_label_value]

    #리뷰가 '진정성'이 있는 것들만 필터링
    df_test_real = df_test.query("predicted_label == '진정성'")
    
    #카카오맵 id
    kakaomap_id = df_test.kakaomap_id.unique().item()

    #막대그래프에 쓰일 변수
    rating_xlabel = sorted(list(df_test_real.rating.unique()))  #x축 라벨
    rating_ylabel = sorted(df_test_real.rating.value_counts())  #y축 값
    bar_rating_list = [rating_xlabel, rating_ylabel]

    wordcloud_review = df_test_real.processed_cleaned

    #모든 리뷰의 토큰을 하나의 리스트로 합침
    all_words = []
    for review in wordcloud_review:
        if isinstance(review, str):
            word_list = ast.literal_eval(review)
            all_words.extend(word_list)
        elif isinstance(review, list):
            all_words.extend(review)

    #하나의 텍스트로 합치기
    wordcloud_text = ' '.join(all_words)

    #상세페이지 칸에 쓰일 것들
    all_rating = round(df_test.rating.mean(), 1)    #전체 리뷰의 평점
    real_rating = round(df_test_real.rating.mean(), 1)  #진정성 리뷰의 평점
    detail_rating_list = [all_rating, real_rating]

    return pie_label_list, bar_rating_list, wordcloud_text, kakaomap_id, store_name, detail_rating_list

#pie_label_list, bar_rating_list, wordcloud_text, kakaomap_id, store_name, detail_rating_list = get_kakaomap(conn)

#평점만큼 별 개수 출력하는 함수
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

#차트 생성 함수
def plot_kakaomap(
        #파이차트에 쓰일 범례, 값, #히스토그램에 쓰일 평점값, 상세페이지 url에 넣을 id, 음식점명, 별점 평균
        pie_label_list, bar_rating_list, wordcloud_text, kakaomap_id, store_name, detail_rating_list
):
    #st.set_page_config(layout="wide")   #화면 넓게

    #카카오맵 타이틀..?
    #st.header('@@음식점 카카오맵 리뷰')

    #파이차트(라벨링)
    labeling_pie_fig = go.Figure()  #그래프 객체 생성
    labeling_pie_fig.add_trace(
        go.Pie(
            labels=pie_label_list[0],  #홍보성/진정성
            values=pie_label_list[1], #라벨링별 예측값 비율
            #차트 색상
            marker=dict(
                colors=['#fee500', "#fbef85"]
            )
        )
    )
    labeling_pie_fig.update_traces(
        textposition='inside',  #차트 안에 범례 텍스트 생성
        textinfo='percent+label'
    )
    labeling_pie_fig.update_layout(
        title=dict(
            text='<b>홍보성과 진정성 여부</b>',    #그래프 title
            #그래프 위치
            xanchor='center',
            x=.5,
            font=dict(size=20)  #폰트 사이즈
        ),
        #범례 위치
        legend=dict(
            orientation='h',
            xanchor='center',
            x=.5,
            yanchor='bottom',
            y=-.2
        )
    )

    #막대그래프(음식점 리뷰 평점)
    rating_bar_fig = go.Figure()   #그래프 객체 생성
    rating_bar_fig.add_trace(
        go.Bar(
            x=bar_rating_list[0],
            y=bar_rating_list[1],
            marker_color="#402424",  #그래프 색상
        )
    )
    rating_bar_fig.update_layout(
        title=dict(
            text='<b>진정성 리뷰 평점 분포도</b>', #그래프 타이틀
            #그래프 위치
            xanchor='center',
            x=.5,
            font=dict(size=20)  #폰트 사이즈
        )
    )
    #x축 설정
    rating_bar_fig.update_xaxes(
        tickmode='array',
        tickvals=[1, 2, 3, 4, 5],
        ticktext=['1', '2', '3', '4', '5'],
        range=[0.5, 5.5]
    )
    rating_bar_fig.update_yaxes(dtick=1)    #y축 눈금 간격 1로 설정

    #워드클라우드
    wordcloud = WordCloud(
        font_path =r"C:\Users\QQQ\AppData\Local\Microsoft\Windows\Fonts\NanumGothic.ttf",
        background_color='white',   #default: black
        random_state=42 #단어 위치 고정
    ).generate(wordcloud_text)
    wordcloud_plot_fig = plt.figure()  #객체 생성
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.rc('font', family='Malgun Gothic')  # 맑은 고딕
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스(-) 기호 깨짐 방지
    plt.title('진정성 리뷰 워드클라우드', fontdict={'fontweight':'bold'}, fontsize=8) #그래프 제목, 볼드체
    plt.axis('off')
    plt.show()

    #상세페이지
    restaurant_name = store_name
    all_rating_avg = detail_rating_list[0]  #전체 리뷰 평점 평균
    real_rating_avg = detail_rating_list[1] #진정성 리뷰 평점 평균
    url = f'https://place.map.kakao.com/{kakaomap_id}'

    detail_all_list = [restaurant_name, all_rating_avg, real_rating_avg, url]

    return labeling_pie_fig, rating_bar_fig, wordcloud_plot_fig, detail_all_list

# run python file code
# import streamlit as st
# from pathlib import Path
# import duckdb
# from test_kakaomap_module import get_kakaomap, plot_kakaomap, rating_stars

# db_path = Path("G:\내 드라이브") / "reviews.db"
# conn = duckdb.connect(db_path)

# if conn:
#     pre_label_list, bar_rating_list, wordcloud_text, kakapmap_id, store_name, detail_rating_list = get_kakaomap(conn)
#     labeling_pie_fig, rating_hist_fig, wordcloud_plot_fig, detail_all_list = plot_kakaomap(pre_label_list, bar_rating_list, wordcloud_text, kakapmap_id, store_name, detail_rating_list)
    
#     if labeling_pie_fig and rating_hist_fig and wordcloud_plot_fig:
#         st.set_page_config(layout='wide')   #streamlit 화면 넓게
#         store_name = detail_all_list[0]
#         st.header(f'{store_name} 음식점 카카오맵 리뷰')

#         #그래프(좌측), 상세(우측) 컬럼 생성
#         plot_col, detail_col = st.columns([.7, .3])  #7:3 비율로 분할

#         #그래프 컬럼의 subplots
#         with plot_col:
#             pie_col, hist_col = st.columns(2)

#             #파이차트
#             with pie_col:
#                 st.plotly_chart(labeling_pie_fig, use_container_width=True)

#             #히스토그램
#             with hist_col:
#                 st.plotly_chart(rating_hist_fig, use_container_width=True)

#             #워드클라우드
#             st.pyplot(wordcloud_plot_fig)

#         #상세 컬럼
#         with detail_col:
#             all_rating_avg = detail_all_list[1]  #전체 리뷰 평점 평균
#             real_rating_avg = detail_all_list[2] #진정성 리뷰 평점 평균

#             css = """
#             <style>
#                 #map-link{
#                     font-size: 25px;
#                     color: #000;
#                     font-weight:bold;
#                     text-decoration: None;
#                 }
#                 #pointer{
#                     color: #fee500;
#                     margin-left: 10px;
#                 }
#                 .stars{
#                     color: #fee500;
#                 }
#             </style>
#             """
#             st.markdown(css, unsafe_allow_html=True)
#             url = detail_all_list[3]
#             all_rating_avg_stars = rating_stars(all_rating_avg) #전체 리뷰 평점 평균의 별
#             real_rating_avg_stars = rating_stars(real_rating_avg) #진정성 리뷰 평점 평균의 별

#             html = f"""
#                 <div class="header">
#                     <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
#                     <a href="{url}" id="map-link">{store_name} 상세페이지</a>
#                     <i class="fa-solid fa-arrow-pointer fa-2xl" id="pointer"></i>
#                 </div>
#                 <div class="container">
#                     <div class='all_stars_container'>
#                         <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
#                         <span id="rating_text">전체 평점 {all_rating_avg}</span>
#                         <span>{all_rating_avg_stars}</div>
#                     </div>
#                     <div class='real_stars_container'>
#                         <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
#                         <span id="rating_text">진정성 리뷰 평점 {real_rating_avg}</span>
#                         <span>{real_rating_avg_stars}</div>
#                     </div>
#                 </div>
#             """
#             st.markdown(html, unsafe_allow_html=True)    

#     else:
#         st.error('could not find graphs')

# conn.close()