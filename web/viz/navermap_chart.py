## navermap 페이지 chart 그리기 ##

import streamlit as st
import pandas as pd
import duckdb
import plotly.graph_objects as go
from pathlib import Path
from difflib import get_close_matches

def navermap_make_chart(call_store_name):
    """
        navermap_reviews 테이블 컬럼목록 :
            'review_id', 'review_text', 'store_id', 'store_naver_name', 'category',
            'image_links', 'video_thumbnail_links', 'store_reply', 'num_of_media',
            'visit_count', 'author_total_reviews', 'author_total_images',
            'reactions_fun', 'reactions_helpful', 'reactions_wannago',
            'reactions_cool', 'review_year', 'rating', 'review_datetime',
            'visit_keywords', 'purchase_item', 'keyword_tags_hangul',
            'is_advert_label', 'is_advert_prob', 'sentiment', 'confidence'
        call_store_name : 호출할 가게 이름
        return : 파이그래프 fig객체, 바그래프 fig객체
    """
    
    query = """
        SELECT
            review_text, store_id, store_naver_name, purchase_item, is_advert_label
        FROM
            navermap_reviews
    """
    ## DB서 DF 가져오기(로컬 테스트용)
    # db_path = Path("G:\내 드라이브") / "reviews.db"
    # conn = duckdb.connect(db_path, read_only=True)
    # df_navermap = conn.execute(query).df()

    ## DB서 DF 가져오기
    df_navermap = execute_cached_query_to_df(query)

    df_navermap['is_advert_label'] = df_navermap['is_advert_label'].map({1:'홍보성', 0:'진정성'})

    # call_store_name = "순대일번지"    # 테스트용 가게 이름
    # 가장 유사한 가게명 검색 (호출명과 정확히 일치하지 않을 경우 고려)
    matched_store_name = None
    df_test = pd.DataFrame()

    # if df_navermap:
    store_list = df_navermap['store_naver_name'].dropna().unique().tolist()
    closest_matches = get_close_matches(call_store_name, store_list, n=1, cutoff=0.3)

    if closest_matches:
        matched_store_name = closest_matches[0]
        matched_store_ids = df_navermap[df_navermap['store_naver_name'] == matched_store_name]['store_id'].unique()
        df_test = df_navermap[
            (df_navermap['store_naver_name'] == matched_store_name) &
            (df_navermap['store_id'].isin(matched_store_ids))
        ].copy()

    else:
        st.warning("❗ 유사한 가게를 찾을 수 없습니다.")
    
    # 단순 가게명 일치검색
    # df_test = df_navermap.query("store_name == call_store_name")

    #파이차트에 쓰일 변수
    pre_label_name = list(df_test.is_advert_label.unique()) #홍보/진정성
    pre_label_value = df_test['is_advert_label'].value_counts() #라벨링값
    pie_label_list = [pre_label_name, pre_label_value]

    # 파이차트(라벨링)
    labeling_pie_fig = go.Figure()  #그래프 객체 생성
    labeling_pie_fig.add_trace(
        go.Pie(
            labels=pie_label_list[0],   #홍보/진정성
            values=pie_label_list[1],   #라벨링별 예측값 비율
            #차트 색상
            marker=dict(
                colors=["#2DB400", "#8CFB67"]
            )
        )
    )
    labeling_pie_fig.update_traces(
        textposition='inside',  #차트 안에 범례 텍스트 생성
        textinfo='percent+label'
    )
    labeling_pie_fig.update_layout(
        title=dict(
            text='<b>홍보성 vs 진정성 여부</b>',    #그래프 title
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

    #리뷰가 '진정성'이 있는 것들만 필터링
    df_test_real = df_test.query("is_advert_label == '진정성'")

    #막대그래프에 쓰일 변수
    item_counts = df_test_real['purchase_item'].value_counts()
    top_items = item_counts.head(5)

    # 막대그래프 생성
    bar_trace = go.Bar(
        x=top_items.index,  # x축: 아이템 이름
        y=top_items.values, # y축: 아이템 개수
        marker_color="#2DB400" # 막대 색상(네이버 로고색)
    )
    bar_purchase_item_fig = go.Figure(data=[bar_trace])
    
    # 레이아웃 설정 (타이틀, 축 레이블, 티커 색상 등)
    bar_purchase_item_fig.update_layout(
        title=dict(text='<b>가장 많이 선택한 메뉴 TOP.5</b>',
                    xanchor='center', x=.6, font=dict(size=20)),
        xaxis_title={
            'text': '<b>선택 메뉴명</b>',
            'font': {'size': 14, 'color': 'black'} # X축 레이블 색상
        },
        yaxis_title={
            'text': '<b>선택 개수</b>',
            'font': {'size': 14, 'color': 'black'} # Y축 레이블 색상
        },
        xaxis=dict(tickfont=dict(color='black')), # X축 티커 색상
        yaxis=dict(tickfont=dict(color='black')), # Y축 티커 색상
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    
    conn.close()

    return labeling_pie_fig, bar_purchase_item_fig


## code for running test
# call_store_name = "순대일번지"  # 테스트용(기본값) 가게 이름

# if call_store_name:
#     labeling_pie_fig, bar_purchase_item_fig = navermap_make_chart(call_store_name)
        
#     if labeling_pie_fig and bar_purchase_item_fig:
#         st.set_page_config(layout='wide')   #streamlit 화면 넓게
#         st.header(f'{call_store_name} 네이버맵 리뷰')

#         # 파이그래프(좌측), 바그래프(우측) 컬럼 생성
#         plot_col, detail_col = st.columns([.7, .3])  #7:3 비율로 분할

#         #그래프 컬럼의 subplots
#         with plot_col:
#             pie_col, bar_col = st.columns(2)

#             # 파이차트
#             with pie_col:
#                 st.plotly_chart(labeling_pie_fig, use_container_width=True)

#             # 바그래프
#             with bar_col:
#                 st.plotly_chart(bar_purchase_item_fig, use_container_width=True)

#         # 상세 컬럼
#         with detail_col:
#             st.write("(상세정보 노출 위치)")
               
#     else:
#         st.error('could not find graphs')
