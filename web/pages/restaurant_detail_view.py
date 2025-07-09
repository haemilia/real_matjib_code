import streamlit as st
from web.viz.restaurants_map_plot import plot_restaurants_on_map
from web.utils.data import get_map_data, get_kakaomap_data, get_instagram_data, get_naver_noun_data, get_naver_detail_data, get_naver_chart_data
from web.viz.kakaomap_plot_final import (
    rating_stars,
    make_chart,
    unpack_detail_list,
    make_store_html_and_css,
    make_review_html_and_css,
    make_img_html
)
from web.viz.insta_plot import plot_instagram
from web.viz.navermap_noungraph import visualize_restaurant_nlp_insights, create_text_placeholder_chart
from web.viz.naver_detail import get_detail, get_detail_html_css
from web.viz.navermap_chart import navermap_draw_chart

def restaurant_detail_view():
    st.write(f"# {st.session_state.selected_restaurant}")
    st.markdown("---")
    # Add a prominent "Back" button at the top of the detail page
    if st.button("⬅️ 지도로 다시 돌아가기", key="back_to_main_map_button_top"):
        st.session_state.current_page = "main_map" #메인 지도 페이지로 다시 돌아가기
        st.session_state.selected_restaurant = None
        st.rerun() 
    col_map, col_tabs = st.columns([1, 4]) # 지도가 1/5, 상세 페이지가 4/5 차지
    with col_map:
        #st.write(f"### 선택된 음식점: {st.session_state.selected_restaurant}")
        map_df = get_map_data()
        detail_map_fig = plot_restaurants_on_map(map_df,
                                                 active_restaurant=st.session_state.selected_restaurant)
        st.plotly_chart(detail_map_fig,
                        key="detail_map",
                        use_container_width=False,
                        config={'displayModeBar': False})
    with col_tabs:
        #st.write(f"#### {st.session_state.selected_restaurant}")

        tab_overall, tab_kakao, tab_naver, tab_insta = st.tabs(["통합",
                                                                "카카오맵",
                                                                "네이버지도",
                                                                "인스타그램"])
        with tab_overall:
            st.subheader('플랫폼별로 "진짜" 리뷰와 "가짜" 리뷰의 비율을 시각화한 파이차트')
            kakao_col, naver_col, insta_col = st.columns(3)
            with kakao_col:
                st.subheader("카카오맵 ")
                click_store = st.session_state.selected_restaurant
                pie_label_list, bar_rating_list, wordcloud_text, _, _ = get_kakaomap_data(click_store)
                if len(pie_label_list) == 0:
                    st.info("해당 음식점에 대한 카카오맵 리뷰 데이터가 없습니다.")
                else:
                    pie_fig, _, _ = make_chart(pie_label_list, bar_rating_list, wordcloud_text)
                    st.plotly_chart(pie_fig, use_container_width=True, key="overall_kakao_pie")
            with naver_col:
                st.subheader("네이버지도")
                click_store = st.session_state.selected_restaurant
                df_store_chart = get_naver_chart_data(click_store)
                if df_store_chart is None or df_store_chart.empty:
                    st.info("해당 음식점에 대한 네이버 리뷰 데이터가 없습니다.")
                else:
                    naver_labeling_pie_fig, _ = navermap_draw_chart(df_store_chart)
                    st.plotly_chart(naver_labeling_pie_fig, use_container_width=True, key="overall_naver_pie")
            with insta_col:
                st.subheader("인스타그램")
                click_store = st.session_state.selected_restaurant
                pie_label_list, reviewtxt_for_wordcloud, commentstxt_for_wordcloud = get_instagram_data(click_store)
                if len(pie_label_list) == 0:
                    st.info("해당 음식점에 대한 인스타그램 리뷰 데이터가 없습니다.")
                else:
                    labeling_pie_fig, _, _ = plot_instagram(pie_label_list, 
                                                            reviewtxt_for_wordcloud, 
                                                            commentstxt_for_wordcloud)
                    st.plotly_chart(labeling_pie_fig, use_container_width=True, key="overall_insta_pie")
        with tab_kakao:
            st.header("""카카오맵 (Kakao Map)""")
            ### ---------------------------------------------------------------------------------------------
            click_store = st.session_state.selected_restaurant
            pie_label_list, bar_rating_list, wordcloud_text, store_name, detail_list = get_kakaomap_data(click_store)
            print("bar_rating_list:", bar_rating_list)

            # 값이 비어있거나 없는 경우 '없음' 출력
            if len(pie_label_list) == 0:
                st.info('해당 음식점에 대한 카카오맵 리뷰 데이터가 없습니다.')

            else:
                #1. detail_list에서 변수 한 번에 분리
                (
                    kakaomap_id, road_address, all_rating_avg, real_rating_avg,
                    reviewer_name, review_text, reviewer_rating, review_date, img_url_list,
                    url, all_rating_avg_stars, real_rating_avg_stars
                ) = unpack_detail_list(detail_list)

                #2. 차트 객체 생성
                pie_fig, bar_fig, wordcloud_fig = make_chart(pie_label_list, bar_rating_list, wordcloud_text)

                #3. 상세페이지에서 음식점 간략 소개 함수
                store_css, store_html = make_store_html_and_css(
                    store_name, url, road_address, all_rating_avg, all_rating_avg_stars,
                    real_rating_avg, real_rating_avg_stars
                )

                chart_col, detail_col = st.columns([.5, .5])    #차트 컬럼/상세페이지 컬럼 분리

                #4-1. 차트 컬럼
                with chart_col:
                    if bar_rating_list and wordcloud_text and real_rating_avg:  #'진정성' 리뷰가 있는 경우
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(pie_fig, use_container_width=True, key="kakao_pie")
                        with col2:
                            st.plotly_chart(bar_fig, use_container_width=True, key="kakao_bar")
                        st.pyplot(wordcloud_fig)
                    else:   #없는 경우
                        st.plotly_chart(pie_fig, use_container_width=True, key="kakao_pie")

                # #4-2. 상세페이지 컬럼
                with detail_col:
                    st.markdown(store_css, unsafe_allow_html=True)
                    st.markdown(store_html, unsafe_allow_html=True)
                    
                    if reviewer_name:
                        stars = rating_stars(reviewer_rating[0])
                        review_html, review_css = make_review_html_and_css(
                            reviewer_name[0], stars, review_date[0], review_text[0]
                        )
                        st.markdown(review_css, unsafe_allow_html=True)
                        st.markdown(review_html, unsafe_allow_html=True)

                        if img_url_list[0]:
                            st.markdown(make_img_html(img_url_list[0]), unsafe_allow_html=True)
                        
                        if real_rating_avg:
                            for i in range(1, len(reviewer_name)):
                                stars = rating_stars(reviewer_rating[i])
                                review_html, _ = make_review_html_and_css(
                                    reviewer_name[i], stars, review_date[i], review_text[i]
                                )
                                st.markdown(review_html, unsafe_allow_html=True)
                                if img_url_list[i]:
                                    st.markdown(make_img_html(img_url_list[i]), unsafe_allow_html=True)

        ### -----------------------------------------------------------------------------------------------
        with tab_naver:
            st.header("""네이버지도 (Naver Map)""")
            click_store = st.session_state.selected_restaurant

            chart_col, detail_col = st.columns([.7, .3])
            with chart_col:
                with st.container(border=True):
                    df_store_chart = get_naver_chart_data(click_store)
                    if df_store_chart is None or df_store_chart.empty:
                        st.info("해당 음식점에 대한 네이버 리뷰 요약 데이터가 없습니다.")
                    else:
                        naver_labeling_pie_fig, naver_bar_purchase_item_fig = navermap_draw_chart(df_store_chart)
                        if naver_labeling_pie_fig and naver_bar_purchase_item_fig:
                            pie_col, bar_col = st.columns(2)

                            with pie_col:
                                st.plotly_chart(naver_labeling_pie_fig, use_container_width=True, key="naver_pie")
                            with bar_col:
                                st.plotly_chart(naver_bar_purchase_item_fig, use_container_width=True, key="naver_bar")
                    

                st.markdown("---") # Visual separator between the two containers

                noun_summary_row = get_naver_noun_data(click_store)
                if noun_summary_row is None or noun_summary_row.empty:
                    st.info("해당 음식점에 대한 명사 분석이 없습니다.")
                else:
                    with st.container(border = True):
                        visualize_restaurant_nlp_insights(noun_summary_row)
            with detail_col:
                # with st.container(border=True):
                df_store, click_store = get_naver_detail_data(click_store)
                if df_store is None or df_store.empty:
                    st.info("해당 음식점에 대한 네이버 리뷰 예시 정보가 없습니다.")
                else:
                    store_dict, review_dict = get_detail(df_store, click_store)
                    store_name_structure, store_box_structure, review_structure = get_detail_html_css(store_dict, review_dict)

                    store_name_html = store_name_structure.get('html')
                    store_name_css = store_name_structure.get('css')
                    store_box_html = store_box_structure.get('html')
                    store_box_css = store_box_structure.get('css')
                    review_html = review_structure.get('html')
                    review_css = review_structure.get('css')

                    st.markdown(store_name_html, unsafe_allow_html=True)
                    st.markdown(store_name_css, unsafe_allow_html=True)
                    st.markdown(store_box_css, unsafe_allow_html=True)
                    st.markdown(store_box_html, unsafe_allow_html=True)
                    
                    st.markdown(review_html, unsafe_allow_html=True)
                    st.markdown(review_css, unsafe_allow_html=True)

        with tab_insta:
            st.header("""인스타그램 (Instagram)""")
            click_store = st.session_state.selected_restaurant
            pie_label_list, reviewtxt_for_wordcloud, commentstxt_for_wordcloud = get_instagram_data(click_store)
            if len(pie_label_list) == 0 or len(reviewtxt_for_wordcloud) == 0:
                st.write("해당 음식점에 대한 인스타그램 리뷰 데이터가 없습니다.")
            else:
                labeling_pie_fig, review_wordcloud_plot_fig, _ = plot_instagram(pie_label_list, 
                                                                                reviewtxt_for_wordcloud, 
                                                                                commentstxt_for_wordcloud)
                if labeling_pie_fig and review_wordcloud_plot_fig:
                    #st.header(f'{click_store} 음식점 인스타그램 리뷰')

                    # 그래프(좌측), 워드클라우드(우측) 컬럼 생성
                    plot_col, w_cloud = st.columns([.3, .7])  #3:7 비율로 분할

                    # 파이차트
                    with plot_col:
                        st.plotly_chart(labeling_pie_fig, use_container_width=True, key="insta_pie")
                    
                    # 리뷰 워드클라우드
                    with w_cloud:
                        st.subheader("리뷰 워드클라우드")
                        st.pyplot(review_wordcloud_plot_fig)
                
                else:
                    st.error('could not find graphs')


    st.markdown("---")
    if st.button("⬅️ 지도로 다시 돌아가기", key="back_to_main_map_button_bottom"):
        st.session_state.current_page = "main_map" #메인 지도 페이지로 다시 돌아가기
        st.session_state.selected_restaurant = None
        st.rerun() 