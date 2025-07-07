import streamlit as st
from web.viz.restaurants_map_plot import plot_restaurants_on_map
from web.utils.data import get_map_data, get_kakaomap_data
from web.viz.kakaomap_plot_final import (
    rating_stars,
    make_chart,
    unpack_detail_list,
    make_store_html_and_css,
    make_review_html_and_css,
    make_img_html
)
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
        st.write(f"### 선택된 음식점: {st.session_state.selected_restaurant}")
        map_df = get_map_data()
        detail_map_fig = plot_restaurants_on_map(map_df,
                                                 active_restaurant=st.session_state.selected_restaurant)
        st.plotly_chart(detail_map_fig,
                        key="detail_map",
                        use_container_width=False,
                        config={'displayModeBar': False})
    with col_tabs:
        st.write(f"#### {st.session_state.selected_restaurant}")

        tab_overall, tab_kakao, tab_naver, tab_insta = st.tabs(["통합",
                                                                "카카오맵",
                                                                "네이버지도",
                                                                "인스타그램"])
        with tab_overall:
            st.header(f"{st.session_state.selected_restaurant}에 대한 진품명품의 총평은?")
            st.write("어쩌구 저쩌구")
        with tab_kakao:
            st.header("""카카오맵 (Kakao Map)""")
            ### ---------------------------------------------------------------------------------------------
            click_store = st.session_state.selected_restaurant
            pie_label_list, bar_rating_list, wordcloud_text, store_name, detail_list = get_kakaomap_data(click_store)

            # 모든 값이 비어있거나 없는 경우 '없음' 출력
            if not pie_label_list and not bar_rating_list and not wordcloud_text and not store_name and not detail_list:
                st.write('없음')

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
                            st.plotly_chart(pie_fig, use_container_width=True)
                        with col2:
                            st.plotly_chart(bar_fig, use_container_width=True)
                        st.pyplot(wordcloud_fig)
                    else:   #없는 경우
                        st.plotly_chart(pie_fig, use_container_width=True)

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
            st.write("어쩌구 저쩌구")
        with tab_insta:
            st.header("""인스타그램 (Instagram)""")
            st.write("어쩌구 저쩌구")
    st.markdown("---")
    if st.button("⬅️ 지도로 다시 돌아가기", key="back_to_main_map_button_bottom"):
        st.session_state.current_page = "main_map" #메인 지도 페이지로 다시 돌아가기
        st.session_state.selected_restaurant = None
        st.rerun() 