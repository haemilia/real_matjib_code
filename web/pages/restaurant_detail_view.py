import streamlit as st
from web.viz.restaurants_map_plot import plot_restaurants_on_map
from web.utils.data import get_map_data

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
            st.write("어쩌구 저쩌구")
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