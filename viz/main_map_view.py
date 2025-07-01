import streamlit as st
from viz.restaurants_map_plot import plot_restaurants_on_map, get_map_data
def main_map_view(con):
    # --- Fetch and Plot Map Data ---
    st.subheader("연남동 일반 음식점")
    map_df = get_map_data(con) # Use the 'con' guaranteed to be live

    if not map_df.empty:
        map_figure = plot_restaurants_on_map(map_df)
        if map_figure:
            event = st.plotly_chart(map_figure,
                                    key="main_map",
                                    on_select="rerun",
                                    config={'displayModeBar':True},
                                    selection_mode="points", # 클릭해서 선택할 수 있도록 함
                                    use_container_width=False)
            print(f"plotly main map event received: {event}")
        else:
            st.error("지도를 그릴 수 없습니다.")
    else:
        st.error("지도를 그리기 위한 데이터에 접근할 수 없습니다.")

    # 클릭 사건이 일어남 + 포인트에 대한 정보 있음
    if event.selection and event.selection.points:
        try:
            selected_restaurant_name = event.selection.points[0]["text"]
            st.session_state.selected_restaurant = selected_restaurant_name
            st.session_state.current_page = 'detail_view'
            st.rerun()
        except KeyError:
            st.warning("해당 마커에 대한 데이터를 찾을 수 없습니다.")
            st.write("Full selection event data (check your console):")
            st.json(event.selection.points[0]) # Display the first point's data in the app for inspection
    