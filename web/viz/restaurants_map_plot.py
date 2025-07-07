#%%
import plotly.graph_objects as go

    
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

