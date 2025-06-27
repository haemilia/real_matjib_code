import plotly.graph_objects as go
import streamlit as st

def plot_restaurants_on_map(
        #음식점 위도/경도, 음식점명, 타이틀
        restaurants_lat, restaurants_lon, store_name, title='연남동 일반 음식점 지도 시각화',
        #연남동 위도/경도
        yeonnam_lat=37.5628, yeonnam_lon=126.9222, zoom=15
):
    fig = go.Figure()   #fig 객체 생성
    fig.add_trace(
        go.Scattermap(
            lat=restaurants_lat,    #음식점 위도
            lon=restaurants_lon,    #음식점 경도
            mode='markers',
            #마커 스타일 지정
            marker=go.scattermap.Marker(
                size=5,
                color='red',
                opacity=0.4,
                symbol='circle'
            ),
            text=store_name,    #음식점명
            hoverinfo='text'    #마커에 마우스 hover시, 음식점명 출력
        )
    )
    #그래프의 레이아웃과 시각적 속성 설정
    fig.update_layout(
        title=dict(text=title), #그래프 제목
        autosize=True,  #창 크기에 따라 자동으로 그래프 크기 조절
        hovermode='closest',    #마우스를 가장 가까운 점에만 반응하도록 설정
        showlegend=False,   #범례 표시 여부
        map=dict(
            bearing=0,  #지도 회전 각도
            #지도 중심 좌표(연남동)
            center=dict(
                lat=yeonnam_lat,
                lon=yeonnam_lon
            ),
            zoom=zoom,
            style='outdoors'    #지도 배경 스타일
        )
    )
    return fig   #plotly.fig 객체 리턴

