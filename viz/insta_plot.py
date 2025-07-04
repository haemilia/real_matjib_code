import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import duckdb
import plotly.graph_objects as go
from pathlib import Path
from difflib import get_close_matches
from wordcloud import WordCloud
from konlpy.tag import Okt

def get_instagram(conn, call_store_name):
    """
        instagram_restaurants 테이블 컬럼목록 :
            store_name, search_name, model_input_review, label, reviewer_id, review, tags, comments, review_date        
        conn : DB연결
        call_store_name : 호출할 가게 이름
        return : 파이차트용 변수, 워드클라우드용 텍스트 모음(리뷰, 코멘트)
    """
    
    query = """
        SELECT
            store_name, search_name, label, review, tags, comments
        FROM
            instagram_restaurants
    """
    df_instagram = conn.execute(query).df()

    # call_store_name = "순대일번지"    # 테스트용 가게 이름
    # 가장 유사한 가게명 검색 (호출명과 정확히 일치하지 않을 경우 고려)
    matched_store = None
    df_test = pd.DataFrame()

    if call_store_name:
        store_list = df_instagram['search_name'].dropna().unique().tolist()
        closest_matches = get_close_matches(call_store_name, store_list, n=1, cutoff=0.3)

        if closest_matches:
            matched_store = closest_matches[0]
            df_test = df_instagram[df_instagram['search_name'] == matched_store]
            df_test['label'] = df_test['label'].replace('일반', '진정성')

        else:
            st.warning("❗ 유사한 가게를 찾을 수 없습니다.")
    
    # 단순 가게명 일치검색
    # df_test = df_instagram.query("store_name == call_store_name")

    #파이차트에 쓰일 변수
    pre_label_name = list(df_test.label.unique()) #홍보/진정성
    pre_label_value = df_test['label'].value_counts() #라벨링값
    pie_label_list = [pre_label_name, pre_label_value]

    # 리뷰 중 '진정성'이 있는 것들만 필터링 (이 코드에서 사용 안함)
    # df_test_real = df_test.query("label == '진정성'")        

    ## 리뷰 및 코멘트 토큰화(워드클라우드용)
    okt = Okt()
    stopwords = [
        '은', '는', '이', '가', '을', '를', '과', '와', '도', '만', '으로', '로', '적', '인', '이다', '이고', '이며', '이니',
        '수', '개', '분', '등', '고', '게', '듯', '음', '안', '것', '때', '곳', '분들', '요', '에서', '하다', '되다',
        '데', '그냥', '네', '응', '오', '아', '그', '저', '저런', '그것', '저것', '무엇', '뭐', '때문', '일단', '나', '한',
        '에', '의', '엔', '내', '거', '건', '랑', '푹', '님', '난', '들', '특히', '탱', '이네', '이랑', '곧', '금방', '이에요',
        '드리다', '나다', '나고', '나니', '니', '상', '떨기', '아예', '재', '편', '인데', '스레', '들다', '벌써', '보단',
        '급', '나면', '셈', '씩', '쯤', '함', '딱', '정말', '로서'
    ]    # 불용어 지정

    # 리뷰 텍스트 토큰화
    joined_review_text = " ".join(df_test['review'].dropna().astype(str))
    review_tokens = [word for word in okt.morphs(joined_review_text) if word not in stopwords and len(word) > 1]
    # review_tokens = [word for word in okt.nouns(joined_review_text) if word not in stopwords and len(word) > 1]
    reviewtxt_for_wordcloud = " ".join(review_tokens)

    # 코멘트 텍스트 토큰화
    joined_comments_text = " ".join(df_test['comments'].dropna().astype(str))
    comments_tokens = [word for word in okt.morphs(joined_comments_text) if word not in stopwords and len(word) > 1]
    # comments_tokens = [word for word in okt.nouns(joined_comments_text) if word not in stopwords and len(word) > 1]
    commentstxt_for_wordcloud = " ".join(comments_tokens)

    return pie_label_list, reviewtxt_for_wordcloud, commentstxt_for_wordcloud

# 차트 생성 함수
def plot_instagram(
        pie_label_list, reviewtxt_for_wordcloud, commentstxt_for_wordcloud
        ):
    # st.set_page_config(layout="wide")   #화면 넓게
    # 탭 타이틀
    # st.header('@@음식점 카카오맵 리뷰')

    # 파이차트(라벨링)
    labeling_pie_fig = go.Figure()  #그래프 객체 생성
    labeling_pie_fig.add_trace(
        go.Pie(
            labels=pie_label_list[0],   #홍보/진정성
            values=pie_label_list[1],   #라벨링별 예측값 비율
            #차트 색상
            marker=dict(
                colors=["#fe2600", "#ffb012"]
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

    # 워드클라우드
    review_wordcloud = WordCloud(
        # font_path =Path(__file__).parent / "NanumGothic.ttf",
        font_path='C:/Windows/Fonts/NanumGothic.ttf',   # 로컬 테스트용 폰트 경로
        background_color='white',   #default: black
        colormap='plasma',  # 색상 테마 지정
        # random_state=42 #단어 위치 고정
    ).generate(reviewtxt_for_wordcloud)
    review_wordcloud_plot_fig = plt.figure()  #객체 생성
    plt.imshow(review_wordcloud, interpolation='bilinear')
    # 그래프에서 한글, 유니코드 '-' 깨짐방지'
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False    
    plt.title('진정성 리뷰 워드클라우드', fontdict={'fontweight':'bold'}, fontsize=8) #그래프 제목, 볼드체
    plt.axis('off')
    plt.show()

    comments_wordcloud = WordCloud(
        # font_path =Path(__file__).parent / "NanumGothic.ttf",
        font_path='C:/Windows/Fonts/NanumGothic.ttf',   # 로컬 테스트용 폰트 경로
        background_color='white',   #default: black
        colormap='Greys',  # 그레이스케일 컬러맵
        # random_state=42 #단어 위치 고정
    ).generate(commentstxt_for_wordcloud)
    commments_wordcloud_plot_fig = plt.figure()  #객체 생성
    plt.imshow(comments_wordcloud, interpolation='bilinear')
    # 그래프에서 한글, 유니코드 '-' 깨짐방지'
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('진정성 코멘트 워드클라우드', fontdict={'fontweight':'bold'}, fontsize=8) #그래프 제목, 볼드체
    plt.axis('off')
    plt.show()

    return labeling_pie_fig, review_wordcloud_plot_fig, commments_wordcloud_plot_fig  


### code for run python file (filename: )

# import streamlit as st
# import duckdb
# import plotly.graph_objects as go
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path
# from insta_plot import get_instagram, plot_instagram

# DB위치 및 연결설정
db_path = Path("G:\내 드라이브") / "reviews.db"     # 주의! 로컬 접속용 경로
conn = duckdb.connect(db_path, read_only=True)
call_store_name = "순대일번지"  # 테스트용(기본값) 가게 이름

if conn:
    pie_label_list, reviewtxt_for_wordcloud, commentstxt_for_wordcloud = get_instagram(conn, call_store_name)
    labeling_pie_fig, review_wordcloud_plot_fig, commments_wordcloud_plot_fig = plot_instagram(pie_label_list, reviewtxt_for_wordcloud, commentstxt_for_wordcloud)
        
    if labeling_pie_fig and review_wordcloud_plot_fig and commments_wordcloud_plot_fig:
        st.set_page_config(layout='wide')   #streamlit 화면 넓게
        st.header(f'{call_store_name} 음식점 인스타그램 리뷰')

        # 그래프(좌측), 워드클라우드(우측) 컬럼 생성
        plot_col, w_cloud = st.columns([.3, .7])  #3:7 비율로 분할

        # 파이차트
        with plot_col:
            st.plotly_chart(labeling_pie_fig, use_container_width=True)
        
        # 리뷰 워드클라우드
        with w_cloud:
            st.pyplot(review_wordcloud_plot_fig)

        # 코멘트 워드클라우드(참조용)
        # st.pyplot(commments_wordcloud_plot_fig)
       
    else:
        st.error('could not find graphs')

conn.close()