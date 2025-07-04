### 카카오맵 리뷰 구분 멀티모달 모델 학습 ###

# --- 1. 라이브러리 임포트 ---
import pandas as pd
import numpy as np
import re
import os
import warnings
warnings.filterwarnings('ignore') # 불필요한 경고 무시

from tqdm.auto import tqdm
from konlpy.tag import Okt 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix 
from sentence_transformers import SentenceTransformer
import torch 

# 이미지 특징 추출을 위한 라이브러리 추가
from PIL import Image # 이미지 로드
from torchvision import models, transforms # 사전 학습된 이미지 모델 및 전처리
import urllib

# XGBoost 모델을 위한 라이브러리
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

tqdm.pandas() # tqdm for pandas 적용 (apply 함수에 프로그레스 바 표시)
file_path = 'kakaomap_yeonnam_reviews_autolabeled_final.csv'
## GPU 사용 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

print("--- 1. 필수 라이브러리 임포트 및 초기 설정 완료 ---")

# --- 2. 데이터 로드 및 기본 전처리 ---
print("\n--- 2. 데이터 로드 및 기본 전처리 ---")

try:
    df = pd.read_csv(file_path, encoding='utf-8-sig') 

    ## 기본 결측치 제거
    df['리뷰내용'] = df['리뷰내용'].replace('포토 리뷰 홍보', np.nan)   # 텍스트 모델 학습시 사용한 데이터 보강용 텍스트 제거
    df.dropna(subset=['예측라벨'], inplace=True)    # 라벨이 없는 행 제거
    df.dropna(subset=['리뷰내용', '사진URL'], how='all', inplace=True) # 리뷰 본문, 사진URL 모두 없는 행 제거
    # df.dropna(subset=['리뷰내용', '예측라벨'], inplace=True)  # 리뷰 본문 or 라벨링 데이터 없는 행 제거
    print(f"데이터 로드 완료. 총 {len(df)}개의 리뷰 데이터.")

    okt = Okt()
    ## 리뷰내용을 토큰화해 반환하는 함수
    def preprocess_and_tokenize(text):
        if not isinstance(text, str):
            return []
            
        # 한글, 영어, 숫자, 공백 외의 모든 문자를 제거
        cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text) 
        
        # 여러 개의 공백을 하나의 공백으로 줄이고 양쪽 끝의 공백 제거
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # 형태소 분석 및 어간 추출
        tokens = okt.morphs(cleaned_text, norm=True, stem=True)
        return tokens
    
    ## DF에 토큰화된 리뷰 컬럼(processed_review) 추가
    print("※ 리뷰내용 토큰화 작업")
    df['processed_review'] = df['리뷰내용'].progress_apply(preprocess_and_tokenize)
    
    ## 토큰화에 따른 새로운 결측치 제거     # 멀티모달모델 특성상 미사용
    # print("※ 토큰화 후 새로운 결측치 제거 작업")
    # initial_rows = len(df)    
    # df = df[df['processed_review'].progress_apply(lambda x: len(x) > 0)].copy() 
    # rows_after_removal = len(df)
    # if initial_rows - rows_after_removal > 0:
    #     print(f"- 토큰화 후 비어 있는 {initial_rows - rows_after_removal}개의 리뷰를 제거 -")
    # print(f"- 최종 리뷰 개수: {rows_after_removal}개 -")

    # 불용어 목록
    korean_stopwords = [
        '은', '는', '이', '가', '을', '를', '과', '와', '도', '만', '으로', '로', '적', '인', '이다', '이고', '이며', '이니',
        '수', '개', '분', '등', '고', '게', '듯', '음', '안', '것', '때', '곳', '분들', '요', '에서', '하다', '되다',
        '데', '그냥', '네', '응', '오', '아', '그', '저', '저런', '그것', '저것', '무엇', '뭐', '때문', '일단', '나', '한',
        '에', '의', '엔', '내', '거', '건', '랑', '푹', '님', '난', '들', '특히', '탱', '이네', '이랑', '곧', '금방', '이에요',
        '드리다', '나다', '나고', '나니', '니', '상', '떨기', '아예', '재', '편', '인데', '스레', '들다', '벌써', '보단',
        '급', '나면', '셈', '씩', '쯤', '함', '딱'
    ]
    stopwords_set = set(korean_stopwords)

    print("※ 불용어 처리 작업")
    df['processed_review_cleaned'] = df['processed_review'].progress_apply(
        lambda tokens: [token for token in tokens if token not in stopwords_set]
    )
    print("- 불용어 처리 완료 -")

except Exception as e:
    print(f"***** 데이터 로드 및 전처리 오류: {e} *****")
    raise


### --- 3. 수치형 특징 추출 및 스케일링 ---
print("\n--- 3. 광고성 키워드 빈도 및 수치형 특징 추출 및 스케일링 ---")

ad_keywords = [
    '방문', '건강', '데이트', '부모', '가족', '우연', '지나가다', '친구', '분위기', '비주얼',
    '맛집', '가성비', '이벤트', '인스타그램', '인스타', '유튜브', '친절', '퀄리티', '핫플',
    '깔끔', '청결', '좋아요', '맛있다', '또오다', '나중', '오겠다', '호기심', '잘생김',
    '무조건', '음식', '새로', '오픈', '인생' , '오래', '번창', '사장', '오랜', '생일', '의사',
    '포토', '리뷰', '홍보'
]

ad_keywords_stemmed_set = set()
for keyword in ad_keywords:
    stemmed_tokens = okt.morphs(keyword, norm=True, stem=True)
    ad_keywords_stemmed_set.update(stemmed_tokens)

ad_keywords_set = ad_keywords_stemmed_set
print(f"- 변환된 광고성 키워드 셋: {ad_keywords_set}")

## 광고성 키워드 카운터 생성 함수
def count_ad_keywords(tokens_list, keywords_set):
    count = 0
    for token in tokens_list:
        if token in keywords_set:
            count += 1
    return count

print("※ 광고성 키워드 카운터 생성중")
df['ad_keyword_count'] = df['processed_review_cleaned'].progress_apply(
    lambda x: count_ad_keywords(x, ad_keywords_set)
)

numerical_features = ['별점평균', '후기개수', '팔로워수', '별점', 'ad_keyword_count']
X_numerical = df[numerical_features].copy() 

for col in numerical_features:
    if X_numerical[col].isnull().sum() > 0:
        X_numerical[col].fillna(0, inplace=True) 
        print(f"** 경고: '{col}' 컬럼에 결측치가 있어 0으로 채웠습니다. **")

scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical)

print(f"- 스케일링된 수치형 특징의 형태: {X_numerical_scaled.shape}")


### --- 4. SentenceTransformer를 활용한 텍스트 임베딩 ---
print("\n--- 4. SentenceTransformer를 활용한 텍스트 임베딩 ---") 

print("- SentenceTransformer 모델 로드 중... (jhgan/ko-sroberta-multitask) ")
sbert_model = SentenceTransformer('jhgan/ko-sroberta-multitask', device=device)
print("- SentenceTransformer 모델 로드 완료. 리뷰 텍스트 임베딩 생성 시작...")
review_contents = df['리뷰내용'].tolist() # '리뷰내용' 컬럼의 모든 값을 리스트로 변환

# NaN이 아닌 리뷰 내용의 인덱스와 실제 텍스트만 추출
valid_review_indices = [i for i, x in enumerate(review_contents) if pd.notna(x)]
valid_review_texts = [review_contents[i] for i in valid_review_indices]

if valid_review_texts:
    # 유효한 텍스트(문자열)에 대해서만 SentenceTransformer 임베딩 수행
    sbert_embeddings_valid = sbert_model.encode(valid_review_texts, show_progress_bar=True, convert_to_numpy=True)
    
    # 전체 리뷰 수에 해당하는 크기의 0 벡터 배열을 생성하여 임베딩을 저장할 공간 마련
    # sbert_embeddings_valid.shape[1] : SentenceTransformer 임베딩의 차원
    review_embeddings = np.zeros((len(review_contents), sbert_embeddings_valid.shape[1]))
    
    # 유효한 임베딩 값을 원래 데이터프레임에서의 위치에 맞춰 할당
    for i, original_idx in enumerate(valid_review_indices):
        review_embeddings[original_idx] = sbert_embeddings_valid[i]
else:
    # 모든 '리뷰내용'이 NaN이거나 유효한 텍스트가 없는 경우
    # SBERT 모델의 임베딩 차원을 가져와서 0 벡터로 전체를 채움
    try:
        embedding_dim = sbert_model.get_sentence_embedding_dimension()
    except:
        # 모델 차원을 알 수 없는 경우 - 기본값 768 (jhgan/ko-sroberta-multitask 모델)
        embedding_dim = 768
    review_embeddings = np.zeros((len(review_contents), embedding_dim))

print(f"- 생성된 텍스트 임베딩 형태: {review_embeddings.shape}")


### --- 5. 이미지 특징 추출 (사전 학습된 CNN 활용) ---
print("\n--- 5. 이미지 특징 추출 (사전 학습된 CNN 활용) ---")

# 이미지가 저장된 폴더 경로
IMAGE_FOLDER_PATH = "review_images" # 사용자가 명시한 폴더 이름

# URL 문자열에서 원본 이미지 URL 리스트를 추출하는 함수
def urls_string_to_list(url_string):
    if pd.isna(url_string) or url_string == 'N/A':
        return []
    urls_list = []
    temp_url_list = [url.strip() for url in url_string.split(',') if url.strip()]
    for encoded_url in temp_url_list:
        parsed_url = urllib.parse.urlparse(encoded_url) # urllib.parse가 이 함수에서만 필요하므로 여기서 임포트
        query_params = urllib.parse.parse_qs(parsed_url.query)
        if 'fname' in query_params and query_params['fname']:
            decoded_fname = urllib.parse.unquote(query_params['fname'][0])
            urls_list.append(decoded_fname)
        else:
            continue
    return urls_list

# 이미지 URL에서 고유 ID (파일명)를 추출하는 함수
def get_unique_id_from_url(decoded_inner_url):
    # 1번 패턴: t1.daumcdn.net/local/kakaomapPhoto/review/<ID>%3Foriginal
    match_pattern1 = re.search(r't1\.daumcdn\.net/local/kakaomapPhoto/review/([a-f0-9]+)(?:%3Foriginal)?', decoded_inner_url)
    if match_pattern1:
        return match_pattern1.group(1)

    # 2번 패턴: t1.daumcdn.net/local/review_placeapp/<ID>.jpeg (or other extension)
    match_pattern2 = re.search(r't1\.daumcdn\.net/local/review_placeapp/([a-zA-Z0-9_-]+)\.[a-z]+$', decoded_inner_url)
    if match_pattern2:
        # 2번 패턴의 경우 저장 시 소문자로 변환되므로, 여기서도 소문자로 변환
        return match_pattern2.group(1).lower()
        
    return None # 매칭되는 패턴이 없는 경우

# GPU 사용 설정 (SBERT와 동일)
# device = "cuda" if torch.cuda.is_available() else "cpu"

# 사전 학습된 모델 로드 (EfficientNetB0 권장)
weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
image_model = models.efficientnet_b0(weights=weights)

# 마지막 분류 계층 제거 (특징 추출기 역할만 하도록)
image_model.classifier = torch.nn.Identity() 
image_model.to(device)
image_model.eval() # 모델을 평가 모드로 설정 (드롭아웃, 배치 정규화 등에 영향)

# 이미지 전처리 파이프라인 정의 (ImageNet 모델의 표준 전처리)
preprocess = weights.transforms()

image_embeddings = []
print("- 이미지 특징 추출 시작...")

# 이미지 파일명에 사용될 수 있는 일반적인 확장자들
POSSIBLE_EXTENSIONS = [".jpg", ".png", ".jpeg"]
# POSSIBLE_EXTENSIONS = [".jpg", ".png", ".jpeg", ".gif", ".webp", ".bmp"]

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Image Feature Extraction"):
    image_urls_string = row['사진URL']
    image_urls_list = urls_string_to_list(image_urls_string)
    
    current_image_embedding = None

    if image_urls_list: # 이미지가 있을 경우
        first_image_url = image_urls_list[0] # 첫 번째 이미지 URL 사용
        unique_id = get_unique_id_from_url(first_image_url) # 고유 ID 추출
        
        if unique_id:
            found_image_path = None
            # 가능한 확장자를 모두 시도하여 실제 파일 찾기
            for ext in POSSIBLE_EXTENSIONS:
                potential_path = os.path.join(IMAGE_FOLDER_PATH, f"{unique_id}{ext}")
                if os.path.exists(potential_path):
                    found_image_path = potential_path
                    break # 파일 찾으면 루프 종료
            
            if found_image_path:
                try:
                    img = Image.open(found_image_path).convert('RGB') # 이미지를 RGB로 로드
                    img_tensor = preprocess(img).unsqueeze(0).to(device) # 배치 차원 추가 및 GPU 이동
                    
                    with torch.no_grad(): # 역전파 계산 비활성화 (메모리 절약, 속도 향상)
                        features = image_model(img_tensor)
                        current_image_embedding = features.cpu().numpy().flatten()
                except Exception as e:
                    # print(f"Warning: 이미지 처리 실패 ({found_image_path}): {e}. 0 벡터 사용.") # 너무 많은 출력 방지
                    current_image_embedding = np.zeros(image_model.features[-1].out_channels) 
            else:
                # print(f"Warning: 이미지 파일 없음 (ID: {unique_id}). 0 벡터 사용.") # 너무 많은 출력 방지
                current_image_embedding = np.zeros(image_model.features[-1].out_channels)
        else:
            # print(f"Warning: URL에서 고유 ID 추출 실패 ({first_image_url}). 0 벡터 사용.") # 너무 많은 출력 방지
            current_image_embedding = np.zeros(image_model.features[-1].out_channels)
    else:
        # 이미지가 없는 경우 (사진URL 컬럼이 비어있거나 'N/A') 0 벡터로 대체
        current_image_embedding = np.zeros(image_model.features[-1].out_channels)
        
    image_embeddings.append(current_image_embedding)
        
image_embeddings = np.array(image_embeddings)
print(f"- 생성된 이미지 임베딩 형태: {image_embeddings.shape}")


### --- 6. 모든 특징 결합 ---
print("\n--- 6. 모든 특징 결합 ---") 

try:
    X_combined_final = np.concatenate([review_embeddings, X_numerical_scaled, image_embeddings], axis=1) 
except Exception as e:
    print(f"***** 오류: np.concatenate 중 오류 발생. 오류 메시지: {e}")
    print("*** NumPy 및 SciPy 버전을 확인하거나, 커널 재시작 후 다시 시도해보세요. ***")
    raise 

print(f"- 결합된 최종 특징 벡터의 형태 (SentenceTransformer + Numerical + Image): {X_combined_final.shape}")
print(f"- 최종 특징 벡터의 타입: {type(X_combined_final)}")

y = df['예측라벨'].values
print(f"- 라벨(y)의 형태: {y.shape}")

print("-- 멀티모달 특징 추출 및 결합 완료! --")


### --- 7. 데이터 분할 ---
print("\n--- 7. 데이터 분할 ---") 

X_train, X_test, y_train, y_test = train_test_split(
    X_combined_final, y, test_size=0.2, random_state=42, stratify=y
)

print(f"- 학습 데이터 형태: {X_train.shape}, {y_train.shape}")
print(f"- 테스트 데이터 형태: {X_test.shape}, {y_test.shape}")
print(f"- 학습 세트의 광고성(0.0) 비율: {np.mean(y_train == 0):.2f}")
print(f"- 테스트 세트의 광고성(0.0) 비율: {np.mean(y_test == 0):.2f}")



# --- 8. XGBoost 모델 학습 (하이퍼파라미터 세부 튜닝) ---
print("\n--- 8. XGBoost 모델 학습 (하이퍼파라미터 세부 튜닝) ---")

# 클래스 불균형 처리를 위한 scale_pos_weight 계산
count_class_0 = np.bincount(y_train.astype(int))[0] # 광고성(0.0) 개수
count_class_1 = np.bincount(y_train.astype(int))[1] # 진짜리뷰(1.0) 개수

# 소수 클래스(0.0, 광고성)에 더 큰 가중치를 부여하기 위해
# (다수 클래스 수) / (소수 클래스 수)로 계산
scale_pos_weight_value = count_class_1 / count_class_0 if count_class_0 > 0 else 1

print(f"- 계산된 scale_pos_weight (다수/소수 비율, 1.0/0.0): {scale_pos_weight_value:.2f}")

# 탐색할 하이퍼파라미터 그리드 정의 (조정된 파라미터 그리드)
# 최적값(250627) : {'learning_rate': 0.07, 'max_depth': 7, 'n_estimators': 1000}
param_grid = {
    'n_estimators': [600, 800, 1000], 
    'learning_rate': [0.03, 0.05, 0.07], 
    'max_depth': [5, 6, 7],            
}

# XGBoost Classifier 초기화
base_model = XGBClassifier(
    random_state=42,
    tree_method='gpu_hist', # GPU 사용 설정
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight_value
)

# GridSearchCV 설정
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=1, 
    verbose=2
)

# GridSearchCV를 사용하여 최적의 모델 학습
grid_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터 및 최고 점수 출력
print(f"\n- 최적 하이퍼파라미터: {grid_search.best_params_}")
print(f"- 최고 ROC-AUC 점수 (교차 검증): {grid_search.best_score_:.4f}")

# 최적의 모델을 최종 모델로 사용
model = grid_search.best_estimator_
print("-- XGBoost 모델 학습 완료 (GridSearchCV를 통해 최적화된 모델)! --")



### --- 9. 모델 성능 평가 (XGBoost 모델용) ---
print("\n--- 9. 모델 성능 평가 ---")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # 1.0 (진짜 리뷰) 클래스에 대한 예측 확률

# Classification Report 출력
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['광고성(0.0)', '진짜리뷰(1.0)']))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Confusion Matrix 출력
cm = confusion_matrix(y_test, y_pred, labels=[0.0, 1.0])
print("\n--- Confusion Matrix ---")
print("             [예측]")
print("           광고성(0.0) | 진짜리뷰(1.0)")
print("         ---------------------")
print(f"실제 광고성(0.0) |    {cm[0,0]:<5}  |   {cm[0,1]:<5}")
print(f"실제 진짜리뷰(1.0)|    {cm[1,0]:<5}  |   {cm[1,1]:<5}")

print("\n모델 학습 및 평가 완료!")


########## 학습 모델 및 관련 설정 저장(선택) ##########

# import joblib
# import os

# ### --- 10. 모델 및 관련 파라미터 저장 ---
# print("\n--- 10. 모델 및 관련 파라미터 저장 ---")

# # 모델 저장 경로 설정 (프로젝트 폴더 내 'multimodal_model_artifacts' 폴더 생성 권장)
# model_save_dir = 'multimodal_model_artifacts' # 이전 응답의 SAVE_DIR과 동일
# os.makedirs(model_save_dir, exist_ok=True) # 폴더가 없으면 생성
# print(f"'{model_save_dir}' 폴더를 확인/생성했습니다.")

# # 1. 분류 모델 (XGBoost) 저장
# model_path = os.path.join(model_save_dir, 'ad_classifier.joblib')
# joblib.dump(model, model_path)
# print(f"광고/진짜 분류 모델 저장 완료: {model_path}")

# # 2. StandardScaler (수치형 특징 스케일링용) 저장
# scaler_path = os.path.join(model_save_dir, 'numerical_scaler.joblib')
# joblib.dump(scaler, scaler_path)
# print(f"수치형 특징 스케일러 저장 완료: {scaler_path}")

# # 불용어 저장
# stopwords_file = os.path.join(model_save_dir, 'korean_stopwords.txt')
# with open(stopwords_file, 'w', encoding='utf-8') as f:
#     for word in korean_stopwords: # 'korean_stopwords' 변수가 스크립트 상단에 정의되어 있어야 함
#         f.write(word + '\n')
# print(f"불용어 목록 저장 완료: {stopwords_file}")

# # 광고성 키워드 저장 (원본 리스트 저장)
# ad_keywords_file = os.path.join(model_save_dir, 'ad_keywords.txt')
# with open(ad_keywords_file, 'w', encoding='utf-8') as f:
#     for word in ad_keywords: # 'ad_keywords' 변수가 스크립트 상단에 정의되어 있어야 함
#         f.write(word + '\n')
# print(f"광고성 키워드 목록 저장 완료: {ad_keywords_file}")

# # 3. SentenceTransformer 모델 저장 (모델 자체를 저장)
# # sbert_model은 학습 후 생성된 SentenceTransformer 인스턴스입니다.
# sbert_model_path = os.path.join(model_save_dir, 'sbert_korean_model')
# sbert_model.save_pretrained(sbert_model_path)
# print(f"SentenceTransformer 모델이 '{sbert_model_path}' 에 저장되었습니다.")

# # 4. EfficientNetB0 이미지 특징 추출기 (PyTorch 모델의 state_dict) 저장
# # image_model은 EfficientNetB0 인스턴스입니다.
# image_model_state_dict_path = os.path.join(model_save_dir, "efficientnet_b0_image_feature_extractor.pth")
# torch.save(image_model.state_dict(), image_model_state_dict_path)
# print(f"EfficientNetB0 이미지 특징 추출기 (state_dict)가 '{image_model_state_dict_path}' 에 저장되었습니다.")


# print("\n모든 관련 모델, 스케일러 및 키워드 목록 저장이 완료되었습니다.")