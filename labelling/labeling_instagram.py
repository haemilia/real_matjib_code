import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
import wandb
from transformers import TrainingArguments, Trainer
import optuna   #기계학습 모델의 하이퍼파라미터 자동 조정하고 최적화하는 오픈 소스 라이브러리
from transformers import pipeline
import duckdb

#토크나이저 관련 경고 무시
os.environ['TOKENIZERS-PARALLELISM'] = 'true'

#device 지정: 딥러닝 학습 속도 향상
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

df = pd.read_excel('real_matjib/dataset/instagram_labeling.xlsx')

#결측치 없애고 데이터타입 str로
df = df.fillna('').astype(str)

df['reviews'] = (
    #사전모델의 max_lengths가 512여서 가장 긴 df['text']는 마지막에 추가
    df['ids'] + ' ' +   #계정 아이디
    df['tags'] + ' ' +  #태그
    df['cmts'] + ' ' +  #댓글
    df['text']  #본문
)

#컬럼명 수정
df = df.rename(columns={'category':'label'})

#깔끔하게 보이기 위해 내부에 리스트 문자열 제거
df['reviews'] = df['reviews'].str.replace(r'[\[\]]', '', regex=True)

#필요한 컬럼만 가져오기
df_reviews = df[['food_house', 'search', 'reviews', 'label']].copy()
df_labeled = df_reviews.loc[:999].copy()    #라벨링 #length: 1000
df_unlabeled = df_reviews.loc[1000:].copy()  #라벨링 X

#라벨링 학습위해서는 라벨링 값 int 타입으로 전환
df_labeled['label'] = df_labeled['label'].map({'일반':0, '홍보':1}).astype(int)

#라벨링된 데이터 Dataset으로 변환
ds_labeled = Dataset.from_pandas(df_labeled)
split_ds = ds_labeled.train_test_split(test_size=0.2, seed=42)
dataset = DatasetDict({
    'train': split_ds['train'],
    'validation': split_ds['test']
})

#len(dataset['train']), len(dataset['validation'])  #(800, 200)이어야 함

#예측용 데이터 Dataset으로 변환
ds_unlabeled = Dataset.from_pandas(df_unlabeled)

#토크나이저 준비
tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')

def preprocess_function(examples):
    return tokenizer(
        examples['reviews'],
        truncation = True,
        padding = True,
        max_length = 512
    )

#토크나이저 적용
tokenized_datasets = {
    'train': dataset['train'].map(preprocess_function, batched=True),
    'validation': dataset['validation'].map(preprocess_function, batched=True),
    'test': ds_unlabeled.map(preprocess_function, batched=True)
}

#print(len(tokenized_datasets['validation']))  # 200이어야 함

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        'klue/roberta-large',
        num_labels = 2
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    #sigmoid로 확률값 변환(이진분류)
    probs = 1/(1+np.exp(-logits))
    predictions = (probs > 0.5).astype(int)
    predicted_classes = predictions.argmax(axis=1)

    metrics = {
        'eval_accuracy': accuracy_score(labels, predicted_classes),
        'eval_f1': f1_score(labels, predicted_classes),
        'eval_precision': precision_score(labels, predicted_classes),
        'eval_recall': recall_score(labels, predicted_classes)
    }

    try:
        metrics['eval_roc_auc'] = roc_auc_score(labels, probs[:, 1])
        metrics['eval_pr_auc'] = average_precision_score(labels, probs[:, 1])
    except ValueError:
        metrics['eval_roc_auc'] = float('nan')
        metrics['eval_pr_auc'] = float('nan')

    print("compute_metrics 반환값:", metrics)  # 디버깅용
    return metrics

def train_with_params(params, trial_number=None, is_best=False):
    # wandb run 생성 (is_best=True이면 'best_params'로 이름 지정)
    run_name = f'experiment-{trial_number + 1}' if not is_best else 'best_params'
    output_dir = f'/results/exp{trial_number}' if not is_best else '/.results/best_params'
    
    run = wandb.init(
        project='hugging_face_with_Optuna' if not is_best else 'instagram_labeling_with_Optuna',
        name=run_name,
        config=params,
        reinit=True
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=params['learning_rate'],
        per_device_train_batch_size=params['train_batch_size'],
        per_device_eval_batch_size=params['eval_batch_size'],
        num_train_epochs=params['num_train_epochs'],
        weight_decay=params['weight_decay'],
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        report_to='wandb'
    )
    
    trainer = Trainer(
        model=model_init(),
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    trainer.train()
    
    metrics = trainer.evaluate()
    print("trainer.evaluate() 결과:", metrics)
    
    pred_output = trainer.predict(tokenized_datasets['validation'])
    y_true = pred_output.label_ids
    logits = pred_output.predictions
    probs = 1/(1+np.exp(-logits))
    probs_class1 = probs[:, 1]
    probs_for_wandb = np.stack([1-probs_class1, probs_class1], axis=1)
    
    wandb.log({'ROC AUC': wandb.plot.roc_curve(y_true, probs_for_wandb)})
    wandb.log({'PR AUC': wandb.plot.pr_curve(y_true, probs_for_wandb)})

    final_metrics = trainer.evaluate()
    
    run.finish()
    
    return trainer.model, final_metrics

def objective(trial):
    #하이퍼파라미터 탐색
    params = {
        'learning_rate' : trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True),
        #'batch_size' : trial.suggest_categorical('batch_size', [8]),
        'train_batch_size' : trial.suggest_categorical('train_batch_size', [4, 8]),
        'eval_batch_size' : trial.suggest_categorical('eval_batch_size', [8, 16]),
        'num_train_epochs' : trial.suggest_int('num_train_epochs', 3, 10),
        'weight_decay' : trial.suggest_float('weight_decay', 0.001, 0.01, log=True)
    }

    result = train_with_params(params, trial_number=trial.number)
    if isinstance(result, tuple):
        metrics = result[1]  # 튜플이면 두 번째 원소가 metrics
    else:
        metrics = result     # 아니면 그대로 metrics

    return metrics['eval_accuracy']


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)   #하이퍼파라미터 새로운 조합 시도 횟수(experiment 횟수)
print('Best params: ', study.best_params)
#Best params:  {'learning_rate': 4.032897223604241e-05, 'train_batch_size': 8, 'eval_batch_size': 32, 'num_train_epochs': 8, 'weight_decay': 0.0018773593172162295}

#최적의 파라미터 조합 적용
best_params = study.best_params
model, metrics = train_with_params(best_params, is_best=True)

#모델 저장
model.save_pretrained('instagram_reviews_labeling_model')
tokenizer.save_pretrained('instagram_reviews_labeling_model')

test_inputs = tokenized_datasets['test'].to_dict()
input_ids = torch.tensor(test_inputs['input_ids']).to('cuda:0')
attention_mask = torch.tensor(test_inputs['attention_mask']).to('cuda:0')

batch_size = 8  # GPU 메모리 여유에 따라 조정
#데이터가 너무 많아서 한 번에 메모리에 올리기 어렵거나 예측이 너무 오래 걸릴 때
chunk_size = 500  # 한 번에 예측할 데이터 개수 (input_ids 기준)
results = []

for chunk_start in range(0, len(input_ids), chunk_size):
    chunk_end = chunk_start + chunk_size
    chunk_input_ids = input_ids[chunk_start:chunk_end]
    chunk_attention_mask = attention_mask[chunk_start:chunk_end]
    
    chunk_results = []
    for i in range(0, len(chunk_input_ids), batch_size):
        batch_input_ids = chunk_input_ids[i:i+batch_size]
        batch_attention_mask = chunk_attention_mask[i:i+batch_size]
        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        chunk_results.append(outputs.logits)
    results.extend(chunk_results)

logits = torch.cat(results, dim=0)
predictions = np.argmax(logits.cpu().numpy(), axis=1)

df_unlabeled['label'] = predictions
df_unlabeled['label'] = df_unlabeled['label'].map({0:'일반', 1:'홍보'}).astype(str)

df_result = pd.concat([df_labeled, df_unlabeled], axis=0)

#엑셀 파일을 duck db 파일로 변환
#데이터베이스 파일에 연결
conn = duckdb.connect('instagram_labeling.duckdb')

#df_result를 DuckDB 테이브롤 저장
conn.sql("CREATE TABLE INSTA_LABEL AS SELECT * FROM df_result")

#연결 종료
conn.close()