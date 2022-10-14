# Capstone-2022-1-05



## 1. 프로젝트 소개
### **Backgroup**

한국어에는 영어와 달리 조사가 존재하는데, 현재 배포된 사전학습 언어모델은 영어의 언어단위 구성방법(Tokenizer)를 사용하기에 한국어 조사의 구조적 역할을 인식하지 못하여 구문분석이 온전히 이루어 지지 못합니다. 이를 해결하기 위한 사전학습 언어모델을 개발하고자 하였습니다.
또한, 조사에 따라 문장의 의미가 달라지기 때문에 조사와 서술어의 호응 관계에 따라 문장의 의미 파악을 보다 잘 할 수 있는 모델을 개발하는것을 목표로 하였습니다.

### **Problem definition**
1. 한국어 구문 분석을 위한 한국어 특화 사전학습 언어모형 구축
2. 문형정보 기반의 품사 규칙을 이용한 한국어 언어모델 개발

### **Evaluation**
![cross entropy](https://user-images.githubusercontent.com/82591396/195675312-74d3c29c-343f-487a-a9f9-856dbf7d63db.png) \n

예측모형은 실제 분포인 q 를 모르고 모델링을 하여 q 분포를 예측하고자 하는 것입니다.
예측 모델링을 통해 구한 분포 p(x)와 실제 분포인 q(x)의 차이로 실제값과 예측값의 차이를 표현할 수 있습니다.
cross entropy의 결과값은 불확실성으로 낮을 수록 실제 분포를 잘 예측한 것으로 판단할 수 있습니다.



## 2. 팀소개

강혜빈 hv9739@pusan.ac.kr (문형정보탐색, 데이터 전처리, 사전학습모델 input ids 생성)

구채원 gcw127@pusan.ac.kr (문형정보탐색, 데이터 전처리, 사전학습모델 input ids 생성)

조소연, thdus8526@pusan.ac.kr (문형정보탐색, 데이터 전처리, attention score 정리)

## 3. 시스템 구성도

### **1. 한국어 구문 분석을 위한 한국어 특화 사전학습 언어모형**
![bert tunning](https://user-images.githubusercontent.com/82591396/195672484-adc29b19-a3e6-4962-8745-b862886b4bdb.png)
기존의 BERT에 형태소 embedding 추가하여 형태소 정보를 학습할 수 있게 한다.

### **2. 문형정보 기반의 품사 규칙을 이용한 한국어 언어모델**
우리말샘 참고하여 규칙 도출(https://opendict.korean.go.kr/service/dicStat)
![image](https://user-images.githubusercontent.com/82591396/195675722-555398a6-ca80-4bb4-b5d6-8ed35496d91e.png)


![relation check](https://user-images.githubusercontent.com/82591396/195674579-121bce0e-657a-4d16-bfd0-903354f006e9.png)

조사 '에'가 '갔'과 높은 연관성을 가진다는 것을 볼 수 있다.

![attention](https://user-images.githubusercontent.com/82591396/195672541-6d4183ea-f402-44bb-8391-3427b934a49b.png)

높은 연관성을 가지는 단어는 BERT의 Self Attention시 score를 크게 준다.
→attention 적용 전보다 후에서 부사격 조사인 '에'와 동사 '갔'의 상관관계가 높아짐을 확인할 수 있다.

## 4. 소개 및 시연 영상
유튜브 링크 달기!!!!

## 5. 설치 및 사용법
본 프로젝트는 Ubuntu 20.04 버전에서 개발되었으며 함께 포함된 다음의 스크립트를 수행하여 관련 패키지들의 설치와 빌드를 수행할 수 있습니다.

### 환경
- Python 3.7

### 사용 라이브러리
- pip install tensorflow

- pip install transformers

- pip install torch

- pip install 다운로드파일명.whl
  import MeCab
  (별도의 설치 필요)

### Train Models
- RoBERTa
  - klue/roberta-small(https://huggingface.co/klue/roberta-small)
  - klue/roberta-base(https://huggingface.co/klue/roberta-base)
  - klue/roberta-large(https://huggingface.co/klue/roberta-large/tree/main)

- BERT
  - klue/bert-base(https://huggingface.co/klue/bert-base)

- KoBigBird
  - monologg/kobigbird-bert-base(https://huggingface.co/monologg/kobigbird-bert-base)

### Dataset
- 모두의 말뭉치
- 2018 동아일보 신문기사
- Klue NLI (https://huggingface.co/datasets/klue)



## 6. 주차별 계획
!̆̈[plan](https://user-images.githubusercontent.com/82591396/195677334-aabbe7e1-0b56-4685-afc4-be562ea5c2cc.png)
