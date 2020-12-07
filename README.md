# Pater sentence tagging 논문 문장 의미 태깅
<img width="800" alt="캡처" src="https://user-images.githubusercontent.com/61795757/99826724-19a3f480-2b9c-11eb-834a-c162730c52c4.PNG">
논문의 구조에 따라 문장들이 의도하는 역할을 구분하는 라벨(태그) 부착
연구목적: 문제정의, 가설설정, 기술정의
연구방법: 대상데이터, 분석방법, 제안방법, 이론/모형
연구결과: 성능/효과, 후속연구/제안


## 사용언어 및 모듈
- Python
- PyTorch 
- PyTorch Ignite
- TorchText 
- Khaiii API
- Huggingface (for KcBERT)

## Pre-process

1. Format
Input file 형식: (tag  sentence) 탭으로 분리된 파일 (.tsv)
My data: 63437개의 data (논문 태그, 논문 문장)

![data](https://user-images.githubusercontent.com/61795757/99828174-f417ea80-2b9d-11eb-8ee0-55913230ed72.PNG)

2. Pre-process
정규표현식을 통한 불용어제거, 띄어쓰기 수정, 전각문자 반각문자 치환
- txt 파일에 넣어 refine.py 로 일괄적 처리

3. Tokenization
카카오 형태소 분석기 Khaiii Tokenizer을 이용해 문장 토큰화

4. Dataset shuffle and Split, Cross Validation
Train set : Test set = 8:2
Validation set: Train의 20% 

## Model

### cnn, rnn ensemble

__1. train.py__
cnn, rnn 모두 학습

<img width="557" alt="cnn,rnn emsemble" src="https://user-images.githubusercontent.com/61795757/99826409-aa2e0500-2b9b-11eb-81d8-f2b1df806ee1.PNG">


__2. classify.py__

(Best CNN + Best RNN)/2

- 저장된 cnn 모델 중 가장 좋은 성능을 가진 모델과 rnn 모델 중 가장 좋은 성능을 가진 best 모델 
모두 추론 적용
- 추론 적용 결과 값들의 평균 구해서 k 번째로 높은 순위를 가진 라벨과 문장 출력 (top-k 설정)

#### Customized model
Top 2 결과를 출력했을 때 1st label 과 2nd label 정확도 차이가 작은 조합을 선별해 오답노트 형식처럼 binary classification 모델로 재훈련

<img width = "560" src="https://user-images.githubusercontent.com/59900689/101310352-fc587100-3891-11eb-8ca4-5ab845c2143c.png">


### KcBERT

__1. train.py__

<img width="560" alt="kc-bert" src="https://user-images.githubusercontent.com/61795757/99826428-b4500380-2b9b-11eb-8ead-a31097f7907e.PNG">

Pre-trained BERT model의 weights 로드 --> fine-tuning 수행

__2. classify.py__
- Pre-trained BERT model로 fine-tuning한 최종 모델에 추론 적용
- 추론 적용 결과 값들 중 k번째로 높은 순위를 가진 라벨과 문장 출력 (top-k 설정)

### Inference result
- 실행 결과 (라벨, 문장)

![결과](https://user-images.githubusercontent.com/61795757/99828145-eb271900-2b9d-11eb-8ecf-f81449dbb5b2.png)

- 학습 조건 및 Accuracy
<img width="400" alt="final_model" src="https://user-images.githubusercontent.com/61795757/99826922-553ebe80-2b9c-11eb-85bf-50b7de3f1cba.PNG">

## Reference
- [Simple Neural Text Classification](https://github.com/kh-kim/simple-ntc])
- [KcBERT: Korean comments BERT](https://github.com/Beomi/KcBERT)
- [Khaiii tokenizer](https://github.com/kakao/khaiii)
