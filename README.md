# Pater sentence tagging 논문 문장 의미 태깅
<img width="462" alt="캡처" src="https://user-images.githubusercontent.com/61795757/99826724-19a3f480-2b9c-11eb-834a-c162730c52c4.PNG">
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

#### Format
<img width="276" alt="data" src="https://user-images.githubusercontent.com/61795757/99826322-8ec2fa00-2b9b-11eb-99dd-b9f9f8058c25.PNG">
Input file 형식: (tag  sentence) 탭으로 분리된 파일 (.tsv)
My data: 63437개의 data (논문 태그, 논문 문장)

#### Tokenization
카카오 형태소 분석기 Khaiii Tokenizer을 이용해 문장 토큰화

#### Dataset shuffle and Split, Cross Validation
Train set : Test set = 8:2
Validation set: Train의 20% 

## Model

### cnn, rnn ensemble

#### train.py
cnn, rnn 모두 학습

#### classify.py
<img width="557" alt="cnn,rnn emsemble" src="https://user-images.githubusercontent.com/61795757/99826409-aa2e0500-2b9b-11eb-81d8-f2b1df806ee1.PNG">
(Best CNN + Best RNN)/2
- 저장된 cnn 모델 중 가장 좋은 성능을 가진 모델과 rnn 모델 중 가장 좋은 성능을 가진 best 모델 
모두 추론 적용
- 추론 적용 결과 값들의 평균 구해서 k 번째로 높은 순위를 가진 라벨과 문장 출력 (top-k 설정)

### KcBERT

#### train.py
<img width="560" alt="kc-bert" src="https://user-images.githubusercontent.com/61795757/99826428-b4500380-2b9b-11eb-8ead-a31097f7907e.PNG">
Pre-trained BERT model의 weights 로드 --> fine-tuning 수행

#### classify.py
- Pre-trained BERT model로 fine-tuning한 최종 모델에 추론 적용
- 추론 적용 결과 값들 중 k번째로 높은 순위를 가진 라벨과 문장 출력 (top-k 설정)

### Inference result
- 실행 결과 (라벨, 문장)
![result](https://user-images.githubusercontent.com/61795757/99826609-f1b49100-2b9b-11eb-9bba-3255adcd6341.png)

- 학습 조건 및 Accuracy
<img width="570" alt="final_model" src="https://user-images.githubusercontent.com/61795757/99826922-553ebe80-2b9c-11eb-85bf-50b7de3f1cba.PNG">

## Reference
- [Simple Neural Text Classification](https://github.com/kh-kim/simple-ntc])
- [KcBERT: Korean comments BERT](https://github.com/Beomi/KcBERT)
- [카카오 형태소 분석기] (https://github.com/kakao/khaiii)
