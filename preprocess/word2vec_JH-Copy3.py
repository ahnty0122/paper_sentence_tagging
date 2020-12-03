#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 0. 라이브러리 임포트


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import re
import os
import csv
from khaiii import KhaiiiApi


# # 1. 데이터 로드  
#   
#   
# ### Text_train, Label_train, Text_test, Label_train 

# In[3]:


#엑셀 파일로 저장된 데이터 불러오기

data = pd.read_csv('data.csv')


# In[4]:


tf = data['대표여부'] == 'Y'
data = data[tf]
data = data.values     #4개의 칼럼 ('논문고유번호', '태그', '문장', '대표여부') 으로 되어있음 (DataFrame > ndarrapy)
print("<< 원본 csv파일의 크기 >>")
print("\n")
print("length: ",len(data))
print("dimension: ",data.ndim)
print("shape: ", data.shape)

#'문장', '태그' 열 저장
np.save('NLP_TextData.npy', data[:,2])
np.save('NLP_LabelData.npy',data[:,1])

TextData = np.load('NLP_TextData.npy', allow_pickle=1)
LabelData = np.load('NLP_LabelData.npy', allow_pickle=1)


# In[5]:


############ 당장 필요 X , 확인하기 위함 ############
# 저장된 데이터 한눈에 보기 : 태그 분포 시각화
# matplotlib 패키지 plt에서 한글 폰트 불러오기
import matplotlib.font_manager as fm
from matplotlib import rc
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
font_list[:] # 윈도우에 깔려 있는 기본 폰트 확인
  
rc('font',family="NanumGothic")

# 그래프 (Yaxis: 갯수, Xaxis: 태그9개)
temp = pd.Series(LabelData)
print(temp.value_counts())
temp.value_counts().plot(kind = 'bar')


# In[6]:


# Traindata, Testdata로 나누기 (7:3)
# 문장-태그의 순서를 동일하게 유지하면서 random.shuffle() 적용
# ndarray : Text_train, Text_test, Label_train, Label_test

print("TextData의 크기: ", TextData.shape)
print("LabelData의 크기:", LabelData.shape)

s = np.arange(TextData.shape[0])
np.random.shuffle(s)

TextData = TextData[s]
#print ("셔플 한 뒤의 문장 배열: ", TextData)
LabelData = LabelData[s]
#print ("셔플 한 뒤의 라벨 배열: ", LabelData) 

from sklearn.model_selection import train_test_split
Text_train, Text_test = train_test_split(TextData, test_size=0.3, shuffle=False)
Label_train, Label_test = train_test_split(LabelData, test_size=0.3, shuffle=False)


# In[7]:


# 잘 나뉘었는지 갯수 확인
print(len(Text_train), len(Text_test))
print(len(Label_train), len(Label_test))

print(Text_train[1004])
print(Label_train[1004])


# # 2.1 문장 전처리 : 띄어쓰기 교정

# In[8]:


# 정규 표현식 
import re

def clean_text(texts):
    corpus = []
    for i in texts:
        review = re.sub(r'\s+', ' ', i) #remove extra space
        review = re.sub(r'\s+', ' ', review) #remove spaces
        review = re.sub(r"^\s+", '', review) #remove space from start
        review = re.sub(r'\s+$', '', review) #remove space from the end
        corpus.append(review)
    return corpus

Text_train = clean_text(Text_train)


# # 2.2 문장 전처리 : 필요한 품사만 남기기

# In[9]:


from khaiii import KhaiiiApi
api = KhaiiiApi()


# In[10]:


tags = ['NNG', 'NNP', 'NNB', 'VV', 'VA', 'VX', 'MAG', 'MAJ', 'XSV', 'XSA']

def pos_text(Text_train):
    corpus = []
    for i in Text_train:
        pos_tagged = ''
        for word in api.analyze(i):
            for morph in word.morphs:
                if morph.tag in tags:
                    pos_tagged += morph.lex + ' ' 
        corpus.append(pos_tagged.strip())
    return corpus

Text_train = pos_text(Text_train)


# In[11]:


for i in range(10):
    print(Text_train[i])


# In[44]:


# 임베딩 모델 인풋으로 가공
from tensorflow.keras.preprocessing.text import text_to_word_sequence

W2V = []

for sentence in Text_train:
    W2V.append( text_to_word_sequence(sentence) )


# In[45]:


# 워드 임베딩
import gensim
from gensim.models import Word2Vec

#model = Word2Vec(sentences=W2V, size=100, window=5, min_count=5, workers=4, sg=0)
word2vec_model = gensim.models.Word2Vec.load('/scratch/kedu19/workspace/ko.bin')
word2vec_model.wv.vectors.shape
print(word2vec_model.wv.most_similar("소방"))


# In[46]:


def get_vector(word):
    if word in word2vec_model:
        return word2vec_model[word]
    else:
        return None

for sentence in W2V:
    for words in sentence:
        if words in word2vec_model:
            words = get_vector(words)
       
시바...


# In[30]:





# # 3.1 정수 인코딩 : 원 핫 인코딩
# 

# In[15]:


'''
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
maxwords = 15000
Text_train = np.array(Text_train) # 리스트에서 다시 ndarray로 자료형 변환
tokenizer = Tokenizer(num_words=maxwords, ) # 품사 전처리 한 결과(2.2)를 띄어쓰기 기준으로 토큰화
tokenizer.fit_on_texts(Text_train) 
sequences = tokenizer.texts_to_sequences(Text_train) #인코딩된 정수를 Text_train에 대입, 빈도수 상위 15000개까지만 포함된 상태

## 확인용 출력 >> 나중에 지워버리기
for i in range(3):
    print(Text_train[i])
    print('문장 바꿈')
    
sequences[:3]
'''


# In[16]:


print(tokenizer.word_index)


# In[ ]:


'''
print('문장의 최대 길이 :',max(len(l) for l in sequences))
print('문장의 평균 길이 :',sum(map(len, sequences))/len(sequences))
plt.hist([len(s) for s in sequences], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import urllib.request

max_len = 113  ## 일단 최댓값으로 패딩
Text_train = pad_sequences(sequences, maxlen = max_len)

print('전체 데이터의 크기(shape):', Text_train.shape)
'''


# In[ ]:


'''
# label 인코딩
idx_encode = preprocessing.LabelEncoder()  # 사이킷런 툴임
idx_encode.fit(Label_train)

Label_train = idx_encode.transform(Label_train) # 주어진 고유한 정수로 변환
Label_test = idx_encode.transform(Label_test) # 고유한 정수로 변환

label_idx = dict(zip(list(idx_encode.classes_), idx_encode.transform(list(idx_encode.classes_))))

print(label_idx)

print(Label_train[2])

#Label_train = to_categorical(np.asarray(Label_train))
#print('레이블 데이터의 크기(shape):', Label_train.shape)
'''


# In[ ]:


'''
ValueError: You are passing a target array of shape (11151, 1) while using as loss `categorical_crossentropy`. `categorical_crossentropy` expects targets to be binary matrices (1s and 0s) of shape (samples, classes). If your targets are integer classes, you can convert them to the expected format via:
```
from keras.utils import to_categorical
y_binary = to_categorical(y_int)
```

Alternatively, you can use the loss function `sparse_categorical_crossentropy` instead, which does expect integer targets.

'''


# In[ ]:


'''# label 인코딩
idx_encode = preprocessing.LabelEncoder()  # 사이킷런 툴임
idx_encode.fit(Label_train)

Label_train = idx_encode.transform(Label_train) # 주어진 고유한 정수로 변환
Label_test = idx_encode.transform(Label_test) # 고유한 정수로 변환

label_idx = dict(zip(list(idx_encode.classes_), idx_encode.transform(list(idx_encode.classes_))))
print(label_idx)


Label_train = to_categorical(np.asarray(Label_train))
print('레이블 데이터의 크기(shape):', Label_train.shape)'''


# In[ ]:


'''# 무작위 데이터 인코딩 된 결과 확인하기
print(Text_train[0])
print(Label_train[0])
print()
print()
print(Text_train.shape)
'''


# In[ ]:


'''from tensorflow.keras.layers import SimpleRNN, Embedding, Dense,Dropout
from tensorflow.keras.models import Sequential

def fit_and_evaluate(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Dense(128, input_shape=(128,), activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(9, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=64, epochs=5, verbose=2, validation_split=0.1)
    score = model.evaluate(X_test, y_test, batch_size=64, verbose=0)
    return score[1]
    
    
score = fit_and_evaluate(Text_train, Label_train, Text_test, Label_test) # 모델을 훈련하고 평가.
print( score)
    '''


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




