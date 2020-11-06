import re
import pandas as pd
import csv
import numpy as np
import json
from khaiii import KhaiiiApi
import random
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import urllib.request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def clean_text(text): # 정규표현식으로 문장 띄어쓰기 교정

    corpus = []
    for i in text:
        review = re.sub(r'\s+', ' ', i) #remove extra space
        review = re.sub(r'\s+', ' ', review) #remove spaces
        review = re.sub(r"^\s+", '', review) #remove space from start
        review = re.sub(r'\s+$', '', review) #remove space from the end
        corpus.append(review)
    return corpus

def pos_text(Text_train): # 필요한 품사 지정 후 단어 추출 (카카오 형태소 분석기 사용)
    corpus = []
    api = KhaiiiApi()
    tags = ['NNG', 'NNP', 'NNB', 'VV', 'VA', 'VX', 'MAG', 'MAJ', 'XSV', 'XSA']
    for i in Text_train:
        pos_tagged = ''
        for word in api.analyze(i):
            for morph in word.morphs:
                if morph.tag in tags:
                    pos_tagged += morph.lex + ' ' 
        corpus.append(pos_tagged.strip())
    return corpus

def load_data_and_write_to_file(data_file, train_data_file, test_data_file, test_sample_percentage): # 원래 data 불러와서 textdata, labeldata로 나누기

    data = pd.read_csv(data_file)
    TextData = np.array(data.iloc[:,2])
    LabelData = np.array(data.iloc[:,1])

    TextData = clean_text(TextData) # 정규표현식으로 제거
    TextData = pos_text(TextData)
    TextData = np.array(TextData)
    x_text = TextData
    y = LabelData
    
    s = np.arange(TextData.shape[0])
    np.random.shuffle(s)

    TextData = TextData[s]
    #print ("셔플 한 뒤의 문장 배열: ", TextData)
    LabelData = LabelData[s]
    #print ("셔플 한 뒤의 라벨 배열: ", LabelData) 

    x_train, x_test = train_test_split(TextData, test_size=0.3, shuffle=False)
    y_train, y_test = train_test_split(LabelData, test_size=0.3, shuffle=False)

    # Write to CSV file
    with open(train_data_file, 'w', newline='', encoding='utf-8-sig') as f:
        print('Write train data to {} ...'.format(train_data_file))
        writer = csv.writer(f)
        writer.writerows(zip(x_train, y_train))
    with open(test_data_file, 'w', newline='', encoding='utf-8-sig') as f:
        print('Write test data to {} ...'.format(test_data_file))
        writer = csv.writer(f)
        writer.writerows(zip(x_test, y_test))


def preprocess(data_file, vocab_file, padding_size, test=False): # 단어를 숫자로 바꾸기--임베딩 과정
    print("Loading data from {} ...".format(data_file))
    df = pd.read_csv(data_file, header=None, names=["x_text", "y_label"])
    TextData, LabelData = df["x_text"].tolist(), df["y_label"].tolist()

    if not test:
        # Texts to sequences
        TextData = np.array(TextData)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(TextData)
        x = tokenizer.texts_to_sequences(TextData)
        word_dict = tokenizer.word_index
        #json.dump(word_dict, open(vocab_file, 'w'), ensure_ascii=False)
        vocab_size = len(word_dict) + 1
        # max_doc_length = max([len(each_text) for each_text in x])
        x = pad_sequences(x, maxlen=padding_size,
                                                 padding='post', truncating='post')
        idx_encode = preprocessing.LabelEncoder()  # 사이킷런 툴임
        idx_encode.fit(LabelData)

        LabelData = idx_encode.transform(LabelData) # 고유한 정수로 변환
        label_idx = dict(zip(list(idx_encode.classes_), idx_encode.transform(list(idx_encode.classes_))))
        print(label_idx)
        y = to_categorical(np.asarray(LabelData))
        print("단어 집합의 크기: {:d}".format(vocab_size))
        print("Shape of train data: {}".format(np.shape(x)))
        print('레이블 데이터의 크기(shape):', y.shape)
        return x, y, vocab_size
    else:
        #word_dict = json.load(open(vocab_file, 'r'))
        tokenizer = Tokenizer()
        TextData = np.array(TextData)
        tokenizer.fit_on_texts(TextData)
        x = tokenizer.texts_to_sequences(TextData)
        word_dict = tokenizer.word_index
        vocabulary = word_dict.keys()
        #x = [[word_dict[each_word] if each_word in vocabulary else 1 for each_word in each_sentence.split()] for each_sentence in TextData]
        x = pad_sequences(x, maxlen=padding_size,
                                                 padding='post', truncating='post')
        idx_encode = preprocessing.LabelEncoder()  # 사이킷런 툴임
        idx_encode.fit(LabelData)

        LabelData = idx_encode.transform(LabelData) # 고유한 정수로 변환
        label_idx = dict(zip(list(idx_encode.classes_), idx_encode.transform(list(idx_encode.classes_))))
        print(label_idx)
        y = to_categorical(np.asarray(LabelData))
        
        print("Shape of test data: {}\n".format(np.shape(x)))
        return x, y