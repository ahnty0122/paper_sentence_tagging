import torch.nn as nn


class RNNClassifier(nn.Module):
 
    def __init__(
        self,
        input_size, ## vocab 의 size, 지정해줄 필요 없이 torchtext가 자동으로 읽어옴
        ## onehot vector를 위해 필요함
        word_vec_size, ## word embedding vector 가 몇차원으로 projection 될거냐
        hidden_size, ## 위의 임베딩 벡터 받아서 모델에 넣을건데, 그 모델의 hidden size
        n_classes, ## 최종적으로 class 의 갯수
        n_layers=4, ## bi-directional LSTM 사용할건데, 거기에 몇 개의 layer 쌓을건지
        dropout_p=.3, ## LSTM 의 layer 와 layer 사이에 dropout 얼마나 줄건지
    ):
        self.input_size = input_size  # vocabulary_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.emb = nn.Embedding(input_size, word_vec_size)
        ## 임베딩 레이어는 linear 레이어와 수학적으로 같은데,
        ## input_size 를 받아서 word_vec_size 로 뱉어낸다.
        ## word size 받아서 몇차원의 word_vec 로 임베딩할것인지
        
                ## RNN 선언
        self.rnn = nn.LSTM( ## 우리는 LSTM 사용할것임. 아래는 LSTM hyperparameter 들
            input_size=word_vec_size, ## word embedding 거쳐서 나온 것을 넣어주니까
            hidden_size=hidden_size, ## LSTM 안의 hidden layer
            num_layers=n_layers,
            dropout=dropout_p,
            batch_first=True, ## 처음 값을 batch size로 받아오기 ( , , , ) 이 인풋의 처음값
            bidirectional=True, ## non auto regressive 이니까.
        )
        
        self.generator = nn.Linear(hidden_size * 2, n_classes)
        # softmax 전 차원축소해주기
        # bi directional 이니 2배해주고, 결국 n_classes 개로 차원축소
        # We use LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        self.activation = nn.LogSoftmax(dim=-1)
        # 그냥 softmax 가 아닌 log softmax 를 사용
        # Negative log likelihood loss 사용하게됨. 속도 조금 더 빨라짐

    def forward(self, x):
        # |x| = (batch_size, length)
        x = self.emb(x)
        # |x| = (batch_size, length, word_vec_size)
        x, _ = self.rnn(x)
        # |x| = (batch_size, length, hidden_size * 2)
        y = self.activation(self.generator(x[:, -1]))
        # |y| = (batch_size, n_classes)
        # -1 을 통해서 length 의 마지막 값만 가져올 수 있다

        return y
