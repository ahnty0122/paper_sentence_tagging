#-*-coding:utf-8-sig-*- 
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from simple_ntc.trainer import Trainer
from simple_ntc.data_loader import DataLoader

from simple_ntc.models.rnn import RNNClassifier
from simple_ntc.models.cnn import CNNClassifier

## train.py 는 가장 큰 개념. train.py 안에 data loader 와 trainer 의 상호작용이 있음.
## data loader 에서 mini batch 단위로 데이터를 trainer 에 넘겨주면
## model 에서 cnn, rnn 을 받아서 trainer 에서 train, valid 로 나눠 트레이닝 시키는 것임.


def define_argparser():
    '''
    Define argument parser to set hyper-parameters.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)
    ## 2는 iteration 마다 출력, 1은 에폭마다

    p.add_argument('--min_vocab_freq', type=int, default=3)
    ## 몇번 이상 나오는 단어만 학습하도록. 
    ## 원래는 5였는데 태영이 3으로 수정함.
    p.add_argument('--max_vocab_size', type=int, default=999999)
    ## vocab 사이즈 제어 가능. 

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=10)

    p.add_argument('--word_vec_size', type=int, default= 256)
    p.add_argument('--dropout', type=float, default=.3)

    p.add_argument('--max_length', type=int, default=256)
    ## 256까지만. 나머지는 잘라버림.
    p.add_argument('--rnn', action='store_true')
    p.add_argument('--hidden_size', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=4)

    p.add_argument('--cnn', action='store_true')
    p.add_argument('--use_batch_norm', action='store_true')
    p.add_argument('--window_sizes', type=int, nargs='*', default=[3, 4, 5])
    p.add_argument('--n_filters', type=int, nargs='*', default=[100, 100, 100])

    ## rnn, cnn 동시 학습가능.
    ## classify.py 에서 앙상블함. 
    ## rnn 은 전체 문장의 흐름. context 를 많이 보고
    ## cnn 은 실제 문구나 절, 단어의 패턴이 있나 없나를 보기 때문에. 앙상블시킴.
    
    
    config = p.parse_args() ## config 받도록.

    return config


def main(config):
    loaders = DataLoader(
        train_fn=config.train_fn,
        batch_size=config.batch_size,
        min_freq=config.min_vocab_freq,
        max_vocab=config.max_vocab_size,
        device=config.gpu_id
    )

    print(
        '|train| =', len(loaders.train_loader.dataset),
        '|valid| =', len(loaders.valid_loader.dataset),
    ) ## 여기서 문장 개수를 알 수 있음.
    
    vocab_size = len(loaders.text.vocab)  ## loaders 안에서 text set 의 vocab (train set 기준)
    n_classes = len(loaders.label.vocab)
    print('|vocab| =', vocab_size, '|classes| =', n_classes)
    
    if config.rnn is False and config.cnn is False:
        raise Exception('You need to specify an architecture to train. (--rnn or --cnn)')

    if config.rnn:
        # Declare model and loss.
        model = RNNClassifier(
            input_size=vocab_size,
            word_vec_size=config.word_vec_size,
            hidden_size=config.hidden_size,
            n_classes=n_classes,
            n_layers=config.n_layers,
            dropout_p=config.dropout,
        )
        optimizer = optim.Adam(model.parameters())
        crit = nn.NLLLoss()
        print(model)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        rnn_trainer = Trainer(config)
        rnn_model = rnn_trainer.train(
            model,
            crit,
            optimizer,
            loaders.train_loader,
            loaders.valid_loader
        )
    if config.cnn:
        # Declare model and loss.
        model = CNNClassifier(
            input_size=vocab_size,
            word_vec_size=config.word_vec_size,
            n_classes=n_classes,
            use_batch_norm=config.use_batch_norm,
            dropout_p=config.dropout,
            window_sizes=config.window_sizes,
            n_filters=config.n_filters,
        )
        optimizer = optim.Adam(model.parameters())
        crit = nn.NLLLoss()
        print(model)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        cnn_trainer = Trainer(config)
        cnn_model = cnn_trainer.train(
            model,
            crit,
            optimizer,
            loaders.train_loader,
            loaders.valid_loader
        )

    torch.save({
        'rnn': rnn_model.state_dict() if config.rnn else None,
        'cnn': cnn_model.state_dict() if config.cnn else None,
        'config': config,
        'vocab': loaders.text.vocab,
        'classes': loaders.label.vocab,
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
