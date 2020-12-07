import sys
import argparse
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data

from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
import logging
logging.basicConfig(level=logging.ERROR)


def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--top_k', type=int, default=1)

    config = p.parse_args()

    return config


def read_text():
    text = pd.read_csv("./simple_ntc/data/test_token.tsv", delimiter='\t', names=['tag', 'sentence'])
    text2 = text['sentence']

    lines =[]

    for line in text2:
        if line.strip() != '':
            lines += [line.strip()]

    label = text['tag']

    return lines, label


def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    train_config = saved_data['config']
    bert_best = saved_data['bert']
    index_to_label = saved_data['classes']

    lines, label = read_text()
    with torch.no_grad():
        # Declare model and load pre-trained weights.
        tokenizer = AutoTokenizer.from_pretrained(train_config.pretrained_model_name)
        model = BertForSequenceClassification.from_pretrained(
            train_config.pretrained_model_name,
            num_labels=len(index_to_label)
        )
        model.load_state_dict(bert_best)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
        device = next(model.parameters()).device

        # Don't forget turn-on evaluation mode.
        model.eval()

        y_hats = []
        for idx in range(0, len(lines), config.batch_size):
            mini_batch = tokenizer(
                lines[idx:idx + config.batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            x = mini_batch['input_ids']
            x = x.to(device)
            mask = mini_batch['attention_mask']
            mask = mask.to(device)

            # Take feed-forward
            y_hat = F.softmax(model(x, attention_mask=mask)[0], dim=-1)

            y_hats += [y_hat]
        # Concatenate the mini-batch wise result
        y_hats = torch.cat(y_hats, dim=0)
        # |y_hats| = (len(lines), n_classes)

        probs, indice = y_hats.cpu().topk(config.top_k)
        # |indice| = (len(lines), top_k)
        correct = 0
        for i in range(len(lines)):
            if [label[i]] == [index_to_label[int(indice[i][j])] for j in range(config.top_k)]:
                correct += 1
        print('Accuracy: %f %%' % (100 * correct / len(lines)))        

if __name__ == '__main__':
    config = define_argparser()
    main(config)
