from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pprint import pprint
from konlpy.tag import Mecab

import sys
import pickle
import os
import codecs
import argparse
from collections import Counter
from threading import Thread

PAD = "<pad>"
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK = "<unk>"
NUM = "<num>"
NONE = "0"
CLS = "[CLS]"

mecab = Mecab()


class Vocabulary(object):
    """Vocab Class"""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def transform_word2idx(self, word):
        try:
            return self.word2idx[word]
        except:
            print("key error: "+ str(word))
            word = UNK
            return self.word2idx[word]

    def transform_idx2word(self, idx):
        try:
            return self.idx2word[idx]
        except:
            print("key error: " + str(idx))
            idx = self.word2idx[UNK]
            return self.idx2word[idx]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(text_list, threshold=1, vocab_path="./data_in/word_vocab.pkl", tokenizer_type="mecab"):
    """Build a word vocab"""

    def do_concurrent_tagging(start, end, text_list, counter):
        for i, text in enumerate(text_list[start:end]):
            text = text.strip()
            text = text.lower()

            try:
                if tokenizer_type == "mecab":
                    tokens_ko = mecab.pos(text)
                    tokens_ko = [str(pos[0]) + '/' + str(pos[1]) for pos in tokens_ko]

                counter.update(tokens_ko)

                if i % 1000 == 0:
                    print("[%d/%d (total: %d)] Tokenized input text." % (
                    start + i, start + len(text_list[start:end]), len(text_list)))

            except Exception as e:  # OOM, Parsing Error
                print(e)
                continue

    counter = Counter()

    num_thread = 4
    thread_list = []
    n_x_text = len(text_list)
    for i in range(num_thread):
        thread_list.append(Thread(target=do_concurrent_tagging, args=(
        int(i * n_x_text / num_thread), int((i + 1) * n_x_text / num_thread), text_list, counter)))

    for thread in thread_list:
        thread.start()

    for thread in thread_list:
        thread.join()

    print(counter.most_common(10))  # print most common words
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word(PAD)
    vocab.add_word(START_TOKEN)
    vocab.add_word(END_TOKEN)
    vocab.add_word(UNK)
    vocab.add_word(CLS)

    for i, word in enumerate(words):
        vocab.add_word(str(word))

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    return vocab

def word_to_token(X_str_batch):
    X_token_batch = [mecab.pos(X_str,join=True) for X_str in X_str_batch]
    return X_token_batch

def token_to_word_ids(X_token_batch, vocab):
    X_ids_batch = []
    for X_tokens in X_token_batch:
        X_ids_batch.append([vocab.transform_word2idx(X_token) for X_token in X_tokens])
    return X_ids_batch

def word_ids_to_pad_word_ids(X_ids_batch, vocab, maxlen):
    padded_X_ids_batch = keras.preprocessing.sequence.pad_sequences(X_ids_batch,
                                                            value=vocab.transform_word2idx(PAD),
                                                            padding='post',
                                                            maxlen=maxlen)
    return np.array(padded_X_ids_batch)

def word_to_pad_word_ids(text_batch, vocab, maxlen, add_start_end_token=False):
    X_token_batch = word_to_token(text_batch)

    if add_start_end_token is True:
        X_start_end_token_batch = [[START_TOKEN] + X_token + [END_TOKEN] for X_token in X_token_batch]
        X_start_end_ids_batch = token_to_word_ids(X_start_end_token_batch, vocab)
        pad_X_start_end_ids_batch = word_ids_to_pad_word_ids(X_start_end_ids_batch, vocab, maxlen)

        target_input_token_batch = [[START_TOKEN] + X_token for X_token in X_token_batch]
        target_real_token_batch = [X_token + [END_TOKEN] for X_token in X_token_batch]

        target_input_ids_batch = token_to_word_ids(target_input_token_batch, vocab)
        pad_target_input_ids_batch = word_ids_to_pad_word_ids(target_input_ids_batch, vocab, maxlen)

        target_real_ids_batch = token_to_word_ids(target_real_token_batch, vocab)
        pad_target_real_ids_batch = word_ids_to_pad_word_ids(target_real_ids_batch, vocab, maxlen)
        return pad_X_start_end_ids_batch, pad_target_input_ids_batch, pad_target_real_ids_batch

    X_ids_batch = token_to_word_ids(X_token_batch, vocab)
    pad_X_ids_batch = word_ids_to_pad_word_ids(X_ids_batch, vocab, maxlen)
    return pad_X_ids_batch

def decode_word_ids(word_ids_batch, vocab):
    word_token_batch = []
    for word_ids in word_ids_batch:
        word_token = [vocab.transform_idx2word(word_id) for word_id in word_ids]
        word_token_batch.append(word_token)
    return word_token_batch

    # return ' '.join([reverse_word_index.get(i, '?') for i in text])

def main():
    print("Data_loader")



if __name__ == '__main__':
    main()