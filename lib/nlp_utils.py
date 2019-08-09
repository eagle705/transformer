import sys
import pickle
import os
import codecs
import argparse
from collections import Counter
import numpy as np

from konlpy.tag import Kkma
from konlpy.tag import Twitter
from konlpy.tag import Mecab
from threading import Thread
import jpype
import re
from lib.custom_logger import CustomLogger, DailyCustomLogger



import logging

PAD = "<pad>"
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK = "<unk>"
NUM = "<num>"
NONE = "0"
CLS = "[CLS]"

mecab = Mecab()
kkma = Kkma()
twitter = Twitter()


logger = CustomLogger.__call__(__file__, "nlp_utils", log_level=logging.DEBUG).get_logger()
# logger = CustomLogger.__call__(__file__, log_level=logging.DEBUG).get_logger()

############### <Vocab> ###################

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
            logger.debug("key error: "+ str(word))
            word = UNK
            return self.word2idx[word]

    def transform_idx2word(self, idx):
        try:
            return self.idx2word[idx]
        except:
            logger.debug("key error: " + str(idx))
            idx = self.word2idx[UNK]
            return self.idx2word[idx]

    def __len__(self):
        return len(self.word2idx)


def update_word_pos_vocab(text_list, threshold, base_word_vocab, base_pos_vocab, word_vocab_path, pos_vocab_path, tokenizer_type="mecab"):
    """Update a word & pos vocab"""

    def do_concurrent_tagging(start, end, text_list, word_counter, pos_counter):
        jpype.attachThreadToJVM()
        for i, text in enumerate(text_list[start:end]):
            text = text.strip()
            text = text.lower()

            try:
                if tokenizer_type == "mecab":
                    tokens_ko = mecab.pos(text)
                else:
                    tokens_ko = twitter.pos(text, norm=True)  # , stem=True)

                word_tokens_ko = [str(pos[0]) for pos in tokens_ko]
                pos_tokens_ko = [str(pos[1]) for pos in tokens_ko]
                word_counter.update(word_tokens_ko)
                pos_counter.update(pos_tokens_ko)

                if i % 1000 == 0:
                    print("[%d/%d (total: %d)] Tokenized input text." % (
                    start + i, start + len(text_list[start:end]), len(text_list)))

            except Exception as e:  # for Out of memory
                print(e)
                continue

    word_counter = Counter()
    pos_counter = Counter()

    num_thread = 4
    thread_list = []
    n_x_text = len(text_list)
    for i in range(num_thread):
        thread_list.append(Thread(target=do_concurrent_tagging, args=(
        int(i * n_x_text / num_thread), int((i + 1) * n_x_text / num_thread), text_list, word_counter, pos_counter)))

    for thread in thread_list:
        thread.start()

    for thread in thread_list:
        thread.join()

    print(word_counter.most_common(10))  # print most common words

    ## Word
    words = [word for word, cnt in word_counter.items() if cnt >= threshold]

    for i, word in enumerate(words):
        base_word_vocab.add_word(str(word))
    updated_word_vocab = base_word_vocab

    with open(word_vocab_path, 'wb') as f:
        pickle.dump(updated_word_vocab, f)

    ## Pos
    poss = [pos for pos, cnt in pos_counter.items() if cnt >= 0]  # insert all pos tag

    for i, pos in enumerate(poss):
        base_pos_vocab.add_word(str(pos))
    updated_pos_vocab = base_pos_vocab

    with open(pos_vocab_path, 'wb') as f:
        pickle.dump(updated_pos_vocab, f)

    return updated_word_vocab, updated_pos_vocab

def build_vocab(text_list, threshold, vocab_path="word_vocab.pkl", with_pos=True, tokenizer_type="mecab"):
    """Build a word vocab"""

    def do_concurrent_tagging(start, end, text_list, counter):
        jpype.attachThreadToJVM()
        for i, text in enumerate(text_list[start:end]):
            text = text.strip()
            text = text.lower()

            try:
                if tokenizer_type == "mecab":
                    tokens_ko = mecab.pos(text)
                else:
                    tokens_ko = twitter.pos(text, norm=True)  # , stem=True)
                if with_pos is True:
                    tokens_ko = [str(pos[0]) + '/' + str(pos[1]) for pos in tokens_ko]
                else:
                    tokens_ko = [str(pos[0]) for pos in tokens_ko]
                counter.update(tokens_ko)

                if i % 1000 == 0:
                    logger.info("[%d/%d (total: %d)] Tokenized input text." % (
                    start + i, start + len(text_list[start:end]), len(text_list)))

            except Exception as e:  # for Out of memory
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


def build_word_pos_vocab(text_list, threshold, word_vocab_path="word_vocab.pkl", pos_vocab_path="pos_vocab.pkl", tokenizer_type="mecab"):
    """Build a word and pos vocab"""

    def do_concurrent_tagging(start, end, text_list, word_counter, pos_counter):
        jpype.attachThreadToJVM()
        for i, text in enumerate(text_list[start:end]):
            text = text.strip()
            text = text.lower()

            try:
                if tokenizer_type == "mecab":
                    tokens_ko = mecab.pos(text)
                else:
                    tokens_ko = twitter.pos(text, norm=True)  # , stem=True)

                word_tokens_ko = [str(pos[0]) for pos in tokens_ko]
                pos_tokens_ko = [str(pos[1]) for pos in tokens_ko]
                word_counter.update(word_tokens_ko)
                pos_counter.update(pos_tokens_ko)

                if i % 1000 == 0:
                    print("[%d/%d (total: %d)] Tokenized input text." % (
                    start + i, start + len(text_list[start:end]), len(text_list)))

            except Exception as e:  # for Out of memory
                print(e)
                continue

    word_counter = Counter()
    pos_counter = Counter()

    num_thread = 4
    thread_list = []
    n_x_text = len(text_list)
    for i in range(num_thread):
        thread_list.append(Thread(target=do_concurrent_tagging, args=(
        int(i * n_x_text / num_thread), int((i + 1) * n_x_text / num_thread), text_list, word_counter, pos_counter)))

    for thread in thread_list:
        thread.start()

    for thread in thread_list:
        thread.join()

    print(word_counter.most_common(10))  # print most common words

    ## Word
    words = [word for word, cnt in word_counter.items() if cnt >= threshold]

    word_vocab = Vocabulary()
    word_vocab.add_word(PAD)
    word_vocab.add_word(START_TOKEN)
    word_vocab.add_word(END_TOKEN)
    word_vocab.add_word(UNK)
    word_vocab.add_word(CLS)

    for i, word in enumerate(words):
        word_vocab.add_word(str(word))

    with open(word_vocab_path, 'wb') as f:
        pickle.dump(word_vocab, f)

    ## Pos
    poss = [pos for pos, cnt in pos_counter.items() if cnt >= 0]  # insert all pos tag
    pos_vocab = Vocabulary()
    pos_vocab.add_word(PAD)
    pos_vocab.add_word(START_TOKEN)
    pos_vocab.add_word(END_TOKEN)
    pos_vocab.add_word(UNK)
    pos_vocab.add_word(CLS)

    for i, pos in enumerate(poss):
        pos_vocab.add_word(str(pos))

    with open(pos_vocab_path, 'wb') as f:
        pickle.dump(pos_vocab, f)

    return word_vocab, pos_vocab


def build_char_vocab(text_list, threshold, vocab_path="char_vocab.pkl"):
    """Build a char vocab"""

    def do_concurrent_collection(start, end, text_list, counter):
        for i, text in enumerate(text_list[start:end]):
            text = text.strip()
            text = text.lower()

            try:
                tokens_ko = [_text for _text in text]  # characters
                counter.update(tokens_ko)

                if i % 1000 == 0:
                    print("[%d/%d (total: %d)] Tokenized input text." % (
                    start + i, start + len(text_list[start:end]), len(text_list)))

            except Exception as e:  # for Out of memory
                print(e)
                continue

    counter = Counter()

    num_thread = 4
    thread_list = []
    n_x_text = len(text_list)
    for i in range(num_thread):
        thread_list.append(Thread(target=do_concurrent_collection, args=(
        int(i * n_x_text / num_thread), int((i + 1) * n_x_text / num_thread), text_list, counter)))

    for thread in thread_list:
        thread.start()

    for thread in thread_list:
        thread.join()

    print(counter.most_common(10))  # print most common words
    chars = [char for char, cnt in counter.items() if cnt >= threshold]

    vocab_char = Vocabulary()
    vocab_char.add_word(PAD)
    vocab_char.add_word(START_TOKEN)
    vocab_char.add_word(END_TOKEN)
    vocab_char.add_word(UNK)
    vocab_char.add_word(CLS)

    for i, char in enumerate(chars):
        vocab_char.add_word(str(char))

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_char, f)

    return vocab_char


############### </Vocab> ###################


############### <Word_embedding> ###################
def build_word2vec(x_text_lines, word2vec_dim, word2vec_path, vocab_path, embedding_weight_path="embedding_weight.pkl", USING_POS_DOC=False):
    from gensim.models import word2vec
    import multiprocessing
    import time

    word2vec_dim = word2vec_dim  # 100 # args

    config = {
        'min_count': 1,  # 3,  # 등장 횟수가 5 이하인 단어는 무시
        'size': word2vec_dim,  # 50차원짜리 벡터스페이스에 embedding
        'sg': 1,  # 0이면 CBOW, 1이면 skip-gram을 사용
        'batch_words': 1000,  # 사전을 구축할때 한번에 읽을 단어 수
        'iter': 3,  # 7,  # 보통 딥러닝에서 말하는 epoch과 비슷한, 반복 횟수를 의미 #너무 오래 걸릴땐 좀 낮춰야
        'workers': multiprocessing.cpu_count()  # 윈도우에서 에러
    }

    docs_ko = []

    if USING_POS_DOC is False:
        # 형태소 분석이 안 된 corpus에 대해서 형태소 분석 후 word2vec계산
        for x_item in x_text_lines:
            x_item = x_item.strip()
            x_item = x_item.lower()

            try:
                # ToDo: Korean
                tokens_ko = mecab.pos(x_item)
                # tokens_ko = twitter.pos(x_item, norm=True)  # , stem=True)
                tokens_ko = [str(pos[0]) + '/' + str(pos[1]) for pos in tokens_ko]
                docs_ko.append(tokens_ko)
            except Exception as e:  # for Out of memory
                print(e)
                continue
    else:
        # 이미 형태소 분석이 된 corpus에 대해서 word2vec 계산
        docs_ko = x_text_lines  # In this case, x_pos_lines must be prepared

    wv_model_ko = word2vec.Word2Vec(**config)
    count_t = time.time()

    wv_model_ko.build_vocab(docs_ko)
    print("wv_model_ko.corpus_count: ", wv_model_ko.corpus_count)
    wv_model_ko.train(docs_ko, total_examples=wv_model_ko.corpus_count, epochs=3)
    print('Running Time : %.02f' % (time.time() - count_t))

    # save model
    wv_model_ko.save(word2vec_path)

    # create vocab
    vocab = Vocabulary()
    vocab.add_word(PAD)  # '<pad>'
    vocab.add_word(START_TOKEN)  # '<s>'
    vocab.add_word(END_TOKEN)  # '</s>'
    vocab.add_word(UNK)  # '<unk>'
    vocab.add_word(CLS)  # '[CLS]'
    special_token_num = len(vocab)

    for index, word in enumerate(wv_model_ko.wv.index2word):
        vocab.add_word(word)

    # save vocab
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    # extract word embedding array & label
    # embedding_weights = wv_model_ko.wv.syn0
    # final_embeddings = embedding_weights
    # labels = wv_model_ko.wv.index2word


    # extract word embedding array & label
    word2vec_matrix = wv_model_ko.wv.syn0
    token_matrix = np.zeros((special_token_num, word2vec_matrix.shape[1]))  # ToDo: 나중에 코드 다듬어야함
    word2vec_matrix = np.concatenate((token_matrix, word2vec_matrix), axis=0).astype('float32')  # embedding_look_up table 조건
    labels = list(vocab.word2idx.keys())
    final_embeddings = word2vec_matrix

    # save embedding weight
    with open(embedding_weight_path, 'wb') as f:
        pickle.dump(word2vec_matrix, f)

    # Plotting
    def plot_with_labels(low_dim_embs, labels, filename='tsne_' + str(word2vec_dim) + '.png'):
        import matplotlib
        matplotlib.use('Agg')

        # font 설정
        import matplotlib.pyplot as plt
        from matplotlib import font_manager, rc

        print("font_list: ", font_manager.get_fontconfig_fonts())
        font_name = font_manager.FontProperties(fname='/Library/Fonts/NanumSquareBold.ttf').get_name()
        rc('font', family=font_name)

        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

        plt.savefig(filename)

    # t-SNE embedding
    def build_t_SNE_plot(final_embeddings, labels, sample_num = 500):
        try:
            from sklearn.manifold import TSNE
            import random
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            random_index = random.sample(range(0, len(final_embeddings)), sample_num)
            tsne_representation = tsne.fit_transform([final_embeddings[i, :] for i in random_index])
            labels = [labels[i] for i in random_index]
            plot_with_labels(tsne_representation, labels)

            return tsne_representation, labels, random_index
        except ImportError:
            print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")

    build_t_SNE_plot(final_embeddings, labels)

    return word2vec_matrix, vocab


# 수정해야함
def load_word2vec(pretrained_word2vec_file):
    """
    word2vec 읽어오는 함수
    :param pretrained_word2vec_file: 학습된 word2vec 파일
    :return:
    """
    from gensim.models import word2vec
    wv_model_ko = word2vec.Word2Vec.load(pretrained_word2vec_file)

    # vocab = Vocabulary()
    # vocab.add_word(PAD) # '<pad>'
    # vocab.add_word(START_TOKEN) # '<s>'
    # vocab.add_word(END_TOKEN) # '</s>'
    # vocab.add_word(UNK) # '<unk>'
    # for index, word in enumerate(wv_model_ko.wv.index2word):
    #     vocab.add_word(word)

    word2vec_matrix = wv_model_ko.wv.syn0
    # print(type(word2vec_matrix)) # <class 'numpy.ndarray'>
    # print(word2vec_matrix.shape) # (71764, 100)

    # <pad>, <s>, </s>, <unk> 토큰등을 zero_vector로 init (추후 추가해야함)
    token_matrix = np.zeros((4, word2vec_matrix.shape[1]))  # ToDo: 나중에 코드 다듬어야함
    word2vec_matrix = np.concatenate((token_matrix, word2vec_matrix), axis=0).astype('float32')  # embedding_look_up table 조건
    # print(word2vec_matrix.dtype) # float32

    return word2vec_matrix


def load_vocab(vocab_path):
    """
    vocab 읽어오기
    :param vocab_path: 사전 파일 path
    :return:
    """
    vocab = None

    try:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
            return vocab
    except Exception as e:
        logger.debug("exception in load_vocab! return None type vocab %s", e)
        return vocab



############### </Word_embedding> ###################

############### <POS> ###################
def pos_text(text, tokenizer_type="mecab", remove_josa=False):
    try:
        text = text.strip().lower()
        if tokenizer_type == "mecab":
            tokens_ko = mecab.pos(text)
            _tokens_ko = []
            if remove_josa is True:
                for word in tokens_ko:
                    if word[1] not in ['JKS', 'JKC', 'JKG', 'JKO', 'JKM', 'JKI', 'JKQ', 'JC', 'JX']:
                        _tokens_ko.append(word)
                tokens_ko = _tokens_ko

        else:
            tokens_ko = twitter.pos(text, norm=True, stem=False)
            _tokens_ko = []
            if remove_josa is True:
                for word in tokens_ko:
                    if word[1] != 'Josa':
                        _tokens_ko.append(word)
                tokens_ko = _tokens_ko
        pos_document = ""
        for pos in tokens_ko:
            pos_document += str(pos[0]) + '/' + str(pos[1]) + ' '
    except:
        pos_document = text  # split이라도 해줘야되지 않나.. 체크~!

    return pos_document


def without_pos_text(text, tokenizer_type="mecab", remove_josa=False):
    pos_document = ""
    try:
        text = text.strip().lower()
        if tokenizer_type == "mecab":
            tokens_ko = mecab.pos(text)
            _tokens_ko = []
            if remove_josa is True:
                for word in tokens_ko:
                    if word[1] not in ['JKS', 'JKC', 'JKG', 'JKO', 'JKM', 'JKI', 'JKQ', 'JC', 'JX']:
                        _tokens_ko.append(word)
                tokens_ko = _tokens_ko
        else:
            tokens_ko = twitter.pos(text, norm=True, stem=False)
            _tokens_ko = []
            if remove_josa is True:
                for word in tokens_ko:
                    if word[1] != 'Josa':
                        _tokens_ko.append(word)
                tokens_ko = _tokens_ko

        # tokens_ko = twitter.pos(str(text), norm=True)#, stem=True)
        for pos in tokens_ko:
            pos_document += str(pos[0]) + ' '
    except:
        pos_document = text
    return pos_document


def without_pos_noun_verb_text(text):
    pos_document = ""
    try:
        text = text.strip().lower()
        tokens_ko = twitter.pos(text, norm=True, stem=False)
        _tokens_ko = []
        for word in tokens_ko:
            if word[1] == 'Noun':
                _tokens_ko.append(word)
            if word[1] == 'Verb':
                _tokens_ko.append(word)
            if word[1] == 'Alpha':
                _tokens_ko.append(word)
            if word[1] == 'Adjective':
                _tokens_ko.append(word)
        tokens_ko = _tokens_ko

        for pos in tokens_ko:
            pos_document += str(pos[0]) + ' '
    except:
        pos_document = text
    return pos_document


def word_pos_token_list(text, tokenizer_type="mecab", remove_josa=False):
    try:
        text = text.strip().lower()
        if tokenizer_type == "mecab":
            tokens_ko = mecab.pos(text)
            _tokens_ko = []
            if remove_josa is True:
                for word in tokens_ko:
                    if word[1] not in ['JKS', 'JKC', 'JKG', 'JKO', 'JKM', 'JKI', 'JKQ', 'JC', 'JX']:
                        _tokens_ko.append(word)
                tokens_ko = _tokens_ko
        else:
            tokens_ko = twitter.pos(text, norm=True, stem=False)
            _tokens_ko = []
            if remove_josa is True:
                for word in tokens_ko:
                    if word[1] != 'Josa':
                        _tokens_ko.append(word)
                tokens_ko = _tokens_ko
        # tokens_ko = twitter.pos(text, norm=True)#, stem=True)
        word_token_list = [str(pos[0]) for pos in tokens_ko]
        pos_token_list = [str(pos[1]) for pos in tokens_ko]
    except:
        word_token_list = []
        pos_token_list = []

    return word_token_list, pos_token_list


def word_with_pos_token_list(text, tokenizer_type="mecab", remove_josa=False):
    try:
        text = text.strip().lower()
        if tokenizer_type == "mecab":
            tokens_ko = mecab.pos(text)
            _tokens_ko = []
            if remove_josa is True:
                for word in tokens_ko:
                    if word[1] not in ['JKS', 'JKC', 'JKG', 'JKO', 'JKM', 'JKI', 'JKQ', 'JC', 'JX']:
                        _tokens_ko.append(word)
                tokens_ko = _tokens_ko
        else:
            tokens_ko = twitter.pos(text, norm=True, stem=False)
            _tokens_ko = []
            if remove_josa is True:
                for word in tokens_ko:
                    if word[1] != 'Josa':
                        _tokens_ko.append(word)
                tokens_ko = _tokens_ko
        # tokens_ko = twitter.pos(text, norm=True)#, stem=True)
        word_with_pos_token_list = [str(pos[0]) + '/' + str(pos[1]) for pos in tokens_ko]
    except:
        word_with_pos_token_list = []

    return word_with_pos_token_list


def pos_token_list(text, tokenizer_type="mecab", remove_josa=False):
    try:
        text = text.strip().lower()
        if tokenizer_type == "mecab":
            tokens_ko = mecab.pos(text)
            _tokens_ko = []
            if remove_josa is True:
                for word in tokens_ko:
                    if word[1] not in ['JKS', 'JKC', 'JKG', 'JKO', 'JKM', 'JKI', 'JKQ', 'JC', 'JX']:
                        _tokens_ko.append(word)
                tokens_ko = _tokens_ko
        else:
            tokens_ko = twitter.pos(text, norm=True, stem=False)
            _tokens_ko = []
            if remove_josa is True:
                for word in tokens_ko:
                    if word[1] != 'Josa':
                        _tokens_ko.append(word)
                tokens_ko = _tokens_ko
        # tokens_ko = twitter.pos(text, norm=True)#, stem=True)
        pos_token_list = [str(pos[1]) for pos in tokens_ko]
    except:
        pos_token_list = []

    return pos_token_list


def word_token_list(text, tokenizer_type="mecab", remove_josa=False):
    pos_document = ""
    try:
        text = text.strip().lower()
        if tokenizer_type == "mecab":
            tokens_ko = mecab.pos(text)
            _tokens_ko = []
            if remove_josa is True:
                for word in tokens_ko:
                    if word[1] not in ['JKS', 'JKC', 'JKG', 'JKO', 'JKM', 'JKI', 'JKQ', 'JC', 'JX']:
                        _tokens_ko.append(word)
                tokens_ko = _tokens_ko
        else:
            tokens_ko = twitter.pos(text, norm=True, stem=False)
            _tokens_ko = []
            if remove_josa is True:
                for word in tokens_ko:
                    if word[1] != 'Josa':
                        _tokens_ko.append(word)
                tokens_ko = _tokens_ko
        # tokens_ko = twitter.pos(str(text), norm=True)#, stem=True)
        word_token_list = [str(pos[0]) for pos in tokens_ko]
    except:
        word_token_list = []

    return word_token_list


############### </POS> ###################


############### <PAD> ###################
def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    elif nlevels == 2:
        # max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        max_length_word = 15
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word,
                                            max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)

    return sequence_padded, sequence_length


def pad_max_sequences(sequences, pad_tok, max_length, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        # max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    elif nlevels == 2:
        # max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        max_length_word = 15
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word,
                                            max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)

    return sequence_padded, sequence_length


from tensorflow import keras
def keras_pad_sequences(input_word_ids, PAD_TOKEN, padding="pre", maxlen=10):
    return keras.preprocessing.sequence.pad_sequences(input_word_ids,  # word ids
                                                      value=PAD_TOKEN,
                                                      padding=padding,  # post -> pre
                                                      maxlen=maxlen)


############### </PAD> ###################


############### <DataLoad> ###################
def load_corpus_only(data_file_dir, num_line=None, encoding="utf-8"):
    """
    :param data_file_dir:
    :return:
    """

    print("Loading data... from " + str(data_file_dir))
    text_list = []

    file_obj = codecs.open(data_file_dir, "r", encoding, errors='ignore')

    if num_line is None:
        lines = file_obj.readlines()  # ToDo: Memory Error Can be + codec can't decode byte 0x9f
    else:
        lines = [file_obj.readline() for i in range(num_line)]

    for line in lines:
        line = line.strip()

        if line == '':
            continue
        text_list.append(line)

    file_obj.close()

    return text_list


def load_corpus_only_from_csv(data_file_dir, num_line=None, encoding="utf-8"):
    """
    :param data_file_dir:
    :return:
    """
    import pandas as pd
    print("Loading data... from " + str(data_file_dir))
    text_list = []

    df = pd.read_csv(data_file_dir)
    df = df.fillna("")

    for i, (category, row) in enumerate(df.iterrows()):
        intent_msg = row[0]
        intent_name = row[1]
        text_list.append(intent_msg.strip().lower())
        text_list.append(intent_name.strip().lower())

    print(text_list)

    return text_list

############### </DataLoad> ###################

############### <DBConnect> ###################
def conn_db():
    import pymysql

    # MySQL Connection 연결
    # ToDo: 정보 수정해야함
    conn = pymysql.connect(host='172.31.30.52', user='test', password='test1234',
                           db='chabot', charset='utf8')

    return conn


def fectch_data():
    conn = conn_db()

    # Connection 으로부터 Cursor 생성
    curs = conn.cursor()

    f = open('smonwar_corpus.txt', 'w')
    sql_batch_size = 300

    for i in range(300):
        # SQL문 실행
        sql = "SELECT title, body FROM bbs_document_history LIMIT %d, %d" % (i * sql_batch_size, sql_batch_size)
        curs.execute(sql)

        # 데이타 Fetch
        rows = curs.fetchall()
        #     print(rows)

        for row in rows:
            text = row[0] + ' ' + row[1] + '\n'
            #         print(text)
            f.write(text)

    for i in range(10):
        # SQL문 실행
        sql = "SELECT title, body FROM blog_document_history LIMIT %d, %d" % (i * sql_batch_size, sql_batch_size)
        curs.execute(sql)

        # 데이타 Fetch
        rows = curs.fetchall()
        #     print(rows)
        for row in rows:
            text = row[0] + ' ' + row[1] + '\n'
            #         print(text)
            f.write(text)

    f.close()

    # Connection 닫기
    curs.close()
    conn.close()


############### </DBConnect> ###################


############### <Mini_batches> ###################
def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)
    Returns:
        list of tuples
    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


############### </Mini_batches> ###################

############### <Preprocessing> ###################


def remove_string_special_characters(text, same_word_dict=None):
    # replace speical char with ' '

    try:
        text = str(text)
        stripped = re.sub('[^\w\s]', '', text)  # 정규식 테스트
        stripped = re.sub('_', '', stripped)

        # change any whitespace to one space
        stripped = re.sub('\s+', ' ', stripped)

        # remove start and end white spaceds
        stripped = stripped.strip().lower()

        if same_word_dict is not None:
            for key, value in same_word_dict.items():
                stripped = re.sub(r"" + key, value, stripped)

            return stripped
        else:
            return stripped
    except:
        print("except in remove_string_special_characters_"+text)
        return text

############### </Preprocessing> ###################

############### <General_utils> ###################

def logit_to_one_hot(y_list, num_class):
    logit = np.array(y_list)
    one_hot = np.zeros((len(y_list), num_class))
    one_hot[np.arange(len(y_list)), logit] = 1
    return one_hot


############### </General_utils> ###################

def main(args):

    text_list = load_corpus_only_from_csv("../data_in/ChatbotData.csv")
    base_word_vocab = load_vocab("../data_in/word_vocab_190408.pkl")
    base_pos_vocab = load_vocab("../data_in/pos_vocab_190408.pkl")

    print(len(base_word_vocab.word2idx))

    updated_word_vocab, updated_pos_vocab = update_word_pos_vocab(text_list, threshold=1, base_word_vocab=base_word_vocab, base_pos_vocab=base_pos_vocab, word_vocab_path="../data_in/word_vocab_190418.pkl", pos_vocab_path="../data_in/pos_vocab_190418.pkl", tokenizer_type="mecab")

    print(len(updated_word_vocab.word2idx))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--word2vec_corpus', type=str, default='./data_in/word2vec_corpus/smonwar_corpus.txt')
    parser.add_argument('--intent_corpus', type=str, default='./data_in/intent_corpus.txt')
    parser.add_argument('--word2vec_dim', type=int, default=100)
    parser.add_argument('--word2vec_path', type=str,
                        default='./model/word2vec/ko_cs_word2vec_100.model')  # dim이랑 코드 엮여야함

    parser.add_argument('--vocab_path', type=str, default='./data_out/vocab_ko_cs.pkl', help='path for saving vocab')

    args = parser.parse_args()
    main(args)
