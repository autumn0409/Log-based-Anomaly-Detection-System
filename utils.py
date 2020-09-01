from collections import Counter
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
import math


EMBEDDING_DIM = 300


def template2tokens(templates):
    # templates: list of str
    # e.g. templates = ['PacketResponder <*> for block <*> terminating', 'Received block <*> of size <*> from <*>']
    # return: list of list of str which has been processed (tokens)
    # e.g. [['packet', 'responder', 'block', 'terminate'], ['receive', 'block', 'size']]

    list_tokens = []

    for i, text in enumerate(templates):
        for j in reversed(range(len(text))):
            if j == 0:
                break
            if not text[j].isalpha():
                text = text[:j] + ' ' + text[j + 1:]
                continue
            if text[j].isupper() and text[j - 1].islower():
                text = text[:j] + ' ' + text[j:]

        tokens = nltk.word_tokenize(text.lower())
        tokens = [
            token for token in tokens if token not in stopwords.words('english')]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token, 'v') for token in tokens]
        list_tokens.append(tokens)

    return list_tokens


def calculate_freq(list_tokens, mode_idf=False, counter=None):
    # list_tokens: list of list of str
    # e.g. [['packet', 'responder', 'block', 'terminate'], ['receive', 'block', 'size']]
    # mode_idf: bool
    # if `mode_idf' is set to True, count as idf
    # counter: collections.Counter object
    # if `counter' parameter is not assigned, generate a new one
    # otherwise, the counter object being assigned will be update (aggregate)

    if counter is None:
        counter = Counter()

    for tokens in list_tokens:
        if mode_idf:
            counter_tokens = Counter(set(tokens))
            counter_tokens['__num_lines__'] = 1
        else:
            counter_tokens = Counter(tokens)

        counter.update(counter_tokens)

    return counter


def template2vec(templates, embedding_table, counter_idf):
    # templates: list of str
    # e.g. templates = ['PacketResponder <*> for block <*> terminating', 'Received block <*> of size <*> from <*>']
    # embedding_table: dict
    # a dict mapping words to vectors of dimension EMBEDDING_DIM
    # counter_idf: collections.Counter object
    # 'counter_idf' indicates the counter for calculating idf
    # return: list of numpy array

    list_vectors = []

    list_tokens = template2tokens(templates)
    for tokens in list_tokens:
        vector_token = np.zeros(EMBEDDING_DIM)

        counter_tf = calculate_freq([tokens])
        num_valid_token = 0

        for token in tokens:
            if token not in embedding_table:
                continue

            num_valid_token += 1

            tf = counter_tf[token] / sum(counter_tf.values())
            idf = np.log(counter_idf['__num_lines__'] /
                         (1 + counter_idf.get(token, 0)))

            vector = embedding_table[token] * tf * idf
            vector_token += vector

        list_vectors.append(vector_token / num_valid_token)

    return list_vectors


class DataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size):
        'Initialization'
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        'Denotes the number of batches'
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        x = self.x[index * self.batch_size:(index + 1) * self.batch_size]
        y = self.y[index * self.batch_size:(index + 1) * self.batch_size]

        x = pad_sequences(x, dtype='object', padding='post',
                          value=np.zeros(300)).astype(np.float32)

        return x, y
