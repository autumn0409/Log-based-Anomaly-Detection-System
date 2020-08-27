import numpy as np
import pickle


embedding_table = {}
with open('crawl-300d-2M.vec') as f:
    for i, line in enumerate(f):
        if i == 0:  # header
            # print('words, dim =', line.split())
            continue

        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embedding_table[word] = coefs

with open('preprocessed_data/embedding_table.pkl', 'wb') as outputfile:
    pickle.dump(embedding_table, outputfile)
