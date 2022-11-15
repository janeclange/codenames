import tqdm
import pickle
import numpy as np
import h5py

# download this file:
# http://conceptnet.s3.amazonaws.com/precomputed-data/2016/numberbatch/19.08/mini.h5
# as 'mini.h5'

# and this file:
# https://gist.github.com/h3xx/1976236/raw/bbabb412261386673eff521dddbe1dc815373b1d/wiki-100k.txt
# as 'wiki-100k.txt'

class Guesser:
    def __init__(self):
        self.words = None
        self.embedding = None
        pass

    def load_data(self):
        hf = h5py.File('mini.h5', 'r')
        data = hf.get('mat')
        raw_words = np.array(data['axis1'])
        vecs = np.array(data['block0_values'])

        with open('wiki-100k.txt', encoding='utf8') as file:
            lines = []
            for line in file:
                if line.rstrip().isalpha():
                    lines.append(line.rstrip().lower())
        with open("codewords_simplified.txt") as file:
            lines2 = [s.strip().lower() for s in file.readlines()]
        common_words = np.sort(np.unique(np.array(lines+lines2)))

        valid_inds = []
        valid_words = []
        for i in range(len(raw_words)):
            w = raw_words[i].decode('utf-8')
            if w[:6] == '/c/en/':
                search_ind = np.searchsorted(common_words, w[6:])
                if search_ind < len(common_words) and common_words[search_ind] == w[6:]:
                    valid_inds.append(i)
                    valid_words.append(w[6:])

        valid_inds = np.array(valid_inds)
        self.words = np.array(valid_words)
        self.embedding = vecs[valid_inds].astype(np.int64)

    def filter_valid_words(self, ws):
        inds = np.searchsorted(self.words, ws)
        #print(list(zip(ws,inds)))
        result = [word for word,ind in zip(ws,inds) if ind<len(self.words) and word==self.words[ind]]
        return result

    def find_vecs(self, ws):
        inds = np.searchsorted(self.words, ws)
        for i in range(len(ws)):
            if self.words[inds[i]] != ws[i]:
                print('WORD MISSING', ws[i])
                return None
        return self.embedding[inds]

    def eval_inner_products(self, w1, w2):
        inner_prods = np.matmul(w1, w2.T).astype(np.float64)
        inner_prods /= np.sqrt(np.outer(np.sum(w1**2, axis=1), np.sum(w2**2, axis=1)))
        return inner_prods

    def guess(self, clue, board, n):
        inner_prods = self.eval_inner_products(self.find_vecs([clue]), self.find_vecs(board))[0]
        top_n_inds = np.argsort(inner_prods)[-n:]
        return np.array(board)[top_n_inds][::-1], inner_prods[top_n_inds][::-1]

    def score_clues(self, target, clues):
        inner_prods = self.eval_inner_products(self.find_vecs(target), self.find_vecs(clues))
        maxes = np.amin(inner_prods, axis=0)
        sort_inds = np.argsort(maxes)[::-1]
        print(inner_prods)
        return np.array(clues)[sort_inds], maxes[sort_inds]
        

if __name__=="__main__":
    g = Guesser()
    g.load_data()
    print('what would I guess for "apple", in descending order of preference?')
    print(g.guess('apple', ['orange', 'phone', 'hair', 'chair'], 4))

    print('what would I clue for "bat" and "eye", in descending order of preference?')
    print(g.score_clues(['bat', 'eye'], ['blind', 'blink', 'pupil', 'mammal']))

    #print(g.score_clues(['africa', 'agent'],['tripoli', 'antimalarial', 'patient', 'service', 'chad', 'transport', 'metaphosphoric_acid', 'ant', 'europe', 'antpecker', 'bishop', 'polish', 'continent', 'snake', 'seven', 'global_south', 'rider', 'double', 'cement', 'gun', 'abc', 'poison', 'bond', 'scour', 'horn', 'vehicle', 'guy', 'name', 'marabou', 'antarctica', 'cat', 'gnu', 'official', 'avail', 'shari', 'political', 'hypo', 'malaria', 'act', 'central_african_republic']))
    print(g.filter_valid_words(['tripoli', 'antimalarial', 'patient', 'service', 'chad', 'transport', 'metaphosphoric_acid', 'ant', 'europe', 'antpecker', 'bishop', 'polish', 'continent', 'snake', 'seven', 'global_south', 'rider', 'double', 'cement', 'gun', 'abc', 'poison', 'bond', 'scour', 'horn', 'vehicle', 'guy', 'name', 'marabou', 'antarctica', 'cat', 'gnu', 'official', 'avail', 'shari', 'political', 'hypo', 'malaria', 'act', 'central_african_republic']))