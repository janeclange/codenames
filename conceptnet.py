import functools
import multiprocessing
import os
import pickle
import numberbatch_guesser
import time
from functools import lru_cache, reduce

import numpy as np
import tqdm

NUM_NEGATIVE_WORDS = 9
# NUM_POSITIVE_WORDS = 9
NUM_POSITIVE_WORDS = 4
powersetify = lambda s : reduce(lambda P, x: P + [subset | {x} for subset in P], s, [set()])

class ConceptNetGraph:
    def __init__(self):
        self.edges = {} #dictionary of nodes -> list of edges
        pass
    def parse_graph(self):
        #read the file with all the edges in concept net, and put them in the dictionary
        with open("conceptnet-assertions-5.7.0.csv",encoding="utf8") as f:
            for line in tqdm.tqdm(f, total=34074917):
                if "/r/ExternalURL/" in line:
                    continue
                l = line.split("\t")
                if l[2][:6] != "/c/en/":
                    continue
                if l[3][:6] != "/c/en/":
                    continue
                a = l[2].split("/")[3]
                b = l[3].split("/")[3]
                if a not in self.edges:
                    self.edges[a] = []
                if b not in self.edges:
                    self.edges[b] = []
                self.edges[a] += [b]
                self.edges[b] += [a]
        with open("conceptnet-assertions-en","wb") as f:
            pickle.dump(self, f)
    def load_graph():
        if "conceptnet-assertions-en" not in os.listdir():
            raise FileNotFoundError("You need to download the conceptnet assertions file from https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz")
        with open("conceptnet-assertions-en","rb") as f:
            return pickle.load(f)
    def get_distance_k_neighbors(self, word, k):
        # return a dictionary where keys are distance k words, and values are lists that are the path to that word
        l={word:[word]}
        for i in range(k):
            l_temp = list(l.keys()).copy()
            for w in l_temp:
                for neighbor in self.edges[w]:
                    if neighbor in l:
                        continue
                    else:
                        degree = len(self.edges[neighbor])
                        if degree>10 and degree<1000:
                            l[neighbor] = l[w] + [(neighbor,degree)]
        return l
    def get_two_word_clue(self, word1:str, word2:str, guesser):
        word1_1 = self.get_distance_k_neighbors(word1,1)
        word2_1 = self.get_distance_k_neighbors(word2,1)
        possible_clues = set(word1_1.keys()).intersection(set(word2_1.keys()))
        possible_clues = guesser.filter_valid_words(list(possible_clues))
        if possible_clues:
            return guesser.score_clues([word1,word2],possible_clues)[0]
        word1_2 = self.get_distance_k_neighbors(word1,2)
        word2_2 = self.get_distance_k_neighbors(word2,2)
        possible_clues = set(word1_1.keys()).intersection(set(word2_2.keys())).union(set(word1_2.keys()).intersection(set(word2_1.keys())))
        possible_clues = guesser.filter_valid_words(list(possible_clues))
        if possible_clues:
            return guesser.score_clues([word1,word2],possible_clues)[0]
        else:
            return []

    def all_intersections(sets, current=None):
        if sets:
            return [all_intersections(sets[1:], current=current), all_intersections(sets[1:], current=current.intersection(sets[0]) if current!=None else sets[0])]
        else:
            return current
    def get_k_word_clue(self, words, guesser):
        neighbors_1 = [self.get_distance_k_neighbors(w,1) for w in words]
        neighbors_2 = [self.get_distance_k_neighbors(w,2) for w in words]
        size_1_intersections = []


        
# Helper for Cluer Plus
def eval_permutation(i, guesser, cluer, positive_words, negative_words, neutral_words, assasin_words, num_moves):
    clue_size = len(i)  # TODO see next comment
    if clue_size == 1:
        clues = guesser.filter_valid_words(list(cluer.get_distance_k_neighbors(list(i)[0], 1).keys()))
        if len(clues) > 0:
            clue = clues[1]
        else:
            print("No valid clue found for words: ", list(i))
            return None
    elif clue_size == 2:
        clues = guesser.filter_valid_words(cluer.get_two_word_clue(*i, guesser))  # TODO clue multiple words
        if len(clues) > 0:
            clue = clues[0]
        else:
            print("No valid clue found for words: ", list(i))
            return None
    else:
        raise ValueError("clue size must be 1 or 2")
    # Get the guessed words
    try:
        guesser_rankings, _guesser_scores = guesser.guess(clue, list(positive_words) + list(negative_words) + list(
            negative_words) + list(assasin_words), clue_size)  # TODO weigh guesses based on clue.
    except Exception as e:
        print(e)
        import IPython; IPython.embed()
    # Simulate the guessing process
    guessed_words = set()
    for i in range(clue_size):
        guessed_words.add(guesser_rankings[i])
        if guesser_rankings[i] not in positive_words:
            break
    # Get the score of this new board.
    _best_clue, expected_score = cached_cluer_plus(guesser, cluer, positive_words - guessed_words,
                                                   negative_words - guessed_words,
                                                   neutral_words - guessed_words, assasin_words - guessed_words,
                                                   num_moves + 1, False)
    return clue, expected_score

# @lru_cache(maxsize=100000)
def cached_cluer_plus(guesser, cluer, positive_words, negative_words, neutral_words, assasin_words, num_moves = 0, use_multi = False):
    permutations = powersetify(positive_words)
    permutations.remove(set())
    # Enforce that the number of words for a clue is not greater than 2
    permutations = [a for a in permutations if len(a) < 3]
    # Check if the board is in a final state
    terminal = len(assasin_words) == 0 or len(positive_words) == 0
    if terminal:
        return "", num_moves + (NUM_NEGATIVE_WORDS - len(negative_words)) + (not len(assasin_words)) * 25
    # Consider giving a clue for each possible combination of positive words
    if use_multi:
        cpus = multiprocessing.cpu_count()
        print(f"Creating multiprocessing pool with {cpus} cpus")
        with multiprocessing.Pool(cpus) as p:
            func = functools.partial(eval_permutation, guesser=guesser, cluer=cluer, positive_words=positive_words,
                                     negative_words=negative_words, neutral_words=neutral_words,
                                     assasin_words=assasin_words, num_moves=num_moves)
            results = list(tqdm.tqdm(p.imap(func, permutations), total=len(permutations)))
    else:
        results = []  # pairs
        for i in permutations:
            results.append(
                eval_permutation(i, guesser, cluer, positive_words, negative_words, neutral_words, assasin_words,
                                 num_moves))
    return min(results, key=lambda x: x[1])

def cluer_plus(guesser, cluer, positive_words, negative_words, neutral_words, assasin_words, num_moves = 0, use_multi=True):
    return cached_cluer_plus(guesser, cluer, positive_words, negative_words, neutral_words, assasin_words, num_moves, use_multi=use_multi)


def play_simulation(guesser, cluer, verbose = False, use_multi=True):
    with open("codewords_simplified.txt") as file:
        lines2 = [s.strip().lower() for s in file.readlines()]
    common_words = np.sort(np.unique(np.array(lines2)))
    # Positive words
    positive_words = np.random.choice(common_words, NUM_POSITIVE_WORDS, replace=False)
    common_words = np.setdiff1d(common_words, positive_words)
    positive_words = set(positive_words)
    if verbose:
        print(f"Positive words: {positive_words}")
    # Negative words
    negative_words = np.random.choice(common_words, NUM_NEGATIVE_WORDS, replace=False)
    common_words = np.setdiff1d(common_words, negative_words)
    negative_words = set(negative_words)
    if verbose:
        print(f"Negative words: {negative_words}")
    # Neutral words
    neutral_words = np.random.choice(common_words, 2, replace=False)
    common_words = np.setdiff1d(common_words, neutral_words)
    neutral_words = set(neutral_words)
    if verbose:
        print(f"Neutral words: {neutral_words}")
    # Assasin words
    assasin_words = set(np.random.choice(common_words, 1, replace=False))
    if verbose:
        print(f"Assasin words: {assasin_words}")
    assert len(assasin_words) == 1
    return cluer_plus(guesser, cluer, positive_words, negative_words, neutral_words, assasin_words, use_multi=use_multi)

def run():
    cluer = ConceptNetGraph.load_graph()
    guesser = numberbatch_guesser.Guesser()
    guesser.load_data()
    start = time.time()
    print(play_simulation(guesser, cluer, verbose=True, use_multi=True))
    end = time.time()
    print(f"Duration: {end - start} seconds")

if __name__ == "__main__":
    run()


def compute_all_two_word_clues():
    g = ConceptNetGraph.load_graph()
    guesser = numberbatch_guesser.Guesser()
    guesser.load_data()
    with open("codewords_simplified.txt") as f: 
        codewords = [x.strip() for x in f.readlines()]
        for w in codewords:
            w = w.replace(" ","_").lower()
            for w2 in codewords:
                w2 = w2.replace(" ","_").lower()
                if w!=w2:
                    #print((w,w2),g.get_two_word_clue(w,w2))
                    clues = g.get_two_word_clue(w,w2)
                    #print([w,w2])
                    #print(clues)
                    clues = guesser.filter_valid_words(clues)
                    if clues:
                        scored_clues = guesser.score_clues([w,w2],clues)
                        print((w,w2), scored_clues[0][:5])
                    else:
                        print((w,w2))
