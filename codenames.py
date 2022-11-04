import numpy as np
from random import sample
import gensim 
import itertools
import embeddings
import embeddings.word2vec
import embeddings.glove
import embeddings.fasttext
import embeddings.bert

# model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews_vectors.bin", binary=True)
word2vec = embeddings.word2vec.Word2Vec()
glove = embeddings.glove.Glove()
fasttext = embeddings.fasttext.FastText()
bert = embeddings.bert.Bert()

with open("codewords_simplified.txt") as f: 
    codewords = [x.strip() for x in f.readlines()]

with open("cm_wordlist.txt") as f: 
    clue_words = [x.strip() for x in f.readlines()]

clue_sample = sample(clue_words, 1000)

board_words = sample(codewords, k=25)
assassin = board_words[0]
red_words = board_words[1:10]
blue_words = board_words[10:18]
bystanders = board_words[18:25]

print([assassin, red_words, blue_words, bystanders]) 

done = False 
turns = 0 

class Agent:
    def __init__(self, assassin, red_words, blue_words, bystanders, model):
        self.assassin, self.red_words, self.blue_words, self.bystanders = assassin, red_words, blue_words, bystanders
        self.previous_guesses = []
        self.ally_words_remaining = 8
        self.model = model

    def clue_by_ranking(self):
        rankings = {}
        for w in clue_sample:
            similarities = []
            for v in board_words:
                similarities.append(word2vec.get_word_similarity(w,v), v)
            similarities.sort()
            rankings[w] = similarities

        max_targets = 0
        best_clue = ""
        for w in clue_sample:
            n = 0
            for (s,v) in rankings[w]:
                if v in self.blue_words:
                    n += 1
                else:
                    break
            if n > max_targets:
                best_clue = w 
                max_targets = n
        return best_clue 
        
            
    def clue_w2v_pairs(self):
        best = ("", 0) 
        pairs = itertools.combinations(blue_words, 2)
        for p in pairs:
            candidate = self.model.most_similar(positive=list(p), restrict_vocab=100000, topn=1)[0]
            if candidate[1] >= best[1]:
                best = (candidate[0] + str(p), candidate[1])
        return best[0] 

    def clue(self):
        for w in self.previous_guesses:
            if w in self.blue_words:
                self.blue_words.remove(w)
        return self.clue_by_ranking()

    def human_guess(self):
        return input()
    
    def guess(self, guessable_words, clue, model):
        """Naive nearest neighbor guesser"""
        # words = list of words to guess from
        # clue = (clue word, number of targets)
        # model
        word_scores = []
        for word in guessable_words:
            word_scored.append(model.get_word_similarity(word, clue))
        
        # indices of words in descending order of score
        best_word_ixd = np.flip(np.argsort(word_scores))

        return guessable_words[best_word_ixd[:clue[1]]]

    def clue_thresholds(self, assassin, red_words, blue_words, bystanders, model):
        pass

    def combined_scorer(self, assassin, red_words, blue_words, bystanders, model):
        






spymaster = Agent(assassin, red_words, blue_words, bystanders, word2vec)
guesser = Agent(assassin, red_words, blue_words, bystanders, word2vec)


while not done: 
    print("Clue: " + spymaster.clue())
    turn_done = False
    while not turn_done:
        guess = guesser.human_guess()
        if not guess in board_words or guess in spymaster.previous_guesses:
            break

        spymaster.previous_guesses.append(guess)
        guesser.previous_guesses.append(guess)
        if guess == assassin: 
            print("Assassin")
            turn_done = True
            done = True
            turns += 25
        if guess in red_words: 
            print("Red")
            turns += 1
            turn_done = True
        if guess in blue_words:
            print("Blue")
            spymaster.ally_words_remaining -= 1
            guesser.ally_words_remaining -= 1
            if spymaster.ally_words_remaining == 0:
                turn_done = True
                done = True
        if guess in bystanders:
            print("Bystander")
            turn_done = True
    turns += 1
    print("Turns so far: " + str(turns))

print("Turns: " + str(turns))


