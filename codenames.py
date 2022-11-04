import numpy as np
from random import sample
import gensim 

model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews_vectors.bin", binary=True)

with open("codewords_simplified.txt") as f: 
    codewords = [x.strip() for x in f.readlines()]

with open("cm_wordlist.txt") as f: 
    clue_words = [x.strip() for x in f.readlines()]

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
        self.board_state = (assassin, red_words, blue_words, bystanders)
        self.previous_guesses = []
        self.ally_words_remaining = 8

    def clue(self):
        return model.most_similar(positive=blue_words, negative=red_words, restrict_vocab=10000, topn=1)[0][0]

    def human_guess(self):
        return input()


spymaster = Agent(assassin, red_words, blue_words, bystanders, model)
guesser = Agent(assassin, red_words, blue_words, bystanders, model)


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


