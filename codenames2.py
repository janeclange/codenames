import numpy as np
from random import sample
# import gensim 
import itertools
from cluer import Cluer
import random
from conceptnet import ConceptNetGraph

def lower(array):
    return [a.lower() for a in array]

with open("codewords_simplified.txt") as f: 
    codewords = [x.strip() for x in f.readlines()]

with open("cm_wordlist.txt") as f: 
    clue_words = [x.strip() for x in f.readlines()]

spymaster = Cluer()
spymaster.load_clues()

#clue_sample = sample(clue_words, 1000)

while(True):
    board_words = lower(sample(codewords, k=25))
    assassin = lower([board_words[0]])
    red_words = lower(board_words[1:10])
    blue_words = lower(board_words[10:18])
    bystanders = lower(board_words[18:25])

    spymaster.assassin = assassin
    spymaster.red_words = red_words
    spymaster.blue_words = blue_words
    spymaster.bystanders = bystanders
    spymaster.previous_guesses = []

    #print([assassin, red_words, blue_words, bystanders]) 
    random.shuffle(board_words)
    print(board_words)

    done = False 
    turns = 0 


    while not done: 
        clue_tup = spymaster.clue()
        print("Clue: ", clue_tup)
        n_target = clue_tup[1]
        turn_done = False
        while not turn_done:
            guess = ""
            while not guess in board_words or guess in spymaster.previous_guesses:
                guess = input()

            spymaster.previous_guesses.append(guess.lower())
            print(spymaster.previous_guesses)
            # guesser.previous_guesses.append(guess)
            if guess in assassin: 
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
                n_target -= 1
                # guesser.ally_words_remaining -= 1
                if n_target == 0:
                    turn_done = True
                n_ally_words_left = len(spymaster.blue_words) - len(set(spymaster.previous_guesses).intersection(set(spymaster.blue_words)))
                if n_ally_words_left == 0:
                    done = True
            if guess in bystanders:
                print("Bystander")
                turn_done = True
        turns += 1
        print("Turns so far: " + str(turns))

    print("Game finished!")
    print("Turns: " + str(turns))
    print("True board:")
    print([assassin, red_words, blue_words, bystanders]) 