import os

import numpy as np
from random import sample
# import gensim 
import itertools
from cluer import Cluer
from cluer import Cluer2
import random
import time
import csv
import getpass

def lower(array):
    return [a.lower() for a in array]

with open("codewords_simplified.txt") as f:
    codewords = [x.strip() for x in f.readlines()]

with open("cm_wordlist.txt") as f:
    clue_words = [x.strip() for x in f.readlines()]

if __name__ == "__main__":

    spymaster = Cluer2()
    spymaster.load_clues()

    #clue_sample = sample(clue_words, 1000)
    os.makedirs(f"logs/{getpass.getuser()}", exist_ok=True)
    record_csv = open(f'logs/{getpass.getuser()}/codenames2_record_{int(time.time())}.csv', 'w')
    writer = csv.writer(record_csv)

    try:
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
                target_words = spymaster.word_best_tup
                print("Clue: ", clue_tup)
                spymaster.previous_clues.append(clue_tup[0])
                n_target = clue_tup[1]
                turn_done = False
                guessed_words = []
                while not turn_done:
                    guess = ""
                    while guess not in board_words or guess in spymaster.previous_guesses:
                        guess = input()

                    spymaster.previous_guesses.append(guess.lower())
                    guessed_words.append(guess.lower())
                    print(spymaster.previous_guesses)
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
                writer.writerow(["_".join(target_words), clue_tup[0], "_".join(guessed_words)])
                guessed_words = []
                turns += 1
                print("Turns so far: " + str(turns))

            print("Game finished!")
            print("Turns: " + str(turns))
            print("True board:")
            print([assassin, red_words, blue_words, bystanders])
    finally:
        record_csv.close()
