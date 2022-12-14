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
import sys

def lower(array):
    return [a.lower() for a in array]

with open("codewords_simplified.txt") as f:
    codewords = [x.strip() for x in f.readlines()]

with open("cm_wordlist.txt") as f:
    clue_words = [x.strip() for x in f.readlines()]

def print_board_fancy(board_words, assassin, red_words, blue_words, bystanders, previous_guesses):
    grid_size = 5

    # Print the grid rows with row numbers
    for i in range(grid_size):
        # Print the words in the row, padded with spaces to align
        for j in range(grid_size):
            if board_words[i * grid_size + j] in previous_guesses:
                if board_words[i * grid_size + j] in red_words:
                    color = "\033[1;31m"
                elif board_words[i * grid_size + j] in blue_words:
                    color = "\033[1;34m"
                elif board_words[i * grid_size + j] in bystanders:
                    color = "\033[1;33m"
                elif board_words[i * grid_size + j] in assassin:
                    color = "\033[1;30m"
            else:
                color = "\033[1;37m"
            # Print the word with the specified color
            print("{color}{:^{w}s}\033[0m".format(board_words[i * grid_size + j], w=12, color=color), end="")
        # Start a new line for the next row
        print()
HUMAN=True

if __name__ == "__main__":

    HUMAN = len(sys.argv)>1 and sys.argv[1]=="human"

    spymaster = Cluer2()
    # spymaster.load_clues()

    #clue_sample = sample(clue_words, 1000)
    os.makedirs(f"logs/{getpass.getuser()}", exist_ok=True)
    record_csv = open(f'logs/{getpass.getuser()}/codenames2_record_{int(time.time())}.csv', 'w')
    writer = csv.writer(record_csv)

    turns_csv = open(f'logs/{getpass.getuser()}/codenames2_turns_{int(time.time())}.csv', 'w')
    turns_writer = csv.writer(turns_csv)

    try:
        while(True):
            print("enter a seed, or 0 for a random seed")
            seed = int(input("seed:"))
            if seed:
                random.seed(seed)
            board_words = lower(sample(codewords, k=25))
            #board_words = ['snowman','giant','helicopter','field','scorpion','alps','ray','unicorn','maple','calf','shop','table','circle','part','bridge','turkey','bell','lawyer','play','cricket','log','australia','chair','bar','center']
            #board_words = ['school','pyramid','organ','robin','bomb','superhero','contract','thumb','dwarf','note','microscope','chest','triangle','fair','stream','bugle','stadium','arm','spike','boom','band','antarctica','palm','fall','spring']
            assassin = lower([board_words[0]])
            red_words = lower(board_words[1:10])
            blue_words = lower(board_words[10:18])
            bystanders = lower(board_words[18:25])

            spymaster.new_game(assassin=assassin, red_words=red_words, blue_words=blue_words, bystanders=bystanders)

            spymaster.assassin = assassin
            spymaster.red_words = red_words
            spymaster.blue_words = blue_words
            spymaster.bystanders = bystanders
            spymaster.previous_guesses = []
            spymaster.previous_clues = []
            spymaster.previous_clues_output = []

            #print([assassin, red_words, blue_words, bystanders])
            random.shuffle(board_words)
            print("New game -- here is the board")
            print(board_words)

            if True and HUMAN:
                print_board_fancy(board_words, assassin, red_words, blue_words, bystanders, board_words)

            done = False
            turns = 0
            while not done:
                clue_tup = spymaster.clue() if not HUMAN else ("",25)
                target_words = spymaster.word_best_tup if not HUMAN else []
                print_board_fancy(board_words, assassin, red_words, blue_words, bystanders, spymaster.previous_guesses)

                print("Clue: ", clue_tup)
                spymaster.previous_clues.append(clue_tup[0])
                spymaster.previous_clues_output.append(list(target_words))
                
                # print("Targets:", target_words)
                n_target = clue_tup[1]
                n_target = 25
                turn_done = False
                guessed_words = []
                while not turn_done:
                    guess = ""
                    while guess != "new game" and guess != "quit" and (guess not in board_words or guess in spymaster.previous_guesses):
                        print("Enter a word on the board that hasn't been previously guessed. Enter \"new game\" to refresh the board, or \"quit\" to quit.")
                        guess = input().strip()
                    if guess == "new game":
                        turn_done = True
                        done = True
                        break
                    if guess == "quit": 
                        exit()
                    spymaster.previous_guesses.append(guess.lower())
                    guessed_words.append(guess.lower())

                    if guess in assassin:
                        print("You guessed the Assassin.")
                        turn_done = True
                        done = True
                        turns += 25
                    if guess in red_words:
                        print("You guessed a Red word (that's the opposing team).")
                        turns += 1
                        turn_done = True
                    if guess in blue_words:
                        print("You guessed a Blue word (that's your team).")
                        n_target -= 1
                        # guesser.ally_words_remaining -= 1
                        if n_target == 0:
                            turn_done = True
                        n_ally_words_left = len(spymaster.blue_words) - len(set(spymaster.previous_guesses).intersection(set(spymaster.blue_words)))
                        if n_ally_words_left == 0:
                            done = True
                    if guess in bystanders:
                        print("You guessed a Bystander.")
                        turn_done = True

                    print_board_fancy(board_words, assassin, red_words, blue_words, bystanders, spymaster.previous_guesses)

                writer.writerow(["_".join(target_words), clue_tup[0], "_".join(guessed_words)])
                guessed_words = []
                turns += 1
                print("Turns so far: " + str(turns))

            print("Game finished!")
            print("Turns: " + str(turns))
            turns_writer.writerow([turns])
            print("True board:")
            #print([assassin, red_words, blue_words, bystanders])
            print_board_fancy(board_words, assassin, red_words, blue_words, bystanders, board_words)
            print("Intended clues:")
            print(list(zip(spymaster.previous_clues, spymaster.previous_clues_output)))
    finally:
        record_csv.close()
        turns_csv.close()
