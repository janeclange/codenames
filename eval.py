from codenames2 import *

def guesser(words, clue, n):
    pass
#   return str

def cluer(positive, negative, neutral, assassin):
    pass
#   return str, int

def eval(guesser, cluer, expirements=25):
    turns_per_game = 0
    games = 0
    for i in range(expirements):
        board_words = lower(sample(codewords, k=25))
        assassin = lower([board_words[0]])
        red_words = lower(board_words[1:10])
        blue_words = lower(board_words[10:18])
        bystanders = lower(board_words[18:25])

        random.shuffle(board_words)
        print(board_words)

        done = False
        turns = 0

        while not done:
            clue, n_target = cluer(blue_words, red_words, bystanders, assassin)
            print("Clue: ", clue)
            turn_done = False
            guessed_words = []
            while not turn_done:
                guess = ""
                while not guess in board_words or guess in spymaster.previous_guesses:
                    guess = guesser(board_words, clue, n_target)

                spymaster.previous_guesses.append(guess.lower())
                guessed_words.append(guess.lower())
                print(spymaster.previous_guesses)
                # guesser.previous_guesses.append(guess)
                if guess in assassin:
                    turn_done = True
                    done = True
                    turns += 25
                if guess in red_words:
                    turns += 1
                    turn_done = True
                if guess in blue_words:
                    n_target -= 1
                    # guesser.ally_words_remaining -= 1
                    if n_target == 0:
                        turn_done = True
                    n_ally_words_left = len(spymaster.blue_words) - len(
                        set(spymaster.previous_guesses).intersection(set(spymaster.blue_words)))
                    if n_ally_words_left == 0:
                        done = True
                if guess in bystanders:
                    turn_done = True
            writer.writerow(["_".join(target_words), clue_tup[0], "_".join(guessed_words)])
            turns += 1
        turns_per_game += turns
        games += 1
    return turns_per_game / games

