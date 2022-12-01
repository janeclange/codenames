from codenames2 import *
import numpy as np

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
                while not guess in board_words:
                    guess = guesser(board_words, clue, n_target)

                guessed_words.append(guess.lower())
                board_words.remove(lower([guess]))
                print(spymaster.previous_guesses)
                if guess in assassin:
                    turn_done = True
                    done = True
                    turns += 25
                if guess in red_words:
                    turns += 1
                    turn_done = True
                    red_words.remove(lower([guess]))
                if guess in blue_words:
                    n_target -= 1
                    # guesser.ally_words_remaining -= 1
                    if n_target == 0:
                        turn_done = True
                    blue_words.remove(lower([guess]))
                    n_ally_words_left = len(blue_words)
                    if n_ally_words_left == 0:
                        done = True
                if guess in bystanders:
                    turn_done = True
                    bystanders.remove(lower([guess]))
            writer.writerow(["_".join(target_words), clue_tup[0], "_".join(guessed_words)])
            turns += 1
        turns_per_game += turns
        games += 1
    return turns_per_game / games



def get_adversary_cluer(embedding_type="word2vec"):
    from codenames import codenames
    game = codenames.Codenames(embedding_type)
    def clue(positive, negative, neutral, assassin):
        game._build_game(red=negative, blue=positive)
        best_scores, best_clues, best_board_words_for_clue = game.get_clue(2, 1)
        return best_clues[0], 2

    return clue

def get_guesser(embedding="word2vec"):
    from embeddings import word2vec
    embedding = word2vec.Word2Vec()
    def guesser(words, clue, n):
        scores = np.array([embedding.get_word_similarity(word, clue) for word in words])
        word_idc = np.argmax(scores)
        return words[word_idc]

    return guesser

def our_cluer():
    spymaster = Cluer()
    def cluer(positive, negative, neutral, assassin):
        spymaster.assassin, spymaster.red_words, spymaster.blue_words, spymaster.bystanders = spymaster.lower(
            assassin), spymaster.lower(negative), spymaster.lower(positive), spymaster.lower(neutral)
        return spymaster.clue()

    return cluer

if __name__ == "__main__":
    print(eval(our_cluer(), get_guesser()))
    print(get_adversary_cluer(), get_guesser())



