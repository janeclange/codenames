from codenames2 import *
import numpy as np
import tqdm

def guesser(words, clue, n):
    pass
#   return str

def cluer(positive, negative, neutral, assassin):
    pass
#   return str, int

def eval(cluer, guesser, expirements=25):
    turns_per_game = 0
    games = 0
    for i in tqdm.tqdm(range(expirements)):
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
                    guess = guesser(board_words, clue, n_target).lower()

                guessed_words.append(guess)
                board_words.remove(guess)
                if guess in assassin:
                    turn_done = True
                    done = True
                    turns += 25
                if guess in red_words:
                    turns += 1
                    turn_done = True
                    red_words.remove(guess)
                if guess in blue_words:
                    n_target -= 1
                    # guesser.ally_words_remaining -= 1
                    if n_target == 0:
                        turn_done = True
                    blue_words.remove(guess)
                    n_ally_words_left = len(blue_words)
                    if n_ally_words_left == 0:
                        done = True
                if guess in bystanders:
                    turn_done = True
                    bystanders.remove(guess)
            # writer.writerow(["_".join(target_words), clue_tup[0], "_".join(guessed_words)])
            turns += 1
        turns_per_game += turns
        games += 1
    return turns_per_game / games



def get_adversary_cluer(embedding_type="word2vec"):
    from codenames_adversary import codenames
    game = codenames.Codenames(embedding_type)
    def clue(positive, negative, neutral, assassin):
        game._build_game(red=negative, blue=positive)
        best_scores, best_clues, best_board_words_for_clue = game.get_clue(2, 1)
        z = list(zip(best_scores, best_clues, best_board_words_for_clue))
        # If no clue is given, clue "glove"
        if len(z) == 0:
            z = [(0, ["glove"], ["unk-eliot", "unk-eliot"])]
        best = min(z, key=lambda x: x[0])
        best_word = best[1]
        # import IPython; IPython.embed()
        return best_word[0], len(best[2])

    return clue

def get_guesser(embedding="word2vec"):
    if embedding == "word2vec":
        from embeddings import word2vec
        embedding = word2vec.Word2Vec()
    elif embedding == "glove":
        from embeddings import glove
        embedding = glove.Glove()

    def guesser(words, clue, n):
        scores = np.array([embedding.get_word_similarity(word, clue) for word in words])
        word_idc = np.argmax(scores)
        return words[word_idc]

    return guesser

def our_cluer():
    spymaster = Cluer2()
    def cluer(positive, negative, neutral, assassin):
        spymaster.new_game(assassin=assassin, red_words=red_words, blue_words=blue_words, bystanders=bystanders)
        return spymaster.clue()

    return cluer

if __name__ == "__main__":
    result = eval(get_adversary_cluer(), get_guesser())
    result2 = eval(our_cluer(), get_guesser())
    result3 = eval(get_adversary_cluer(), get_guesser("glove"))
    print(result, result2, result3)




