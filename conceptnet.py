import tqdm
import pickle
import numberbatch_guesser
class ConceptNetGraph:
    def __init__(self):
        self.edges = {} #dictionary of nodes -> list of edges
        self.guesser = None
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
            pickle.dump(self.edges, f)
    def load_graph(self):
        with open("conceptnet-assertions-en","rb") as f:
            g = pickle.load(f)
            self.edges = g
            self.guesser = numberbatch_guesser.Guesser()
    def get_distance_k_neighbors(self, word, k):
        #return a dictionary where keys are distance k words, and values are lists that are the path to that word
        l={word:[word]}
        for i in range(k):
            l_temp = list(l.keys()).copy()
            for w in l_temp:
                for neighbor in self.edges[w]:
                    if neighbor in l:
                        continue
                    else:
                        degree = len(self.edges[neighbor])
                        if degree>10 and degree<5000:
                            l[neighbor] = l[w] + [(neighbor,degree)]
        return l
    def get_two_word_clue(self, word1, word2):
        word1_1 = self.get_distance_k_neighbors(word1,1)
        word2_1 = self.get_distance_k_neighbors(word2,1)
        possible_clues_1 = set(word1_1.keys()).intersection(set(word2_1.keys()))
        # if possible_clues:
        #     return list(possible_clues)
        # guesser = numberbatch_guesser.Guesser()
        # guesser.load_data()
        # word1_2_unfiltered = self.guesser.filter_valid_words(list(self.get_distance_k_neighbors(word1,2).keys()),(word1,word2))
        # scores1 = self.guesser.eval_inner_products(self.guesser.find_vecs([word1]),self.guesser.find_vecs(word1_2_unfiltered))[0]
        # word1_2 = []
        # for i in range(len(scores1)):
        #     if scores1[i] > 0.1:
        #         word1_2.append(word1_2_unfiltered[i])

        # word2_2_unfiltered = self.guesser.filter_valid_words(list(self.get_distance_k_neighbors(word2,2).keys()),(word1,word2))
        # scores2 = self.guesser.eval_inner_products(self.guesser.find_vecs([word2]),self.guesser.find_vecs(word2_2_unfiltered))[0]
        # word2_2 = []
        # for i in range(len(scores2)):
        #     if scores2[i] > 0.1:
        #         word2_2.append(word2_2_unfiltered[i])
        # possible_clues = set(word1_1).intersection(set(word2_2)).union(set(word1_2).intersection(set(word2_1)))
        word1_2 = self.guesser.filter_valid_words(list(self.get_distance_k_neighbors(word1,2).keys()))
        word2_2 = self.guesser.filter_valid_words(list(self.get_distance_k_neighbors(word2,2).keys()))
        possible_clues_2 = set(word1_1).intersection(set(word2_2)).union(set(word1_2).intersection(set(word2_1)))
        possible_clues = possible_clues_1.union(possible_clues_2)
        return list(possible_clues)
        

if __name__=="__main__":
    # g = ConceptNetGraph()
    # g.parse_graph()
    g = ConceptNetGraph()
    g.load_graph()
    g.guesser.load_data()
    print( ("play", "angel"), g.get_two_word_clue("play","angel") )
    # clues = g.get_two_word_clue("nurse","ambulance")
    # print(g.guesser.filter_valid_words(clues, ('nurse','ambulance')))
    # clues = g.guesser.filter_valid_words(clues, ('nurse','ambulance'))
    # scored_clues = g.guesser.score_clues(["nurse","ambulance"],clues)
    # print(scored_clues[0][:1])
    # print( g.guesser.eval_inner_products(g.guesser.find_vecs(["bridge"]),g.guesser.find_vecs(["card"]))[0] )
    # print( g.guesser.eval_inner_products(g.guesser.find_vecs(["deck"]),g.guesser.find_vecs(["card"]))[0] )

    # print( g.guesser.eval_inner_products(g.guesser.find_vecs(["bridge"]),g.guesser.find_vecs(["pier"]))[0] )
    # print( g.guesser.eval_inner_products(g.guesser.find_vecs(["deck"]),g.guesser.find_vecs(["pier"]))[0] )

    """with open("out.txt","w",encoding="utf8") as f:
        f.write(str(g.get_distance_k_neighbors("keyboard",2)))"""

    """with open("codewords_simplified.txt") as f: 
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
                    clues = g.guesser.filter_valid_words(clues)
                    if clues:
                        scored_clues = g.guesser.score_clues([w,w2],clues)
                        print((w,w2), scored_clues[0][:5])
                    else:
                        print((w,w2))"""
