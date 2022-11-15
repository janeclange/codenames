import tqdm
import pickle
import numberbatch_guesser
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
        with open("conceptnet-assertions-en","rb") as f:
            return pickle.load(f)
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
                        if degree>10 and degree<1000:
                            l[neighbor] = l[w] + [(neighbor,degree)]
        return l
    def get_two_word_clue(self, word1, word2):
        word1_1 = self.get_distance_k_neighbors(word1,1)
        word2_1 = self.get_distance_k_neighbors(word2,1)
        possible_clues = set(word1_1.keys()).intersection(set(word2_1.keys()))
        if possible_clues:
            return list(possible_clues)
        word1_2 = self.get_distance_k_neighbors(word1,2)
        word2_2 = self.get_distance_k_neighbors(word2,2)
        possible_clues = set(word1_1.keys()).intersection(set(word2_2.keys())).union(set(word1_2.keys()).intersection(set(word2_1.keys())))
        return list(possible_clues)
        

if __name__=="__main__":
    #g = ConceptNetGraph()
    #g.parse_graph()
    g = ConceptNetGraph.load_graph()
    guesser = numberbatch_guesser.Guesser()
    guesser.load_data()
    with open("out.txt","w",encoding="utf8") as f:
        f.write(str(g.get_distance_k_neighbors("keyboard",2)))

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
