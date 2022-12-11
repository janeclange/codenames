import torch
import sys
import numpy as np
class GraphGlove():
    def __init__(self):
        sys.path.append("graph_glove")
        self.model = torch.load("graphglove_wiki50k_dist_20d.model.pth")
    def precompute_pairwise_dist(self, words):
        self.words = list(words)
        ixs = [self.model.token_to_ix[w] for w in words if w in self.model.token_to_ix]
        self.dists = self.model.graph_embedding.compute_pairwise_distances(indices=ixs)
        
    def graph_glove_clue(self, words, n=20, verbose=False):
        #ixs = [self.model.token_to_ix[w] for w in words]
        #dists = self.model.graph_embedding.compute_pairwise_distances(indices=ixs)
        ixs = np.array([self.words.index(x) for x in words])
        dists = self.dists[ixs,:]
        clue_scores = np.max(dists, axis=0)
        clueix = np.argmin(clue_scores)
        clues = np.argsort(clue_scores)[:n]
        if verbose:
            print([(self.model.ix_to_token[w], dists[:,w]) for w in clues])
            print(dists[:,clueix])
        #return self.model.ix_to_token[clueix]
        return [self.model.ix_to_token[w] for w in clues if w<50000]

def load_graph_glove():
    sys.path.append("graph_glove")
    model = torch.load("graph_glove/graphglove_wiki50k_dist_20d.model.pth")
    print(model.token_to_ix["part"])

if __name__=="__main__":
    g = GraphGlove()
    clue = g.graph_glove_clue(["mouse","dog","cat","platypus","honey"])
    print(clue)