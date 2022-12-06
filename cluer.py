import tqdm
import pickle
import numberbatch_guesser
from conceptnet import ConceptNetGraph
from conceptnet import powersetify
from itertools import permutations
import numpy as np
import random
import graphglove


USE_GRAPH_GLOVE = True
lower = lambda l:[x.lower() for x in l]
class Cluer:
	def __init__(self):
		self.g = ConceptNetGraph()
		self.g.load_graph()
		self.g.guesser.load_data()
		self.clues = None

		if USE_GRAPH_GLOVE:
			print("loading graphglove")
			self.graphglove = graphglove.GraphGlove()
	def new_game(self, assassin = [], red_words = [], blue_words = [], bystanders = []):

		if USE_GRAPH_GLOVE:
			print("precompting distances for graphglove")
			self.graphglove.precompute_pairwise_dist(blue_words)

		self.assassin, self.red_words, self.blue_words, self.bystanders = lower(assassin), lower(red_words), lower(blue_words), lower(bystanders)
		self.previous_guesses = []
		self.previous_clues = []
		self.ally_words_remaining = 8

	def precompute(self):
		self.clues = {}

		with open("codewords_simplified.txt") as f: 
			codewords = [x.strip() for x in f.readlines()]
			# codewords = codewords[:5]
			for i in range(len(codewords)):
				print(i+1, 'out of', len(codewords))
				w = codewords[i].replace(" ","_").lower()
				for j in range(i+1, len(codewords)):
					w2 = codewords[j].replace(" ","_").lower()
					#print((w,w2),g.get_two_word_clue(w,w2))
					clues = self.g.get_two_word_clue(w,w2)
					#print([w,w2])
					#print(clues)
					clues = self.g.guesser.filter_valid_words(clues)
					if clues:
						scored_clues = self.g.guesser.score_clues([w,w2],clues)
						self.clues[(w,w2)] = scored_clues[0][:5]
					else:
						self.clues[(w,w2)] = None

	def save_clues(self):
		with open('clues.pkl', 'wb') as f:
			pickle.dump(self.clues, f)

	def load_clues(self):
		with open('clues.pkl', 'rb') as f:
			cl = pickle.load(f)
		self.clues = cl
		# print(self.clues[('nurse','ambulance')])

	def generate_clues(self, word_tup):
		clues = self.g.get_k_word_clue(tuple(word_tup))
		if (len(word_tup) == 2):
			clues = [w for (x,w) in clues if x <=3]
		else:
			clues = [w for (x,w) in clues]
		return self.g.guesser.score_clues(word_tup, clues)[0][:10]

	def lower(self, array):
		return [a.lower() for a in array]

	def generate_clues_partition(self, word_tup):
		if len(word_tup) == 2:
			clues = self.g.get_two_word_clues(word_tup[0], word_tup[1])
			return self.g.guesser.score_clues(word_tup, clues)[0][:5]
		elif len(word_tup) == 1:
			return []
		
		# if (tuple(word_tup) in clues) and clues[tuple(word_tup)] is not None:
		# 	if (clues[tuple(word_tup)][0]):
		# 		return self.clues[tuple(word_tup)]
		# 	else:
		# 		return [None]
		# else:
		# 	return [None]
		
	def evaluate_tup(self, word_tup, board, clue):
		trials = 500
		# print("evaluating tuple: ", word_tup)
		if clue == None:
			return None
		turn_counts = []
		board_words_ordered, inner_prods = self.g.guesser.guess(clue, board, len(board))
		for i in range(trials):
			#print(i, " out of ", trials)
			#print(tuple(word_tup))
			#print("clue", clue)
			noise = np.random.normal(0,0.05,size=(len(board)))
			scores = np.array(inner_prods) + np.array(noise)
			#print("board", board_words_ordered)
			#print("scores", scores)
			guess_inds = np.argsort(scores)[-1*len(word_tup):][::-1]
			guesses = board_words_ordered[guess_inds]
			#print("guesses", guesses)

			# evaluate guesses
			turns = None
			intended_set = set(word_tup)
			guess_set = set(guesses)
			if intended_set == guess_set:
				turns = 1		# effective turns: 1
				#print("Exact!")
			elif self.assassin[0] in guess_set:
				turns = 25
			elif len(guess_set.intersection(set(self.blue_words))) == 2:
				#print("Two positive")
				turns = 1.5 	# effective turns: slightly more than 1, because you might have messed up your partition
			elif (len(guess_set.intersection(set(self.blue_words))) == 1) and (len(guess_set.intersection(set(self.bystanders))) == 1):
				#print("One pos, one neutral")
				turns = 2 	# effective turns: 1 turn actually passes, +1 penalty because you'll have to reclue the other word
			elif len(guess_set.intersection(set(self.bystanders))) == 2:
				#print("Two neutral")
				turns = 3 		# 1 turn actually passes, and you'll have to reclue both ally words
			elif (len(guess_set.intersection(set(self.blue_words))) == 1) and (len(guess_set.intersection(set(self.red_words))) == 1):
				#print("One pos, one neg")
				turns = 3 		# effective turns: 1 turn passes, +1 for negative word, and you'll also have to reclue the other ally word
			elif (len(guess_set.intersection(set(self.bystanders))) == 1) and (len(guess_set.intersection(set(self.red_words))) == 1):
				#print("One neutral, one neg")
				turns = 4 		# effective turns: 1 turn passes, +1 for negative word, and you'll have to reclue both ally words
			elif len(guess_set.intersection(set(self.red_words))) == 2:
				#print("Two negative")
				turns = 5
				# 1 turn actually passes, 2 turn points for hitting negative words, and you'll have to reclue both ally words
			elif len(guess_set.intersection(set(self.assassin))) == 1:
				#print("Assassin")
				turns = 25
				# doesn't really matter what the penalty is exactly at that point
			else:
				print("Something went wrong...")
				print(guess_set)
			# turns -= 0.2 # make it slightly prefer 2-word clues
			turn_counts.append(turns)
		avg_turns = np.average(turn_counts)
		#print("raw score for ", word_tup, ": ", avg_turns)
		if avg_turns >= 2 and len(word_tup)>1:
			avg_turns = 2 	# we should just give one word hints at that point
		return avg_turns

	def evaluate_tup_greedy(self, word_tup, board, clue, trials=500):
		# print("evaluating tuple: ", word_tup)
		if clue == None:
			return None
		turn_counts = []
		board_words_ordered, inner_prods = self.g.guesser.guess(clue, board, len(board))
		for i in range(trials):
			#print(i, " out of ", trials)
			#print(tuple(word_tup))
			#print("clue", clue)
			
			noise = np.random.normal(0,0.05,size=(len(board)))
			scores = np.array(inner_prods) + np.array(noise)
			#print("board", board_words_ordered)
			#print("scores", scores)
			guess_inds = np.argsort(scores)[-1*len(word_tup):][::-1]
			guesses = board_words_ordered[guess_inds]
			#print("guesses", guesses)

			# evaluate guesses
			turns = None
			intended_set = set(word_tup)
			guess_set = set(guesses)
			if intended_set == guess_set:
				turns = 1 - len(guess_set) 		# effective turns: 1
				#print("Exact!")
			elif self.assassin[0] in guess_set:
				turns = 25
			else:
				# simulate user guessing
				# score = 1
				# done = False
				# while (i < len(guesses)) and (not done):
				# 	if guesses[i] in self.blue_words:
				# 		score -= 1
				# 	elif guesses[i] in self.bystanders:
				# 		done = True
				# 	elif guesses[i] in self.red_words:
				# 		score += 1
				# 		done = True
				# 	i += 1
				# turns = score
				good_score = len(guess_set.intersection(set(self.blue_words)))
				if good_score == len(guess_set):
					good_score -= .5
				neutral_score = len(guess_set.intersection(set(self.bystanders)))
				# if neutral_score>0:
				# 	neutral_score -= 1
				enemy_score = len(guess_set.intersection(set(self.red_words)))
				turns = 1 - good_score + 2*enemy_score
			"""
			elif len(guess_set.intersection(set(self.blue_words))) == 2:
				#print("Two positive")
				turns = 1.5 	# effective turns: slightly more than 1, because you might have messed up your partition
			elif (len(guess_set.intersection(set(self.blue_words))) == 1) and (len(guess_set.intersection(set(self.bystanders))) == 1):
				#print("One pos, one neutral")
				turns = 2 	# effective turns: 1 turn actually passes, +1 penalty because you'll have to reclue the other word
			elif len(guess_set.intersection(set(self.bystanders))) == 2:
				#print("Two neutral")
				turns = 3 		# 1 turn actually passes, and you'll have to reclue both ally words
			elif (len(guess_set.intersection(set(self.blue_words))) == 1) and (len(guess_set.intersection(set(self.red_words))) == 1):
				#print("One pos, one neg")
				turns = 3 		# effective turns: 1 turn passes, +1 for negative word, and you'll also have to reclue the other ally word
			elif (len(guess_set.intersection(set(self.bystanders))) == 1) and (len(guess_set.intersection(set(self.red_words))) == 1):
				#print("One neutral, one neg")
				turns = 4 		# effective turns: 1 turn passes, +1 for negative word, and you'll have to reclue both ally words
			elif len(guess_set.intersection(set(self.red_words))) == 2:
				#print("Two negative")
				turns = 5
				# 1 turn actually passes, 2 turn points for hitting negative words, and you'll have to reclue both ally words
			elif len(guess_set.intersection(set(self.assassin))) == 1:
				#print("Assassin")
				turns = 25
				# doesn't really matter what the penalty is exactly at that point
			else:
				print("Something went wrong...")
				print(guess_set)
			# turns -= 0.2 # make it slightly prefer 2-word clues
			"""
			turn_counts.append(turns)
		avg_turns = np.average(turn_counts)
		#print("raw score for ", word_tup, ": ", avg_turns)
		if avg_turns >= 0 and len(word_tup)>1:
			avg_turns = 0 	# we should just give one word hints at that point
		return avg_turns

	def clue_partitions(self):
		# for w in self.previous_guesses:
		# 	if w in self.blue_words:
		# 		self.blue_words.remove(w)

		board = self.blue_words + self.red_words + self.bystanders + self.assassin
		board = [w for w in board if w not in self.previous_guesses]

		remaining_blue_words = [w for w in self.blue_words if w not in self.previous_guesses]

		# brute force search over all possible partitions
		n_ally_words_left = len(remaining_blue_words)
		indices = [i for i in range(n_ally_words_left)] # pad to be even with -1
		if len(indices) % 2 != 0:
			indices.append(-1)
		p = []
		for perm in permutations(indices):
			p.append(sorted([sorted((perm[2*i], perm[2*i+1])) for i in range(len(indices)//2)]))
			# print(p[-1])
		partitions = np.unique(p, axis=0)

		partition_turn_counts = []

		# for each partition, for each pair/singleton in the partition, compute the behaviour of the guesser using 100(?) random trials
		for i in tqdm.tqdm(list(range(len(partitions)))):
			# print(i, " out of ", len(partitions), "partitions")
			partition = partitions[i]
			total_turns = 0
			for tup in partition:
				tup = [ind for ind in tup if ind != -1]
				word_tup = [remaining_blue_words[t] for t in tup]
				# for this tuple, find the best clue, set the score to the score for the best clue
				clues = self.generate_clues_partition(word_tup)
				if len(clues) == 0:
					total_turns += 2
					continue
				clue_scores = []
				for i in range(len(clues)):
					if clues[i] in self.previous_clues:
						clue_scores.append(np.Inf)
					else:
						clue_scores.append(self.evaluate_tup(word_tup, board, clues[i]))
					# print("current clue", clues[i])
				tup_score = np.min(clue_scores)
				total_turns += tup_score
			partition_turn_counts.append(total_turns)
		# use the partition with the lowest turn count (unless it really sucks, then we should hint singletons instead)
		best_partition = partitions[np.argmin(partition_turn_counts)]
		# calculate the best clue in this partition
		tup_scores = []
		best_clues = []
		# print("best partition", best_partition)
		for i in range(len(best_partition)):
			tup = best_partition[i]
			tup = [ind for ind in tup if ind != -1]
			word_tup = [remaining_blue_words[t] for t in tup]
			clues = self.generate_clues_partition(word_tup)
			# print("clues", clues)
			if len(clues) == 0:
				tup_scores.append(2)
				best_clues.append(None)
				continue
			clue_scores = []
			for i in range(len(clues)):
				if clues[i] in self.previous_clues:
					clue_scores.append(np.Inf)
				else:
					clue_scores.append(self.evaluate_tup(word_tup, board, clues[i]))
				# print("current clue", clues[i])
			# print("clue scores", clue_scores)
			tup_score = np.min(clue_scores)
			tup_scores.append(tup_score)
			best_clues.append(clues[np.argmin(clue_scores)])
			# print("clues", clues)
		# print("best clues", best_clues)
		best_tup = best_partition[np.argmin(tup_scores)]
		best_tup = [ind for ind in best_tup if ind != -1]
		word_best_tup = [remaining_blue_words[t] for t in best_tup]
		best_clue = best_clues[np.argmin(tup_scores)]
		best_clue_score = np.min(tup_scores)
		if best_clue and (best_clue_score < 2):
			clue = best_clue # use pair clue only if it exists and doesn't suck
			n_target = 2
		else:
			#clue = word_best_tup[0]
			clue = self.generate_clues(frozenset([word_best_tup[0]]))[0]

			n_target = 1
		# if (tuple(word_best_tup) in self.clues):
		# 	if (self.clues[tuple(word_best_tup)][0]) and (np.min(tup_scores) < 2): # use pair clue only if it exists and doesn't suck
		# 		clue = best_clues[np.argmin(tup_scores)]
		# 		n_target = 2
		# 	else:
		# 		clue = word_best_tup[0]		# change this later to actually give a one-word clue, for now we'll just assume one word clues are as good as hinting the word itself
		# 		n_target = 1
		# else:
		# 	clue = word_best_tup[0]		# change this later to actually give a one-word clue, for now we'll just assume one word clues are as good as hinting the word itself
		# 	n_target = 1
		# print(best_partition)
		# print(best_tup)
		# print(word_best_tup)
		self.word_best_tup = word_best_tup[:n_target]
		print(clue, n_target, best_clue_score)
		return (clue, n_target)


class Cluer2(Cluer):
	def generate_clues(self, word_tup):
		clues = self.g.get_k_word_clue(tuple(word_tup))
		if (len(word_tup) == 2):
			clues = [w for (x,w) in clues if x <=3]
		else:
			clues = [w for (x,w) in clues]
		if USE_GRAPH_GLOVE:
			clues = self.graphglove.graph_glove_clue(word_tup)
		return self.g.guesser.score_clues(word_tup, clues)[0][:10]
	def clue_greedy(self):
		board = self.blue_words + self.red_words + self.bystanders + self.assassin
		board = [w for w in board if w not in self.previous_guesses]

		remaining_blue_words = frozenset([w for w in self.blue_words if w not in self.previous_guesses])


		partitions = [x for x in powersetify(remaining_blue_words) if len(x)>0]
		best_clues = []
		partition_scores = []
		for p in tqdm.tqdm(partitions):
			possible_clues = self.generate_clues(list(p))
			# print(possible_clues)
			clue_scores = []
			if (len(possible_clues) > 0):
				for clue in possible_clues:
					t = (500 if len(remaining_blue_words)<6 else 250)
					score = self.evaluate_tup_greedy(p, board, clue,trials=t)
					if clue in self.previous_clues:
						clue_scores.append(np.Inf)
					else:
						clue_scores.append(score)
				best_clues.append(possible_clues[np.argmin(clue_scores)])
				partition_scores.append(np.min(clue_scores))
			else:
				best_clues.append(None)
				partition_scores.append(np.Inf)
		# print(partition_scores)
			# for clue in possible_clues:
			# 	clue_scores.append(self.evaluate_tup_greedy(p, board, clue,trials=(50 if len(remaining_blue_words)<6 else 25)))
			# best_clues.append(possible_clues[np.argmin(clue_scores)])
			# partition_scores.append(np.min(clue_scores))
		clue = best_clues[np.argmin(partition_scores)]
		best_score = np.min(partition_scores)
		target = partitions[np.argmin(partition_scores)]
		if best_score >= 0:
			#best we can do is give a one word clue.
			target = [random.choice(list(remaining_blue_words))]
			clue = self.generate_clues(target)[0]
		self.word_best_tup = target
		return (clue, len(target))
	def clue(self):
		board = self.blue_words + self.red_words + self.bystanders + self.assassin
		board = [w for w in board if w not in self.previous_guesses]

		remaining_blue_words = frozenset([w for w in self.blue_words if w not in self.previous_guesses])


		if (len(remaining_blue_words) > len(self.blue_words)):
			return self.clue_greedy()
		else:
			USE_GRAPH_GLOVE = True
			self.clue_partitions()
			USE_GRAPH_GLOVE = False
			return self.clue_partitions()







if __name__=="__main__":
	c = Cluer(assassin=["bell"], blue_words=["bridge","deck","pirate","jupiter"], red_words=["egypt","greece"],bystanders=["africa","air"])
	c.load_clues()
	print( c.clue() )
	# c = Cluer()
	# c.precompute()
	# c.save_clues()
	# bridge_deck_clues = c.clues[("bridge","deck")]
	# bridge_deck_clues = c.g.get_two_word_clue("bridge","deck")
	# print(bridge_deck_clues)
	# filtered_bridge_deck_clues = c.g.guesser.filter_valid_words(bridge_deck_clues, ("bridge","deck"))
	# print(filtered_bridge_deck_clues)
	# print(c.clues)