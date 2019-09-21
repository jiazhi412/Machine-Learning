import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	# Edit here
	# state dict
	state_dict = dict()
	for index,tag in enumerate(tags):
		state_dict[tag] = index

	# obs dict
	obs_dict = dict()
	index2 = 0
	for sentences in train_data:
		for word in sentences.words:
			if word not in obs_dict:
				obs_dict[word] = index2
				index2 += 1

	#initial state
	pi = np.zeros(len(state_dict))
	for i in range(len(tags)):
		for sentences in train_data:
			if sentences.tags[0] == tags[i]:
				pi[i] += 1
	pi /= len(train_data)

	# transition
	A = np.zeros([len(tags),len(tags)])
	num_transition_from = np.zeros(len(tags))
	for tag in tags:
		for sentences in train_data:
			m = 0
			while m < len(sentences.tags) - 1:
				if sentences.tags[m] == tag:
					num_transition_from[state_dict[tag]] += 1
				m += 1

	# for tag1 in tags:
	# 	for tag2 in tags:
	# 		for sentences in train_data:
	# 			m = 0
	# 			while m < len(sentences.tags) - 1:
	# 				if sentences.tags[m] == tag1 and sentences.tags[m + 1] == tag2:
	# 					A[state_dict[tag1]][state_dict[tag2]] += 1
	# 				m += 1

	for sentences in train_data:
		for m in range(len(sentences.tags)-1):
			A[state_dict[sentences.tags[m]]][state_dict[sentences.tags[m+1]]] += 1
	A = np.transpose(A.transpose() / num_transition_from)

	# state-outcome
	B = np.zeros([len(tags),len(obs_dict)])
	num_states = np.zeros(len(tags))
	for tag in tags:
		for sentences in train_data:
			for m in range(len(sentences.tags)):
				if sentences.tags[m] == tag:
					num_states[state_dict[tag]] += 1

	# obs = list(obs_dict.keys())
	# for tag in tags:
	# 	for ob in obs:
	# 		for sentences in train_data:
	# 			for m in range(len(sentences.tags)):
	# 				if sentences.tags[m] == tag and sentences.words[m] == ob:
	# 						B[state_dict[tag]][obs_dict[ob]] += 1

	for sentences in train_data:
		for m in range(len(sentences.tags)):
			B[state_dict[sentences.tags[m]]][obs_dict[sentences.words[m]]] += 1
	B = np.transpose(B.transpose() / num_states)

	model = HMM(pi, A, B, obs_dict, state_dict)
	###################################################
	return model

# TODO:
def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here
	# for sentences in test_data:
	# 	for word in sentences.words:
	# 		if word not in
	# for sentences in test_data:
	# 	model.modify(sentences.words)
	for sentences in test_data:
		tagging.append(model.viterbi(sentences.words))
	###################################################
	return tagging
