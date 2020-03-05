import csv
from random import *
from collections import defaultdict

random_seed = 1

def get_vocab(filename):
	word2idx = defaultdict()
	with open(filename) as inputfile:
		lines = inputfile.readlines()
		for line in lines:
			line = line.strip()
			parts = line.split('\t')
			word2idx[parts[1]] = parts[0]
	return word2idx