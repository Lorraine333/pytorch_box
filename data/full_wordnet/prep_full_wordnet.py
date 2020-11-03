import csv
from collections import defaultdict

result = []
vocab = defaultdict()
i=0
with open('./basic_100train.txt') as inputfile:
	lines = inputfile.readlines()
	for line in lines:
		line = line.strip()
		parts = line.split('\t')
		if parts[1] not in vocab:
			vocab[parts[1]] = i
			i+=1
		if parts[3] not in vocab:
			vocab[parts[3]] = i
			i+=1
		result.append((vocab[parts[3]], vocab[parts[1]], parts[4]))

with open('full_wordnet.tsv', 'w') as outputfile:
	tsv_writer = csv.writer(outputfile, delimiter='\t')
	for r in result:
		tsv_writer.writerow(r)

with open('full_wordnet_noneg.tsv', 'w') as outputfile:
	tsv_writer = csv.writer(outputfile, delimiter='\t')
	for r in result:
		if r[-1] == "1":
			tsv_writer.writerow(r)

with open('full_wordnet_vocab.tsv', 'w') as outputfile:
	tsv_writer = csv.writer(outputfile, delimiter = '\t')
	for key in vocab:
		tsv_writer.writerow((vocab[key], key))