import csv

# def read(filename):
# 	result = []
# 	with open(filename) as tsvfile:
# 		reader = csv.reader(tsvfile, delimiter='\t')
#   		for row in reader:
# 			result.append(row)
# 	return result

# pos = read('mammal_closure_pos.tsv')
# neg = read('mammal_closure_neg.tsv')

# with open('mammal.tsv', 'w') as outputfile:
# 	tsv_writer = csv.writer(outputfile, delimiter='\t')
# 	for p in pos:
# 		tsv_writer.writerow(p+['1'])
# 	for n in neg:
# 		tsv_writer.writerow(n+['0'])

from collections import defaultdict

result = []
vocab = defaultdict()
i=0
with open('./wordnet_valid.txt') as inputfile:
	lines = inputfile.readlines()
	for line in lines:
		line = line.strip()
		parts = line.split('\t')
		if parts[1] not in vocab:
			vocab[parts[1]] = i
			i+=1
		if parts[2] not in vocab:
			vocab[parts[2]] = i
			i+=1
		result.append((vocab[parts[2]], vocab[parts[1]], parts[3]))

with open('wordnet.tsv', 'w') as outputfile:
	tsv_writer = csv.writer(outputfile, delimiter='\t')
	for r in result:
		tsv_writer.writerow(r)

with open('wordnet_vocab.tsv', 'w') as outputfile:
	tsv_writer = csv.writer(outputfile, delimiter = '\t')
	for key in vocab:
		tsv_writer.writerow((vocab[key], key))

