import time
import wandb
import torch
import math
import argparse
from torch.utils.data import DataLoader
from utils import *
from dataset import *
from softbox import SoftBox
from gumbel_box import GumbelBox

box_model = {'softbox': SoftBox,
             'gumbel': GumbelBox}

def random_negative_sampling(samples, probs, vocab_size, ratio, max_num_neg_sample):
	with torch.no_grad():
		negative_samples = samples.repeat(ratio, 1)[:max_num_neg_sample, :]
		negative_samples[:, 1].random_(0, vocab_size)
		negative_probs = torch.zeros(negative_samples.size()[0], dtype=torch.long)
		samples = torch.cat([samples, negative_samples], dim=0)
		probs = torch.cat([probs, negative_probs], dim=0)
	return samples, probs

def train_func(train_data, vocab_size, random_negative_sampling_ratio, optimizer, criterion, device, batch_size, model):
	pos_batch_size = math.ceil(batch_size/(random_negative_sampling_ratio+1))
	max_neg_batch_size = batch_size - pos_batch_size

	# Train the model
	train_loss = 0
	train_acc = 0
	train_size = 0
	data = DataLoader(train_data, batch_size=pos_batch_size, shuffle=True)
	for ids, cls in data:
		optimizer.zero_grad()
		ids_aug, cls_aug = random_negative_sampling(ids, cls, vocab_size, random_negative_sampling_ratio, max_neg_batch_size)
		ids_aug, cls_aug = ids_aug.to(device), cls_aug.to(device)
		output = model(ids_aug)
		loss = criterion(output, cls_aug)
		train_loss += loss.item()
		loss.backward()
		optimizer.step()
		train_acc += (output.argmax(1) == cls_aug).sum().item()
		train_size += ids_aug.size()[0]

	return train_loss / train_size, train_acc / train_size

def test(test_data, criterion, device, batch_size, model):
	loss = 0
	acc = 0
	scores= []
	true = 0
	data = DataLoader(test_data, batch_size=batch_size)
	for ids, cls in data:
		ids, cls = ids.to(device), cls.to(device)
		with torch.no_grad():
			output = model(ids)
			loss = criterion(output, cls)
			loss += loss.item()
			acc += (output.argmax(1) == cls).sum().item()
			scores.extend(output[:, 0])
			true+=cls.sum()

	return loss / len(test_data), acc / len(test_data)


def main(args):
	wandb.init(project="basic_box", config=args)

	train_dataset = PairDataset(args.train_data_path)
	test_dataset = PairDataset(args.test_data_path)
	word2idx = get_vocab(args.vocab_path)

	VOCAB_SIZE = len(word2idx)
	NUN_CLASS = 2

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = box_model[args.model](VOCAB_SIZE, args.box_embedding_dim, NUN_CLASS, [1e-4, 0.2], [-0.1, 0], args).to(device)


	wandb.watch(model)
	min_valid_loss = float('inf')

	criterion = torch.nn.CrossEntropyLoss().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


	for epoch in range(args.epochs):

		start_time = time.time()
		train_loss, train_acc = train_func(train_dataset, VOCAB_SIZE, args.random_negative_sampling_ratio,
										   optimizer, criterion, device, 2**args.log_batch_size, model)
		valid_loss, valid_acc = test(test_dataset, criterion, device, 2**args.log_batch_size, model)

		wandb.log({'train loss': train_loss, 'train accuracy': train_acc, 'valid loss': valid_loss, 'valid accuracy': valid_acc})

		secs = int(time.time() - start_time)
		mins = secs / 60
		secs = secs % 60

		print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
		print(f'\tLoss: {train_loss:.8f}(train)\t|\tAcc: {train_acc * 100:.2f}%(train)')
		print(f'\tLoss: {valid_loss:.8f}(valid)\t|\tAcc: {valid_acc * 100:.2f}%(valid)')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data_path', type=str, default='./data/full_wordnet/full_wordnet_noneg.tsv', help='path to train data')
	parser.add_argument('--test_data_path', type=str, default='./data/full_wordnet/full_wordnet.tsv', help='path to test data')
	parser.add_argument('--vocab_path', type=str, default='./data/full_wordnet/full_wordnet_vocab.tsv', help='path to vocab')
	parser.add_argument('--log_batch_size', type=int, default=13, help='batch size for training will be 2**LOG_BATCH_SIZE')
	parser.add_argument('--learning_rate', type=float, default=5e-3, help='learning rate')
	parser.add_argument('--box_embedding_dim', type=int, default=40, help='box embedding dimension')
	parser.add_argument('--softplus_temp', type=float, default=1.0, help='beta of softplus function')
	parser.add_argument('--random_negative_sampling_ratio', type=int, default=0, help='sample this many random negatives for each positive.')
	parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
	parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training (eg. no nvidia GPU)')

	parser.add_argument('--model', type=str, default='softbox', help='model type: choose from softbox, gumbel')
	# gumbel box parameter
	parser.add_argument('--gumbel_beta', type=float, default=1.0, help='beta value for gumbel distribution')
	parser.add_argument('--scale', type=float, default=1.0, help='scale value for gumbel distribution')


	args = parser.parse_args()
	main(args)

