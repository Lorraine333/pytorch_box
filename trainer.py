import time
import wandb
import torch
import argparse
from torch.utils.data import DataLoader
from utils import *
from dataset import *
from softbox import SoftBox
from gumbel_box import GumbelBox

box_model = {'softbox': SoftBox,
             'gumbel': GumbelBox}

def train_func(train_data, optimizer, criterion, device, batch_size, model):

	# Train the model
	train_loss = 0
	train_acc = 0
	data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
	for ids, cls in data:
		optimizer.zero_grad()
		ids, cls = ids.to(device), cls.to(device)
		output = model(ids)
		loss = criterion(output, cls)
		train_loss += loss.item()
		loss.backward()
		optimizer.step()
		train_acc += (output.argmax(1) == cls).sum().item()

	return train_loss / len(train_data), train_acc / len(train_data)

def test(test_data, optimizer, criterion, device, batch_size, model):
	loss = 0
	acc = 0
	scores= []
	true = 0
	all_labels_and_scores=[]
	# data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
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

	train_dataset = PairDataset(args.data_dir+'/wordnet.tsv')
	test_dataset = PairDataset(args.data_dir+'/wordnet.tsv')
	word2idx = get_vocab(args.data_dir+'/wordnet_vocab.tsv')

	VOCAB_SIZE = len(word2idx)
	NUN_CLASS = 2


	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# model = SoftBox(VOCAB_SIZE, args.box_embedding_dim, NUN_CLASS, [1e-4, 0.2], [-0.1, 0], args).to(device)
	model = box_model[args.model](VOCAB_SIZE, args.box_embedding_dim, NUN_CLASS, [1e-4, 0.2], [-0.1, 0], args).to(device)


	wandb.watch(model)
	min_valid_loss = float('inf')

	criterion = torch.nn.CrossEntropyLoss().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


	for epoch in range(args.epochs):

		start_time = time.time()
		train_loss, train_acc = train_func(train_dataset, optimizer, criterion, device, 2**args.log_batch_size, model)
		valid_loss, valid_acc = test(test_dataset, optimizer, criterion, device, 2**args.log_batch_size, model)

		wandb.log({'train loss':train_loss, 'train accuracy': train_acc, 'valid loss': valid_loss, 'valid accuracy': valid_acc})

		secs = int(time.time() - start_time)
		mins = secs / 60
		secs = secs % 60

		print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
		print(f'\tLoss: {train_loss:.8f}(train)\t|\tAcc: {train_acc * 100:.2f}%(train)')
		print(f'\tLoss: {valid_loss:.8f}(valid)\t|\tAcc: {valid_acc * 100:.2f}%(valid)')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='./data', help='location of data')
	parser.add_argument('--log_batch_size', type=int, default=12, help='batch size for training will be 2**LOG_BATCH_SIZE (default: 8)')
	parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning rate (default: 1)')
	parser.add_argument('--box_embedding_dim', type=int, default=32, help='box embedding dimension (default: 10)')
	parser.add_argument('--softplus_temp', type=float, default=1e-2, help='temperature of softplus function (default: 1)')
	parser.add_argument('--unary_loss_weight', type=float, default=1, help='weight for unary loss during training (default: 0.01)')
	parser.add_argument('--random_negative_sampling_ratio', type=int, default=1, help='sample this many random negatives for each positive.')
	parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 10)')
	parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training (eg. no nVidia GPU)')

	parser.add_argument('--model', type=str, default='gumbel', help='model type: choose from softbox, gumbel')
	# gumbel box parameter
	parser.add_argument('--gumbel_beta', type=float, default=1.0, help='beta value for gumbel distribution')
	parser.add_argument('--scale', type=float, default=1.0, help='scale value for gumbel distribution')


	args = parser.parse_args()
	main(args)



