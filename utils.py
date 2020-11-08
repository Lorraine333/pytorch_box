import csv
import torch
from random import *
from scipy import special
from collections import defaultdict

random_seed = 1

class ExpEi(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input):
		ctx.save_for_backward(input)
		dev = input.device
		with torch.no_grad():
			x = special.exp1(input.detach().cpu()).to(dev)
			input.to(dev)
		return x

	@staticmethod
	def backward(ctx, grad_output):
		input, = ctx.saved_tensors
		grad_input = grad_output*(-torch.exp(-input)/input)
		return grad_input

def get_vocab(filename):
	word2idx = defaultdict()
	with open(filename) as inputfile:
		lines = inputfile.readlines()
		for line in lines:
			line = line.strip()
			parts = line.split('\t')
			word2idx[parts[1]] = parts[0]
	return word2idx

def restore_from(model, optimizer, ckpt_path):
	device = torch.cuda.current_device()
	ckpt = torch.load(ckpt_path)
	ckpt_args = ckpt['args']
	epoch = ckpt['epoch']

	model.load_state_dict(ckpt['state_dict'], strict=True)
	optimizer.load_state_dict(ckpt['optimizer'])
	return model, optimizer, ckpt_args, epoch