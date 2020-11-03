import torch
import wandb
import torch.nn as nn
from basic_box import Box
import torch.nn.functional as F
from torch.distributions import uniform

class SoftBox(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, min_init_value, delta_init_value, args):
        super(SoftBox, self).__init__()
        min_embedding = self.init_word_embedding(vocab_size, embed_dim, min_init_value)
        delta_embedding = self.init_word_embedding(vocab_size, embed_dim, delta_init_value)
        self.temperature = args.softplus_temp
        self.min_embedding = nn.Parameter(min_embedding)
        self.delta_embedding = nn.Parameter(delta_embedding)

    def forward(self, ids):
        """Returns box embeddings for ids"""
        min_rep = self.min_embedding[ids]
        delta_rep = self.delta_embedding[ids]
        max_rep = min_rep+torch.exp(delta_rep)
        # print('min', min_rep.mean())
        # print('delta', torch.exp(delta_rep.mean()))
        # wandb.log(self.min_embedding.mean())
        # wandb.log(self.delta_embedding.mean())
        boxes1 = Box(min_rep[:, 0, :], max_rep[:, 0, :])
        boxes2 = Box(min_rep[:, 1, :], max_rep[:, 1, :])
        pos_predictions = self.get_cond_probs(boxes1, boxes2)
        neg_prediction = torch.ones(pos_predictions.size()).to('cuda:0')-pos_predictions
        prediction = torch.stack([pos_predictions, neg_prediction], dim=1)
        return prediction

    def volumes(self, boxes):
        return F.softplus(boxes.delta_embed, beta=self.temperature).prod(1)

    def intersection(self, boxes1, boxes2):
        intersections_min = torch.max(boxes1.min_embed, boxes2.min_embed)
        intersections_max = torch.min(boxes1.max_embed, boxes2.max_embed)
        intersection_box = Box(intersections_min, intersections_max)
        return intersection_box

    def get_cond_probs(self, boxes1, boxes2):
        log_intersection = torch.log(torch.clamp(self.volumes(self.intersection(boxes1, boxes2)), 1e-10, 1e4))
        log_box2 = torch.log(torch.clamp(self.volumes(boxes2), 1e-10, 1e4))
        return torch.exp(log_intersection-log_box2)

    def init_word_embedding(self, vocab_size, embed_dim, init_value):
        distribution = uniform.Uniform(init_value[0], init_value[1])
        box_embed = distribution.sample((vocab_size, embed_dim))
        return box_embed


