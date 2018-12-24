import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np


def majority_vote(results):
    votes = defaultdict(int)
    for res in results:
        votes[res] += 1
    max = 0
    max_key= None
    # print(votes)
    for key,value in votes.items():
        if value>max:
            max = value
            max_key = key
    return max_key


class EnsembleModule(nn.Module):
    def __init__(self, net):
        super(EnsembleModule, self).__init__()
        self.net = net

    def forward(self, x):
        s = x.shape
        images_batches = x.view(s[0] * s[1], s[2], s[3], s[4])
        predictions = self.net(images_batches)
        predictions = predictions.view(s[0], s[1], -1)
        return self.forward_inner(predictions)

    def forward_inner(self, predictions):
        raise NotImplementedError('Base Class !!!')


class MajorityEnsembleModule(EnsembleModule):
    def __init__(self, net, class_index_input=False):
        super(MajorityEnsembleModule, self).__init__(net)
        self.class_index_input = class_index_input

    def forward_inner(self, predictions):
        if not self.class_index_input:
            predictions = predictions.argmax(-1)
        else:
            predictions = predictions.squeeze(-1)
        predictions = predictions.cpu().numpy()
        return [majority_vote(predictions[i]) for i in range(predictions.shape[0])]


class AverageEnsembleModule(EnsembleModule):
    def forward_inner(self, predictions):
        predictions = F.softmax(predictions, -1)
        s = predictions.shape
        summed_predictions = predictions.view(s[0], s[1], -1).sum(1).cpu().numpy()
        return [np.argmax(summed_predictions[i]) for i in range(s[0])]
