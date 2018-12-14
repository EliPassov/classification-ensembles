import torch
from torch import nn
from collections import defaultdict

class TensorEnsembleModule(nn.Module):
    def __init__(self):
        super(TensorEnsembleModule, self).__init__()

    def forward(self, x):
        if isinstance(x, list):
            results = [res.unsqueeze(0) for res in x]
            x = torch.cat(tuple(results), 0)
        assert isinstance(x, torch.Tensor)
        return self.ensemble(x)

    def ensemble(self, x):
        raise NotImplementedError('Base class method')


class AverageEnsemble(TensorEnsembleModule):
    def ensemble(self, x):
        return x.mean(0)


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


class AugmentedEnsembleNet(nn.Module):
    def __init__(self, net, ensemble_module):
        super(AugmentedEnsembleNet, self).__init__()
        self.net = net
        self.ensemble_module = ensemble_module

    def forward(self, imgs):
        assert isinstance(imgs, list)
        results = []
        for img in imgs:
            results.append(self.net(img))
        return self.ensemble_module(results)
