import torch
from functorch import vmap
from torch import nn
from torch import vmap


class FeatureTransitionModel(nn.Module):
    def __init__(self, n_features, n_layers, lr=0.001):
        super().__init__()

        self.params = nn.Parameter(torch.zeros(n_layers - 1, n_features, n_features))
        torch.nn.init.kaiming_uniform_(self.params)
        self.opt = torch.optim.Adam([self.params], lr=lr, foreach=False)

    @staticmethod
    def fmodel(params, x):
        return x @ params.T

    def train_on(self, acts):
        """
        :param acts: [num_layers, batch_size, n_features]
        """
        out = vmap(self.fmodel)(self.params, acts[:-1])
        loss = ((out - acts[1:]) ** 2).mean()
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        return loss.item()