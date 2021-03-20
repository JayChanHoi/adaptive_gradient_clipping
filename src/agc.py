import torch
from .utils import unitwisenorm

class AGC():
    def __init__(self, optimizer, layer_to_skip, clip_lambda=0.02, eps=0.001):
        '''
        :param layer_to_skip: a list contain the layer name of stat_dict to skip
        :param optimizer: any SGD variants optimizer from torch.optim
        :param clip_lambda: clip factor ranging from 0.01 to 0.16. larger batch size should use smaller clip facotr.
            for instance, batch size = 1024 -> clip_lambda = 0.01
        :param eps: default 1e-3
        '''
        self.optimizer = optimizer
        self.clip_lambda = clip_lambda
        self.eps = eps
        self.layer_to_skip = layer_to_skip

    @torch.no_grad()
    def step(self):
        for group in self.optimizer.param_groups:
            for para in group['params']:
                # if the para group is included in layer_to_skip, just skip
                if para in self.layer_to_skip:
                    continue

                # if no grad, just skip
                if para.grad is None:
                    continue

                para_norm = torch.max(unitwisenorm(para.detach()), torch.tensor(self.eps, device=para.device))
                grad_norm = unitwisenorm(para.grad.detach())
                trigger = (grad_norm / para_norm) > self.clip_lambda
                clip_grad = self.clip_lambda * (para_norm / torch.max(grad_norm, torch.tensor(1e-8, device=para.device))) * para.grad

                para.grad.detach().data.copy_(torch.where(trigger, clip_grad, para.grad))

        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()