import torch
import numpy as np

def ste(x):
    hard_x = torch.round(x)
    return x + (hard_x - x).detach()
    pass

def noisy_ste(x,width=0.):
    noise = (torch.zeros_like(x).uniform_() - 0.5)*width
    noisy_x = x + noise
    hard_x = noisy_x
    hard_x[noisy_x > 0.5] = 1.
    hard_x[noisy_x <= 0.5] = 0.

    return x + (hard_x - x).detach()
    pass

class SignumSTE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
    # @staticmethod
    # def _act(x,condition,out):
    #     out[condition] = (condition.float() - x[condition]).detach()  + x[condition]
    #     pass
    def forward(self,x):

        out = x.clone()
        # _act(x,x>0,out)
        # _act(x,x<0,out)
        b = out.shape[0]
        out[x>0] = (1.- x[x>0]).detach()  + x[x>0]
        print(torch.unique(out))
        out[x<0] = (-x[x<0]).detach()  + x[x<0]
        print(torch.unique(out))
        return out
        pass
