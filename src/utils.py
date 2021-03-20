import torch

def unitwisenorm(x):
    '''
    :param x: gradient tensor or parameter tensor
    :return: norm tensor retain the same dim as the input tensor
    '''
    if x.ndim <= 1:
        keepdim=False
        sum_dim = 0
    elif x.ndim in [2, 3]:
        keepdim=True
        sum_dim = 0
    elif x.ndim == 4:
        keepdim = True
        sum_dim = [1, 2, 3]
    else:
        raise ValueError('the input tensor dimension for unitwisenorm should be less or equal to 4')

    return torch.sum(x**2, dim=sum_dim, keepdim=keepdim)**(0.5)