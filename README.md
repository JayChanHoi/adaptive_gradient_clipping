# Adaptive Gradient Clipping

Adaptive gradient clipping is first introduced in the NFNets. The reason behind agc is to improve the weakness of gradient 
clipping which takes a static gradient threshold to clip. But most of the cases this threshold is sensitive to different setting 
of the hyperparameter and the parameter itself. agc allows the glip threshold to change according to the norm ratio between 
parameter and gradient. This can be treated as a relaxation of gradient clip. 

## Story and Theory behind the scence
ToDo

## How To Use
    # to use this repo, just clone the repo in your own project/repo and import as below. the import path may be changed due to your 
    # repo structure.

    import torch
    from adaptive_gradient_clipping.src.agc import AGC
    
    # model here is any model with torch.nn.Module format
    
    optimizer = torch.optim.Adam(model.parameter(), 0.0001)
    optimizer = AGC(optimizer, 0.04, 0.003)

    # use the agc optimizer in your normal training script as a normal optimizer in torch.optim

##To-Do
- [x] agc wrapper implementation
