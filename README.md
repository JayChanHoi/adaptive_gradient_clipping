# Adaptive Gradient Clipping

Adaptive gradient clipping is first introduced in the [NFNets](https://arxiv.org/pdf/2102.06171.pdf). The reason behind agc is to improve the weakness of gradient 
clipping which takes a static gradient threshold to clip. But most of the cases this threshold is sensitive to different setting 
of the hyperparameter and the parameter itself. agc allows the glip threshold to change according to the norm ratio between 
parameter and gradient. This can be treated as a relaxation of gradient clip. 

## Brief theory explained behind the scene
When using gradient descent optimization algo to train a neural network, the training process will be expected to be unstable 
when ratio between parameter change and parameter are large. So, we can consider to use the ratio of gradient norm to parameter norm 
to measure whether the update will be unstable or not for specific layer of the network. Also the ratio can also act as a factor 
to contorl how the update size should be. If the ratio is too large, the clipping effect will be stronger to stablize the 
gradient update.

## How To Use
    # to use this repo, just clone the repo in your own project/repo and import as below. the import path may be changed due to your 
    # repo structure.

    import torch
    from adaptive_gradient_clipping.src.agc import AGC
    
    # model here is any model with torch.nn.Module format
    
    optimizer = torch.optim.Adam(model.parameter(), 0.0001)
    optimizer = AGC(optimizer, 0.04, 0.003)

    # use the agc optimizer in your normal training script as a normal optimizer in torch.optim

## Experiment result on FashionMinist
### Accuracy
![accuracy](https://github.com/JayChanHoi/adaptive_gradient_clipping/blob/main/doc/accuracy.png)
-------------------------------------------------------------------------------------------------
### loss
![loss](https://github.com/JayChanHoi/adaptive_gradient_clipping/blob/main/doc/loss.png)


## To-Do
- [x] agc wrapper implementation
- [x] experiment on fashionMinist
