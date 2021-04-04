from nfn.nfn import nf_0, nf_1, NF_RESO_CONFIG
from models.nfnet import nfnet_f0, dm_nfnet_f0, nfnet_f1
from efficient_net.efficient_net_model import ENClassifier, model_dict

import torchvision
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
from itertools import count

from tqdm import tqdm
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agc import AGC

def train(model_name='v1_efficient_net_b0_agc'):
    if os.path.isdir('tensorboard/{}'.format(model_name)):
        shutil.rmtree('tensorboard/{}'.format(model_name))
        os.makedirs('tensorboard/{}'.format(model_name))
    else:
        os.makedirs('tensorboard/{}'.format(model_name))

    writer = SummaryWriter('tensorboard/{}'.format(model_name))
    train_transform = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(NF_RESO_CONFIG['nf_1']['train_reso']),
        # torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(model_dict['0']),
        torchvision.transforms.Lambda(lambd=lambda x: x.repeat(3, 1, 1)),

        # torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.RandomVerticalFlip(),
        # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    test_transform = torchvision.transforms.Compose([
        # torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(model_dict['0']),
        torchvision.transforms.Lambda(lambd=lambda x: x.repeat(3, 1, 1)),

        # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_dataset = torchvision.datasets.FashionMNIST(root='data', download=True, train=True, transform=train_transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='data', download=True, train=False, transform=test_transform)
    train_data_generator = DataLoader(train_dataset, batch_size=256)
    test_data_generator = DataLoader(test_dataset, batch_size=256)
    # model = nfnet_f1(num_classes=10)
    model = ENClassifier(model_id=0, num_classes=10)
    if torch.cuda.is_available():
        model.cuda()
        model = torch.nn.DataParallel(model)

    print(model.state_dict().keys())
    optimizer = AGC(torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001), clip_lambda=0.04, layer_to_skip=['fc'])
    loss_func = torch.nn.CrossEntropyLoss()
    model.train()

    for iter in count():
        epoch_loss = []
        train_num_correct_pred = 0
        train_num_pred = 0
        for batch_image, batch_label in tqdm(train_data_generator):
            if torch.cuda.is_available():
                batch_image = batch_image.to('cuda')
                batch_label = batch_label.to('cuda')

            output = model(batch_image)
            loss = loss_func(output, batch_label)

            pred = output.argmax(dim=1)
            correct_pred = (pred == batch_label).sum(dim=0)

            train_num_correct_pred += correct_pred
            train_num_pred += pred.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        print('---------------------------------train iter :{}------------------------------'.format(iter + 1))
        print('loss : {}'.format(sum(epoch_loss)/len(epoch_loss)))
        writer.add_scalars('loss', {'train': sum(epoch_loss)/len(epoch_loss)}, global_step=iter + 1)
        writer.add_scalars('accuracy', {'train': train_num_correct_pred/train_num_pred}, global_step=iter + 1)

        with torch.no_grad():
            eval_loss = []
            num_correct_pred = 0
            num_pred = 0
            model.eval()
            for batch_image, batch_label in tqdm(test_data_generator):
                if torch.cuda.is_available():
                    batch_image = batch_image.to('cuda')
                    batch_label = batch_label.to('cuda')

                output = model(batch_image)
                loss = loss_func(output, batch_label)

                pred = output.argmax(dim=1)
                correct_pred = (pred == batch_label).sum(dim=0)

                eval_loss.append(loss.item())
                num_correct_pred += correct_pred
                num_pred += pred.shape[0]

            print('eval_loss : {}'.format(sum(eval_loss) / len(eval_loss)))
            print('accuracy : {}'.format(num_correct_pred / num_pred))

            writer.add_scalars('loss', {'test': sum(eval_loss) / len(eval_loss)}, global_step=iter + 1)
            writer.add_scalars('accuracy', {'test': num_correct_pred / num_pred}, global_step=iter + 1)

        model.train()
        if iter + 1 == 50:
            break

if __name__ == '__main__':
    train()




