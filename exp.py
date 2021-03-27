from efficientnet_pytorch import EfficientNet
from nfn import nf_resnet50, nf_resnet18

import torchvision
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
from itertools import count

from tqdm import tqdm
import os
import shutil

import torch
import torch.nn as nn

from src.agc import AGC

model_dict = {
    # model_id : res, dropout p
    '0': (224, 0.2),
    '1': (240, 0.2),
    '2': (260, 0.3),
    '3': (300, 0.3),
    '4': (380, 0.4),
    '5': (456, 0.4),
    '6': (528, 0.5),
    '7': (600, 0.5),
    '8': (672, 0.5)
}

class ENClassifier(torch.nn.Module):
    def __init__(self, model_id, num_classes, advprop=True, dropout_p=0.2, temperature=1.0, from_pretrain=False):
        super(ENClassifier, self).__init__()
        if from_pretrain:
            self.model = EfficientNet.from_pretrained('efficientnet-b{}'.format(model_id), advprop=advprop)
        else:
            self.model = EfficientNet.from_name('efficientnet-b{}'.format(model_id))
        self.temperature = temperature

        del self.model._avg_pooling
        del self.model._dropout
        del self.model._fc

        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.class_layer = nn.Linear(self._find_channel(), num_classes)

    def _find_channel(self):
        test_input = torch.rand([1, 3, 512, 910])
        test_output = self.model.extract_features(test_input)

        return test_output.shape[1]

    def __call__(self, inputs):
        bs = inputs.size(0)
        x = self.model.extract_features(inputs)

        # Pooling and final linear layer
        x = self.global_average_pooling(x)
        x = x.reshape(bs, -1)
        x = self.dropout(x)
        x = self.class_layer(x) / self.temperature

        return x

def train(model_name='v0_3_agc'):
    if os.path.isdir('tensorboard/{}'.format(model_name)):
        shutil.rmtree('tensorboard/{}'.format(model_name))
        os.makedirs('tensorboard/{}'.format(model_name))
    else:
        os.makedirs('tensorboard/{}'.format(model_name))

    writer = SummaryWriter('tensorboard/{}'.format(model_name))
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(model_dict['0'][0]),
        torchvision.transforms.Lambda(lambd=lambda x: x.repeat(3, 1, 1)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(model_dict['0'][0]),
        torchvision.transforms.Lambda(lambd=lambda x: x.repeat(3, 1, 1)),
    ])

    train_dataset = torchvision.datasets.FashionMNIST(root='data', download=True, train=True, transform=train_transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='data', download=True, train=False, transform=test_transform)
    train_data_generator = DataLoader(train_dataset, batch_size=256)
    test_data_generator = DataLoader(test_dataset, batch_size=256)
    # model = ENClassifier(model_id=0, num_classes=10)
    model = nf_resnet18(num_classes=10, pretrained=True)
    if torch.cuda.is_available():
        model.cuda()
        model = torch.nn.DataParallel(model)

    print(model.state_dict().keys())
    optimizer = AGC(torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001), clip_lambda=0.08, layer_to_skip=['fc'])
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

        if iter + 1 == 50:
            break

if __name__ == '__main__':
    train()




