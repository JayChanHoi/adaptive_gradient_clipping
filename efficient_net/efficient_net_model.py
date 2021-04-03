from efficientnet_pytorch import EfficientNet

import torch.nn as nn
import torch

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