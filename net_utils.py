import torch
from torch import nn
from torchvision import models

from pytorch_classifier.dataset import CATS_AND_DOGS_CLASSES


class PartialClassesClassifier(nn.Module):
    def __init__(self, net, sub_classes_indices):
        super(PartialClassesClassifier, self).__init__()
        self.net = net
        self.index_to_class = {}
        self.used_indices = []
        for class_index, indices in sub_classes_indices.items():
            self.index_to_class.update({i:class_index for i in indices})
            self.used_indices.extend(indices)
        self.used_indices = torch.LongTensor(self.used_indices)

    def forward(self, x):
        x = self.net(x)
        new_x = torch.ones_like(x)
        new_x = new_x * (-float("inf"))
        new_x[:, self.used_indices] = x[:, self.used_indices]
        max_indices = list(torch.argmax(new_x, -1).cpu().numpy())
        return [self.index_to_class[max_index] for max_index in max_indices]


def get_partial_classifier(model_name):
    net_class = getattr(models, model_name)
    net = net_class(pretrained=True)
    return PartialClassesClassifier(net, CATS_AND_DOGS_CLASSES).cuda()


def create_partially_trainable_net(net, num_classes, layers_to_enable=0):
    for parameter in net.parameters():
        parameter.requires_grad = False
    if isinstance(net, models.SqueezeNet):
        cl = net.classifier
        cl_models = list(cl.children())
        ch_in = cl_models[1].in_channels
        cl_models[1] = nn.Conv2d(ch_in, num_classes, 1,1)
        # cl_models[3] = nn.AvgPool2d()
        net.classifier = nn.Sequential(*cl_models)
        net.classifier.requires_grad=True
        net.num_classes = num_classes

    elif isinstance(net, models.ResNet):
        raise NotImplementedError()
    else:
        raise NotImplementedError
    # if hasattr(net, 'fc'):
    #     net.fc.requires_grad = True
    # else:
    #     fc = torch.nn.Linear(1000,2)
    #     fc.requires_grad=True
    #     net = torch.nn.Sequential([net, fc])
    return net
