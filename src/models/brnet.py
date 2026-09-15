"""Manuscript reference network; three-class head is an explicit adaptation."""
from collections import OrderedDict
import torch
from torch import nn

class BrNet(nn.Module):
    def __init__(self, classes=4):
        super().__init__()
        layers = []
        channels = 1
        for i, width in enumerate((32, 64, 64, 64), 1):
            layers.extend([(f'conv{i}', nn.Conv2d(channels, width, 3)),
                           (f'relu{i}', nn.ReLU()), (f'pool{i}', nn.MaxPool2d(2, 2))])
            channels = width
        self.features = nn.Sequential(OrderedDict(layers))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(64 * 14 * 14, 64)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(64, classes)
        self.softmax = nn.Softmax(dim=1)

    def logits(self, x):
        return self.classifier(self.relu(self.dense(self.flatten(self.features(x)))))

    def forward(self, x):
        return self.softmax(self.logits(x))
