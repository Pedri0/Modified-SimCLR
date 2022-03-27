import torch.nn as nn
from absl import flags
FLAGS = flags.FLAGS

class Projection_Head(nn.Module):

    def __init__(self, inputs, proj_head_selector):
        super(Projection_Head, self).__init__()

        self.first_layer = nn.Linear(inputs, inputs, bias=True)
        self.first_bn = nn.BatchNorm1d(inputs)
        self.relu = nn.ReLU()
        self.proj_head_select = proj_head_selector

        if proj_head_selector >= 2:
            self.second_layer = nn.Linear(inputs, inputs, bias=True)
            self.second_bn = nn.BatchNorm1d(inputs)

        if proj_head_selector >= 3:
            self.third_layer = nn.Linear(inputs, 128, bias=False)
            self.third_bn = nn.BatchNorm1d(128, affine=False)

        self.output = inputs if proj_head_selector < 3 else 128

    def forward(self, x):

        x = self.first_layer(x)
        x = self.first_bn(x)
        x = self.relu(x)
        
        if self.proj_head_select >= 2:
            x = self.second_layer(x)
            x = self.second_bn(x)
            x = self.relu(x)
        
        if self.proj_head_select >= 3:
            x = self.third_layer(x)
            x = self.third_bn(x)

        return x


class Supervised_Head(nn.Module):

    def __init__(self, inputs, num_classes):
        super(Supervised_Head, self).__init__()
        self.num_classes = num_classes
        self.supervised_head = nn.Linear(inputs, self.num_classes, bias=True)

    def forward(self, x):

        x = self.supervised_head(x)

        return x

        

    