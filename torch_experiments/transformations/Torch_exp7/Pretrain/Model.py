import torch.nn as nn

class Projection_Head(nn.Module):

    def __init__(self, inputs):
        super(Projection_Head, self).__init__()
        self.projection_dim = 128

        self.first_layer = nn.Linear(inputs, inputs, bias=True)
        self.first_bn = nn.BatchNorm1d(inputs)

        self.second_layer = nn.Linear(inputs, inputs, bias=True)
        self.second_bn = nn.BatchNorm1d(inputs)

        self.third_layer = nn.Linear(inputs, self.projection_dim, bias=False)
        self.third_bn = nn.BatchNorm1d(self.projection_dim, affine=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        #fist layer
        x = self.first_layer(x)
        x = self.first_bn(x)
        x = self.relu(x)

        #second layer
        x = self.second_layer(x)
        x = self.second_bn(x)
        x = self.relu(x)

        #third layer
        x = self.third_layer(x)
        x = self.third_bn(x)

        return x