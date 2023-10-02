import torch.nn as nn


class NN2HiddenLayer(nn.Module):

    def __init__(self, act_function=nn.ReLU(), input_size=37):
        super(NN2HiddenLayer, self).__init__()

        self.input_size = input_size
        self.output_size = 1

        self.act_function = act_function

        self.input = nn.Linear(self.input_size, 20)
        self.hl1 = act_function
        self.linear_hl2 = nn.Linear(20, 10)
        self.hl2 = act_function
        self.output1 = nn.Linear(10, self.output_size)
        self.output2 = nn.Sigmoid()

    def forward(self, x):

        x = self.input(x)
        x = self.hl1(x)
        x = self.linear_hl2(x)
        x = self.hl2(x)
        x = self.output1(x)
        return self.output2(x)


class Perceptron(nn.Module):

    def __init__(self):
        super(Perceptron, self).__init__()
        self.act_function = None

        self.input_size = 37
        self.output_size = 1
        self.input = nn.Linear(self.input_size, 1)

    def forward(self, x):

        return self.input(x)


class NN3HiddenLayer(nn.Module):

    def __init__(self, act_function=nn.Sigmoid(), input_size=37):
        super(NN3HiddenLayer, self).__init__()

        self.input_size = input_size
        self.output_size = 1

        self.act_function = act_function

        self.input = nn.Linear(self.input_size, 20)
        self.hl1 = act_function
        self.linear_hl2 = nn.Linear(20, 10)
        self.hl2 = act_function
        self.linear_hl3 = nn.Linear(10, 5)
        self.hl3 = act_function
        self.linear_hl4 = nn.Linear(5, self.output_size)
        self.output = nn.Sigmoid()

    def forward(self, x):

        x = self.input(x)
        x = self.hl1(x)
        x = self.linear_hl2(x)
        x = self.hl2(x)
        x = self.linear_hl3(x)
        x = self.hl3(x)
        x = self.linear_hl4(x)
        return self.output(x)
