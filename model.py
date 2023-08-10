import torch.nn as nn
import torch
from torch.autograd import Function

class Encoder(nn.Module):
    def __init__(self, channel_dim, GRAY_SCALE, quant=False, dim_quant=5000):
        super(Encoder, self).__init__()
        self.channel_dim = channel_dim
        self.input_dim = 3 if not GRAY_SCALE else 1
        self.quantilize=quant
        if quant:
            self.tobinary_fc = nn.Linear(2304, dim_quant)
            self.tobinary_act = nn.Tanh()
            self.binarymod = BinarizedModule(self.ps1)

        self.fun = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, kernel_size=(9, 9), stride=(2, 2), padding=(4, 4), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.channel_dim, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=True),
        )

    def forward(self, x):
        x = self.fun(x) 
        if self.quantilize:
            x = self.tobinary_fc(x)
            x = self.tobinary_act(x)
            x = self.binarymod(x)
        return x

class BinarizedModule(nn.Module):
    def __init__(self, ps1=True):
        super(BinarizedModule, self).__init__()
        self.BF = BinarizedF().apply
        self.ps1=ps1

    def forward(self,input):
        # print(input.shape)
        output =self.BF(input)
        return output

class BinarizedF(Function):
    @staticmethod
    def forward(self, input):
        a = torch.ones_like(input)
        b = -torch.ones_like(input)
        output = torch.where(input>=0,a,b)
        return output

    @staticmethod
    def backward(self, output_grad):
        return output_grad

class Decoder(nn.Module):
    def __init__(self, channel_dim, orig_size=(128,128), output_dim=3, quant=False, dim_quant=5000):
        super(Decoder, self).__init__()
        self.channel_dim = channel_dim
        self.orig_size = orig_size
        self.quant = quant
        if quant:
            self.dequant = nn.Linear(dim_quant, 2304)

        self.fun = nn.Sequential(
            nn.ConvTranspose2d(self.channel_dim, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, output_dim, kernel_size=(8, 8), stride=(2, 2), padding=(3, 3), bias=True),
            nn.Tanh(),  # become +-1
        )

    def forward(self, x):
        if self.quant:
            x = self.dequant(x)
        x = x.view(x.shape[0], self.channel_dim, self.orig_size[0]//4, self.orig_size[1]//4)
        return self.fun(x)
