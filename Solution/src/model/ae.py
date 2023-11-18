import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class CAutoEncoder(nn.Module):
    def __init__(self, dim_input, z_dim=32, seed=2022):
        super(CAutoEncoder, self).__init__()
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.z_dim = z_dim
        self.dim_input = dim_input
        self.dim_input_list = [dim_input, max(dim_input // 2, 4), max(dim_input // 3, 4), z_dim]
        self.encoder = nn.Sequential(

            nn.Linear(self.dim_input_list[0] + 1, self.dim_input_list[1]),
            nn.ReLU(True),

            nn.Linear(self.dim_input_list[1], self.dim_input_list[2]),
            nn.ReLU(True),

        )
        self.fc_z = nn.Linear(self.dim_input_list[2], z_dim)
        self.decoder = nn.Sequential(

            nn.Linear(z_dim + 1, self.dim_input_list[2]),
            nn.ReLU(True),

            nn.Linear(self.dim_input_list[2], self.dim_input_list[1]),
            nn.ReLU(True),
            nn.Linear(self.dim_input_list[1], self.dim_input_list[0]),

        )
        self.weight_init()
        self.to(device)

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x, c):
        c = c.view(-1, 1)
        c_ = c.repeat(x.shape[0], 1)
        x_cat = torch.cat([x, c_], dim=1)
        z = self._encode(x_cat)
        z_ = self.fc_z(z)
        z_ = F.normalize(z_, p=2, dim=1)
        z_cat = torch.cat([z_, c_], dim=1)
        x_recon = self._decode(z_cat)
        return x_recon, z_

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def loss(self, x, x_rec):
        mse_loss = nn.MSELoss()
        MSE = mse_loss(x, x_rec)
        return MSE


class AutoEncoder(nn.Module):
    def __init__(self, dim_input, z_dim=32, seed=2022):
        super(AutoEncoder, self).__init__()
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.z_dim = z_dim
        self.dim_input = dim_input
        self.dim_input_list = [dim_input, max(dim_input//2, 4), max(dim_input//4, 4), z_dim]
        self.encoder = nn.Sequential(
            nn.Linear(self.dim_input_list[0], self.dim_input_list[1]),
            nn.ReLU(True),
            nn.Linear(self.dim_input_list[1], self.dim_input_list[2]),
            nn.ReLU(True),

        )
        self.fc_z = nn.Linear(self.dim_input_list[2], z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, self.dim_input_list[2]),
            nn.ReLU(True),
            nn.Linear(self.dim_input_list[2], self.dim_input_list[1]),
            nn.ReLU(True),
            nn.Linear(self.dim_input_list[1], self.dim_input_list[0]),
        )
        self.weight_init()
        self.to(device)

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x):
        z = self._encode(x)
        z_ = self.fc_z(z)
        z_ = F.normalize(z_, p=2, dim=1)
        x_recon = self._decode(z_)
        return x_recon, z_

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def loss(self, x, x_rec):
        mse_loss = nn.MSELoss()
        MSE = mse_loss(x, x_rec)
        return MSE