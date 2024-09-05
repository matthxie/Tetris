import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def xavier_init(net: nn.Module):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)


class FFN(nn.Module):
    def __init__(self, dim0, dim1, dim2, dim3) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim0, dim1),
            nn.GELU(),
            nn.Linear(dim1, dim2),
            nn.GELU(),
            nn.Linear(dim2, dim3),
        )
        xavier_init(self)

    def forward(self, x):
        return self.ffn(x)


class CNNEncoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_dim) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1
        )

        self.fc1 = nn.Linear(256, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        xavier_init(self)

    def forward(self, x):
        x0 = x[:, :-6]
        x0 = x0.view(x.size(0), 20, 10).unsqueeze(1)
        x1 = x[:, -6:]

        x0 = F.relu(self.conv1(x0))
        x0 = F.max_pool2d(x0, 2)
        x0 = F.relu(self.conv2(x0))
        x0 = F.max_pool2d(x0, 2)
        x0 = F.relu(self.conv3(x0))
        x0 = F.max_pool2d(x0, 2)

        x0 = x0.view(x.size(0), -1)
        x0 = F.relu(self.fc1(x0))
        x0 = self.fc2(x0)

        return x0


class GABA(nn.Module):
    def __init__(self, env) -> None:
        super().__init__()

        self.expert1 = FFN(10, 32, 32, 3)
        self.expert2 = FFN(10, 32, 32, 3)
        self.expert3 = FFN(10, 32, 32, 3)

        self.router = FFN(10, 64, 64, 3)

        self.cnn_encoder = CNNEncoder(1, 64, 10)

        self.linear_map = FFN(3, 32, 32, 1)

    def forward(self, x):
        logits = self.cnn_encoder(x)

        router_out = self.router(logits)
        index = torch.argmax(router_out, dim=1)

        experts = [self.expert1, self.expert1, self.expert1]
        experts_out = []

        for batch in range(x.shape[0]):
            expert_output = experts[index[batch]](logits[batch])
            experts_out.append(expert_output)
        experts_out = torch.stack(experts_out)

        output = self.linear_map(experts_out)

        return output
