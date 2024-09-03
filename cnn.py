import torch
import torch.nn as nn
import torch.nn.functional as F

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
            F.gelu(),
            nn.Linear(dim1, dim2),
            F.gelu(),
            nn.Linear(dim2, dim3)
        )
        xavier_init(self)
    
    def forward(self, x):
        return self.ffn(x)
    
class CNNEncoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_dim) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(128 * 8 * 8, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        xavier_init(self)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GABA(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.expert1 = FFN(10, 32, 32, 3)
        self.expert2 = FFN(10, 32, 32, 3)
        self.expert3 = FFN(10, 32, 32, 3)

        self.router = FFN(30, 64, 64, 3)

        self.cnn_encoder = CNNEncoder(1, 64, 10)

    def forward(self, x):
        logits = self.cnn_encoder(x)

        router_out = self.router(logits)

        expert_out1 = self.expert1(logits)
        expert_out2 = self.expert2(logits)
        expert_out3 = self.expert3(logits)
