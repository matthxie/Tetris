import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.spatial_attention import SpatialAttention, ResidualBlock


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class DeepQNetwork(nn.Module):
    def __init__(self, output_dim, env):
        super(DeepQNetwork, self).__init__()
        self.env = env

        self.conv1 = nn.Conv2d(1 + 3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.attention = SpatialAttention(128)
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(4)])

        features_size = 128 * 10 * 5

        self.value_stream = nn.Sequential(
            nn.Linear(features_size, 512), nn.ReLU(), nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(features_size, 512), nn.ReLU(), nn.Linear(512, output_dim)
        )

        self.create_weights()

    def create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def convert_to_one_hot(self, next_blocks):
        next_blocks = torch.tensor(next_blocks[:, :-3]).long() - 1
        one_hot = F.one_hot(next_blocks, num_classes=7).flatten()
        return one_hot.view(next_blocks.shape[0], -1)

    def convert_to_channel(self, board, next_blocks):
        next_piece_channels = next_blocks.view(next_blocks.shape[0], -1, 1, 1).expand(
            next_blocks.shape[0], -1, 20, 10
        )
        return torch.cat((board, next_piece_channels), dim=1)

    def forward(self, x):
        x0 = x[:, :-6]
        x0 = x0.view(x.size(0), 20, 10).unsqueeze(1)
        x1 = x[:, -6:-3].unsqueeze(1)
        # x1 = self.convert_to_one_hot(x1)

        state = self.convert_to_channel(x0, x1)

        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.attention(x)
        x = self.res_blocks(x)

        x = x.view(x.size(0), -1)

        # Dueling network architecture
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q
