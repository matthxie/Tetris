import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class DeepQNetwork(nn.Module):
    def __init__(self, output_dim, env):
        super(DeepQNetwork, self).__init__()
        self.env = env

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=1, padding=0),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 + 3 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

        self.fc0 = nn.Sequential(
            nn.Linear(384, 64),
            nn.ReLU(),
        )

        self.create_weights()

    def create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def convert_info(self, next_blocks):
        next_blocks = torch.tensor(next_blocks[:, :-3]).long() - 1
        one_hot = torch.nn.functional.one_hot(next_blocks, num_classes=7).flatten()
        return one_hot.view(next_blocks.shape[0], -1)

    def forward(self, x):
        x0 = x[:, :-6]
        x0 = x0.view(x.size(0), 20, 10).unsqueeze(1)
        x1 = x[:, -6:]
        x1 = self.convert_info(x1)

        x = self.cnn(x0)
        x = self.fc0(x)

        x = torch.cat((x, x1), 1)
        x = self.fc(x)

        return x
