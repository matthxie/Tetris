import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTForImageClassification

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

        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

        self.expert1 = FFN(10, 32, 32, 3)
        self.expert2 = FFN(10, 32, 32, 3)
        self.expert3 = FFN(10, 32, 32, 3)
        self.expert4 = FFN(30, 64, 64, 3)

        self.cnn_encoder = CNNEncoder(1, 64, 10)

    def forward(self, x):
        # inputs = self.processor(images=x, return_tensors="pt")
        # outputs = self.vit(**inputs)
        # logits = outputs.logits

        logits = self.cnn_encoder(x)

        expert_output1 = self.expert1(logits)
        expert_output2 = self.expert2(logits)
        expert_output3 = self.expert3(logits)
        expert_output4 = self.expert4(logits)

        

        

