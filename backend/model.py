import torch
import torch.nn as nn
import numpy as np

class CNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.fc = nn.Linear(32 * ((input_size - 4) // 4), 64)
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        return self.out(features), features


def osr_decision(features, class_centers, threshold=1.2):
    distances = [torch.norm(features - c) for c in class_centers]
    min_dist = min(distances)

    if min_dist > threshold:
        return "Unknown DDoS", min_dist.item()
    else:
        return "Known Traffic", min_dist.item()
