
import torch.nn as nn


class BaseClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BaseClassifier, self).__init__()
        self.classifier = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x):
        return self.classifier(x)
