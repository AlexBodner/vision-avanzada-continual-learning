from torch import nn
import torch
import torch.nn.functional as F

class CNN(nn.Module):
   def __init__(self, in_channels, embedding_dim):

       super(CNN, self).__init__()

       # 1st convolutional layer
       self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
       # Max pooling layer
       self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
       # 2nd convolutional layer
       self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
       # Fully connected layer
       self.fc1 = nn.Linear(16 * 8 * 8, embedding_dim)

   def forward(self, x):
       x = F.relu(self.conv1(x))
       x = self.pool(x)
       x = F.relu(self.conv2(x))
       x = self.pool(x)
       x = x.reshape(x.shape[0], -1)
       x = self.fc1(x)
       return x
   
class LinearProbe(nn.Module):
    def __init__(self, backbone, embedding_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.classifier(features)