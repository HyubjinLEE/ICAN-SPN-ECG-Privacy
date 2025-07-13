import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from model_ResNet import ECGResNet

class SelfAttention(nn.Module):
    def __init__(self, d_k):
        super(SelfAttention, self).__init__()
        self.d_k = d_k
        self.query = nn.Linear(1, d_k)
        self.key = nn.Linear(1, d_k)

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.query.weight)
        init.zeros_(self.query.bias)
        init.xavier_uniform_(self.key.weight)
        init.zeros_(self.key.bias)

    def forward(self, x):  
        Q = self.query(x)  # [batch_size, feature_dim, d_k]
        K = self.key(x)    # [batch_size, feature_dim, d_k]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # Apply softmax to get attention distribution
        attention_distribution = F.softmax(scores, dim=-1)
        
        # Save attention distribution
        self.attention_distribution = attention_distribution
        
        # Apply attention weights to values
        output = torch.matmul(attention_distribution, x)

        output = output.squeeze(-1)
        
        return output

class TaskModule(nn.Module):
    def __init__(self, feature_dim, d_k, output_dim):
        super(TaskModule, self).__init__()
        
        self.attention = SelfAttention(d_k)
        
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        init.zeros_(self.fc1.bias)
        init.xavier_uniform_(self.fc2.weight)
        init.zeros_(self.fc2.bias)
        init.xavier_uniform_(self.fc3.weight)
        init.zeros_(self.fc3.bias)
        
    def forward(self, x):
        # Apply self-attention
        x = self.attention(x)
        
        # Save attention distribution
        self.attention_distribution = self.attention.attention_distribution

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class ICAN(nn.Module):
    def __init__(self, feature_dim=256, d_k=64, num_patients=48, num_classes=5):
        super(ICAN, self).__init__()
        
        # Feature extractor (ResNet)
        self.feature_extractor = ECGResNet(feature_dim)
        
        # Identification module
        self.identification_module = TaskModule(feature_dim, d_k, num_patients)
        
        # Classification module
        self.classification_module = TaskModule(feature_dim, d_k, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)

        id_logits = self.identification_module(features)
        class_logits = self.classification_module(features)

        return id_logits, class_logits, features
