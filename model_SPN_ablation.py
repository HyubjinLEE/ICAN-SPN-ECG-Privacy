import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class NoiseModel(nn.Module):
    def __init__(self, feature_dim=256):
        super(NoiseModel, self).__init__()
        
        self.fc1 = nn.Linear(feature_dim, feature_dim)
        self.fc2 = nn.Linear(feature_dim, feature_dim)
        self.fc3 = nn.Linear(feature_dim, feature_dim)
        
        self.relu = nn.ReLU()
        
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.fc1.weight)
        init.zeros_(self.fc1.bias)
        init.xavier_uniform_(self.fc2.weight)
        init.zeros_(self.fc2.bias)
        init.uniform_(self.fc3.weight, -0.5, 0.5)
        init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class SPN(nn.Module):
    def __init__(self, ican_model, feature_dim=256):
        super(SPN, self).__init__()
        
        # Pre-trained ICAN model (frozen)
        self.ican = ican_model
        for param in self.ican.parameters():
            param.requires_grad = False
        
        self.noise_model = NoiseModel(feature_dim)
        
    def forward(self, x):
        # Extract features using ICAN's ResNet
        features = self.ican.feature_extractor(x)
        
        # Get attention distributions from ICAN modules
        _ = self.ican.identification_module.attention(features)
        id_attention = self.ican.identification_module.attention.attention_distribution
        
        _ = self.ican.classification_module.attention(features)
        class_attention = self.ican.classification_module.attention.attention_distribution
        
        # Sum attention distributions column-wise
        id_attention_sum = torch.sum(id_attention, dim=1)
        class_attention_sum = torch.sum(class_attention, dim=1)
        
        # Create attention distribution for noise
        noise_attention = F.sigmoid(id_attention_sum - class_attention_sum) 
        features = features.squeeze(-1)
        
        # Generate noise
        noise = self.noise_model(features)

        # Only noise model
        weighted_noise = noise

        # Add noise to original features
        perturbed_features = features + weighted_noise
        perturbed_features = perturbed_features.unsqueeze(-1)
        
        # Pass through ICAN modules
        id_logits = self.ican.identification_module(perturbed_features)
        class_logits = self.ican.classification_module(perturbed_features)
        
        return id_logits, class_logits, features, weighted_noise, perturbed_features
