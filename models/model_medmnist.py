import torch
import torch.nn as nn
import torch.optim as optim
import models.models_medmnist as models

class MedMNISTModelWrapper:
    def __init__(self, config):
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = models.build_medmnist_model(config)
        
        self.model.to(self.device)

    def criterion(self):
        return nn.CrossEntropyLoss()

    def optimizer(self, parameters):
        lr = self.config.get('lr', 1e-4) 
        weight_decay = self.config.get('weight_decay', 0.05) # Weight decay mais alto ajuda a regularizar
        
        return optim.AdamW(
            parameters, 
            lr=lr, 
            weight_decay=weight_decay
        )

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)