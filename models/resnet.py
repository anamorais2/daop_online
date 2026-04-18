import torch
import torch.nn as nn

import torchvision

class TrainResNet18():
    def __init__(self, num_classes=7):
        self.model = torchvision.models.resnet18(weights=None)
        self.num_classes = num_classes
        self.model.fc = nn.Linear(512, self.num_classes)

        self.criterion = nn.CrossEntropyLoss
        self.optimizer_class = torch.optim.Adam 
        self.lr = 0.001
        self.current_optimizer = None
        self.mode = "sl" 
        
    def optimizer(self, parameters):
        return self.optimizer_class(parameters, lr=self.lr)
    
    def load_weights_from_path(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def __str__(self):
        return "ResNet18"
    
    
class TrainResNet50():
    def __init__(self, num_classes=2):
        self.model = torchvision.models.resnet50(weights=None)
        self.num_classes = num_classes

        self.model.fc = nn.Linear(2048, self.num_classes)
        
        self.criterion = nn.CrossEntropyLoss 
        self.optimizer_class = torch.optim.Adam 
        self.lr = 0.001
        self.current_optimizer = None
        self.mode = "sl"
        
    def optimizer(self, parameters):
        return self.optimizer_class(parameters, lr=self.lr)
    
    def load_weights_from_path(self, path):
        self.model.load_state_dict(torch.load(path))
    
    def __str__(self):
        return "ResNet50"
