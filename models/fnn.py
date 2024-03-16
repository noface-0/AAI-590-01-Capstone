import torch
import torch
import torch.nn as nn
import torch.optim as optim


class FNN(nn.Module):
    def __init__(
            self, 
            input_size, 
            hidden_sizes, 
            output_size, 
            dropout_rate=0.5,
            batch_norm=False,
            activation_fn=nn.ReLU
    ):
        super(FNN, self).__init__()
        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(
                    nn.Linear(hidden_sizes[i-1], hidden_sizes[i])
                )

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            
            layers.append(activation_fn())
            
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)