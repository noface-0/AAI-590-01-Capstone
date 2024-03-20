import json
import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class FNN(nn.Module):
    """
    feedforward neural network (FNN) class.
    
    :param input_size (int): Size of the input layer.
    :param hidden_sizes (List[int]): Sizes of the hidden layers.
    :param output_size (int): Size of the output layer.
    :param dropout_rate (float): Dropout rate for regularization.
    :param batch_norm (bool): Whether to use batch normalization.
    :param activation_fn: Activation function to use.
    """
    def __init__(
            self, 
            input_size: int, 
            hidden_sizes: list[int], 
            output_size: int, 
            dropout_rate: float = 0.5,
            batch_norm: bool = False,
            activation_fn = nn.ReLU
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
    

def build_fnn(
        model_params_path: str = 'models/runs/fnn/fnn_params.json', 
        model_weights_path: str = 'models/runs/fnn/fnn.pth'
) -> FNN:
    """
    Load the FNN model with saved parameters and weights.

    :param model_params_path: Path to the JSON file 
        containing model parameters.
    :param model_weights_path: Path to the .pth file 
        containing model weights.

    :return model: The loaded FNN model ready for inference.
    """
    with open(model_params_path, 'r') as f:
        model_params = json.load(f)
    
    model = FNN(
        input_size=model_params['input_size'],
        hidden_sizes=model_params['hidden_sizes'],
        output_size=model_params['output_size'],
        dropout_rate=0.5,
        batch_norm=False
    )
    
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()
    
    return model


def prepare_data(
        data: pd.DataFrame, 
        scaler_path: str='models/runs/fnn/scaler.joblib'
):
    """
    Prepare data for inference.

    :param data: Input data in the form of a pandas DataFrame.
    :param scaler_path: Path to the saved StandardScaler.

    :return features_tensor: Features as a torch.Tensor 
        ready for inference.
    """
    data = data.reindex(sorted(data.columns), axis=1)
    features = data.drop(
        ['timestamp', 'tic', 'target_high'], axis=1, errors='ignore'
    ).values
    
    scaler = joblib.load(scaler_path)
    features_scaled = scaler.transform(features)
    
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    
    return features_tensor


def fnn_prediction(
        input_data: pd.DataFrame,
        model_params_path: str = 'models/runs/fnn/fnn_params.json', 
        model_weights_path: str = 'models/runs/fnn/fnn.pth'
):
    """
    Perform inference with the loaded model.

    :param model: The loaded FNN model.
    :param input_data: The input data for inference, as a torch.Tensor.

    :return output: The model's output as a torch.Tensor.
    """
    pred_data = prepare_data(input_data)
    model = build_fnn(model_params_path, model_weights_path)

    with torch.no_grad():
        output = model(pred_data)

    predictions = output.numpy().flatten().tolist()
    input_data['fnn_pred'] = predictions

    return input_data