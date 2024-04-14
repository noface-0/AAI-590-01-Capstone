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
    feature_scaler_path: str = 'models/runs/fnn/feature_scaler.joblib',
    target_scaler_path: str = 'models/runs/fnn/target_scaler.joblib'
):
    """
    Prepare data for inference.

    :param data: Input data as pandas DataFrame.
    :param feature_scaler_path: Path to feature scaler.
    :param target_scaler_path: Path to target scaler.
    :return features_tensor: Features as torch.Tensor.
    :return target_scaler: Loaded target scaler.
    """
    data = data.reindex(sorted(data.columns), axis=1)
    features = data.drop(
        ['timestamp', 'tic', 'target_high'], axis=1, errors='ignore'
    ).values

    feature_scaler = joblib.load(feature_scaler_path)
    features_scaled = feature_scaler.transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    target_scaler = joblib.load(target_scaler_path)

    return features_tensor, target_scaler


def fnn_prediction(
    input_data: pd.DataFrame,
    model_params_path: str = 'models/runs/fnn/fnn_params.json',
    model_weights_path: str = 'models/runs/fnn/fnn.pth'
):
    """
    Perform inference with the loaded model.

    :param input_data: Input data for inference as pandas DataFrame.
    :param model_params_path: Path to JSON file with model parameters.
    :param model_weights_path: Path to saved model weights.
    :return output_data: Input data with predictions in new column.
    """
    pred_data, target_scaler = prepare_data(input_data)

    model = build_fnn(model_params_path, model_weights_path)

    with torch.no_grad():
        output = model(pred_data)

    predictions_scaled = output.numpy().flatten()
    predictions = target_scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).flatten()

    output_data = input_data.copy()
    output_data['fnn_pred'] = predictions

    return output_data