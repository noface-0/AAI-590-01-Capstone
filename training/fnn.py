import json
import optuna
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from joblib import dump, load
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from processing.transform import AlpacaProcessor
from models.fnn import FNN
from models.losses import (
    LogCoshLoss,
    XSigmoidLoss,
    XTanhLoss
)


def download_data(
    start_date,
    end_date,
    ticker_list,
    time_interval,
    if_vix=True,
    api_key=None,
    api_secret=None,
    api_url=None
) -> pd.DataFrame:
    dp = AlpacaProcessor(api_key, api_secret, api_url)

    data = dp.download_data(
        ticker_list, 
        start_date, 
        end_date, 
        time_interval
    )
    data = dp.clean_data(data)
    data = dp.add_technical_indicators(data, add_fnn=False)

    if if_vix:
        data = dp.add_vix(data)
    else:
        data = dp.add_turbulence(data)
    
    return data


def prepare_data(
    forecast_steps: int,
    data=pd.DataFrame(),
    save_scaler=False
):
    # shifting for future forecast
    data['target_high'] = data.groupby('tic')['high'] \
        .shift(-forecast_steps)
    data = data.dropna(subset=['target_high'])
    data = data.reindex(sorted(data.columns), axis=1)

    features = data.drop(
        ['timestamp', 'tic', 'target_high'], axis=1
    ).values
    target = data['target_high'].values

    X_train, X_val, y_train, y_val = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # scaling on training after split to prevent leakage
    if save_scaler:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        dump(scaler, 'models/runs/fnn/scaler.joblib')
    else:
        # Load existing scaler and apply it without fitting
        scaler = load('models/runs/fnn/scaler.joblib')

        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)


    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return X_train_tensor, train_dataset, val_dataset, train_loader, val_loader


def objective(
        trial, 
        data: pd.DataFrame, 
        forecast_steps: int=5, 
        num_epochs: int=100
):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    hidden_sizes = trial.suggest_categorical(
        'hidden_sizes', [(32, 16), (64, 32), (128, 64), (256, 128)]
    )

    X_train, train_dataset, val_dataset, \
    train_loader, val_loader = prepare_data(
        data=data, 
        forecast_steps=forecast_steps
    )

    model = FNN(
        input_size=X_train.shape[1], 
        hidden_sizes=hidden_sizes, 
        output_size=1
    )
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs.squeeze(), targets).item()
        val_loss /= len(val_loader)
    
    return val_loss


def train(
    data: pd.DataFrame, 
    start_date,
    end_date,
    ticker_list,
    time_interval,
    if_vix=True,
    api_key=None,
    api_secret=None,
    api_url=None,
    forecast_steps=5,
    n_trials=10, 
    num_epochs=10,
):
    if data.empty:
        data = download_data(
            start_date=start_date,
            end_date=end_date,
            ticker_list=ticker_list,
            time_interval=time_interval,
            if_vix=if_vix,
            api_key=api_key,
            api_secret=api_secret,
            api_url=api_url
        )

    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(
            trial, 
            data, 
            forecast_steps, 
            num_epochs
        ),
        n_trials=n_trials
    )
    print(f"Best trial: {study.best_trial.params}")
    
    best_params = study.best_trial.params
    lr = best_params['lr']
    hidden_sizes = best_params['hidden_sizes']
    
    X_train, _, _, train_loader, val_loader \
        = prepare_data(
            data=data, 
            forecast_steps=forecast_steps,
            save_scaler=True
        )
    
    model = FNN(
        input_size=X_train.shape[1],
        hidden_sizes=hidden_sizes,
        output_size=1
    )
    
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), 'models/runs/fnn/fnn.pth')

    model_params = {
        'input_size': X_train.shape[1],
        'hidden_sizes': hidden_sizes,
        'output_size': 1,
        'dropout_rate': 0.5,
        'batch_norm': False
    }
    with open('models/runs/fnn/fnn_params.json', 'w') as f:
        json.dump(model_params, f)
    
    return best_params


def test(
    data: pd.DataFrame, 
    start_date,
    end_date,
    ticker_list,
    time_interval,
    if_vix=True,
    api_key=None,
    api_secret=None,
    api_url=None,
    forecast_steps=5,
    model_path='models/runs/fnn/fnn.pth',
    model_params_path='models/runs/fnn/fnn_params.json'
):
    if data.empty:
        data = download_data(
            start_date=start_date,
            end_date=end_date,
            ticker_list=ticker_list,
            time_interval=time_interval,
            if_vix=if_vix,
            api_key=api_key,
            api_secret=api_secret,
            api_url=api_url
        )

    _, _, _, _, val_loader = prepare_data(
        data=data,
        forecast_steps=forecast_steps
    )

    with open(model_params_path, 'r') as f:
        model_params = json.load(f)
    
    model = FNN(**model_params)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    criterion = nn.HuberLoss()

    val_loss = 0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            val_loss += criterion(outputs.squeeze(), targets).item()
            all_targets.extend(targets.numpy())
            all_outputs.extend(outputs.squeeze().numpy())

    val_loss /= len(val_loader)
    mae = mean_absolute_error(all_targets, all_outputs)
    mse = mean_squared_error(all_targets, all_outputs)
    
    evaluation_results = {
        'Validation Loss': val_loss,
        'Mean Absolute Error': mae,
        'Mean Squared Error': mse
    }
    
    print(f"Evaluation Results: {evaluation_results}")
    
    with open('models/runs/fnn/evaluation.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)