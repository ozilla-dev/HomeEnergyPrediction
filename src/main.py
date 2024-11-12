import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from model import LSTM
import argparse
from torch.optim.lr_scheduler import StepLR

def create_sequences(data, dates, seq_length, prediction_length):
    X = []
    y = []
    sequence_dates = []
    for i in range(len(data) - seq_length - prediction_length + 1):
        x_sequence = data[i:i+seq_length]
        y_sequence = data[i+seq_length:i+seq_length+prediction_length]
        date = dates[i+seq_length:i+seq_length+prediction_length]
        X.append(x_sequence)
        y.append(y_sequence)
        sequence_dates.append(date)
    return np.array(X), np.array(y), np.array(sequence_dates)

def split_data(X, y, sequence_dates):
    train_size = int(len(X) * 0.9)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    train_dates, test_dates = sequence_dates[:train_size], sequence_dates[train_size:]
    return X_train, y_train, X_test, y_test, train_dates, test_dates

def train(model, optimizer, criterion, scheduler, num_epochs, train_loader, device):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(device), y_train.to(device)
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        average_loss = epoch_loss / len(train_loader)
        scheduler.step()

        if (epoch+1) % 5 == 0:
            torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                f'checkpoints/lstm_{epoch+1}.pth')
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}, Learning Rate: {scheduler.get_last_lr()}')

def test(model, criterion, test_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)
            loss = criterion(y_pred, y_test)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss}')

def predict(model, test_loader, scaler, device):
    model.eval()
    predictions = []
    real_values = []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)
            y_pred = y_pred.cpu().numpy()
            y_test = y_test.cpu().numpy()
            predictions.append(y_pred)
            real_values.append(y_test)
            
    predictions = np.concatenate(predictions, axis=0)
    real_values = np.concatenate(real_values, axis=0)
    
    predictions = predictions.reshape(-1, predictions.shape[-1])
    real_values = real_values.reshape(-1, real_values.shape[-1])
    predictions = scaler.inverse_transform(predictions)
    real_values = scaler.inverse_transform(real_values)
    return predictions, real_values
    
def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--predict', action='store_true')
    args = parser.parse_args()

    data = pd.read_csv('data/total_energy_usage.csv', parse_dates=['time'])
    data = data.resample('h', on='time').sum().reset_index() # resample the data to hourly
    data.set_index('time', inplace=True) # replace the index column with the time column

    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month

    hour_encoding = pd.get_dummies(data['hour'], prefix='hour', drop_first=True)
    day_of_week_encoding = pd.get_dummies(data['day_of_week'], prefix='day_of_week', drop_first=True)
    month_encoding = pd.get_dummies(data['month'], prefix='month', drop_first=True)
    
    features = pd.concat([
        data['energy_usage'],
        hour_encoding,
        day_of_week_encoding,
        month_encoding
    ], axis=1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features.values)
    seq_length = 24 * 7 # 7 days of data
    prediction_length = 24 # 1 day of data

    dates = data.index

    X, y, sequence_dates = create_sequences(scaled_data, dates, seq_length, prediction_length)

    X_train, y_train, X_test, y_test, train_dates, test_dates  = split_data(X, y, sequence_dates)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    batch_size = 32
    num_workers = 4
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    input_size = X_train.shape[2]
    hidden_size = 128
    num_layers = 4
    output_size = y_train.shape[2]
    dropout = 0.2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(input_size, hidden_size, num_layers, output_size, prediction_length, dropout)
    model.to(device)
    
    learning_rate = 0.001
    num_epochs = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.90)
    criterion = lambda y_pred, y_true: torch.sqrt(nn.MSELoss()(y_pred, y_true))
    if args.test:
        model.load_state_dict(torch.load('checkpoints/lstm_165.pth', weights_only=True)['model_state_dict']) 
        test(model, criterion, test_loader, device)
    elif args.predict:
        model.load_state_dict(torch.load('checkpoints/lstm_15.pth', weights_only=True)['model_state_dict'])
        predictions, real_values = predict(model, test_loader, scaler, device)
    else:
        train(model, optimizer, criterion, scheduler, num_epochs, train_loader, device)

if __name__ == '__main__':
    main()
