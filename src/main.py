import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from model import LSTM, GRU
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator

def create_sequences(data, dates, seq_length, prediction_length):
    X = []
    y = []
    sequence_dates = []
    for i in range(len(data) - seq_length - prediction_length + 1):
        x_sequence = data[i:i+seq_length]
        y_sequence = data[i+seq_length:i+seq_length+prediction_length, 0]
        date = dates[i+seq_length:i+seq_length+prediction_length]
        X.append(x_sequence)
        y.append(y_sequence)
        sequence_dates.append(date)
    return np.array(X), np.array(y), np.array(sequence_dates)

def split_data(X, y, sequence_dates):
    train_size = int(len(X) * 0.9)
    val_size = int(len(X) * 0.05)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    train_dates = sequence_dates[:train_size]
    val_dates = sequence_dates[train_size:train_size + val_size]
    test_dates = sequence_dates[train_size + val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test, train_dates, val_dates, test_dates

def logcosh_quantile_loss(y_true, y_pred, alpha=0.9):
    diff = y_pred - y_true
    logcosh_loss = torch.log(torch.cosh(diff)).mean()

    quantile_loss = torch.where(diff >= 0, alpha * diff, (alpha - 1) * diff).mean()

    combined_loss = logcosh_loss + quantile_loss
    return combined_loss

def train(model, optimizer, criterion, scheduler, num_epochs, train_loader, val_loader, device):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(device), y_train.to(device)
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_pred = model(X_val)
                loss = criterion(y_pred, y_val)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss)

        if (epoch+1) % 5 == 0:
            torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                f'test_checkpoints/lstm_{epoch+1}.pth')
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, Val Loss: {val_loss}, Learning Rate: {scheduler.get_last_lr()}')

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

def predict(model, test_loader, scaler, device, sample_size=68):
    model.eval()
    predictions = []
    real_values = []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)
            y_pred = y_pred.cpu().numpy()
            y_test = y_test.cpu().numpy()
            predictions.append(y_pred[0])
            real_values.append(y_test[0])
            
    predictions = np.concatenate(predictions, axis=0)
    real_values = np.concatenate(real_values, axis=0)
    
    predictions = predictions.reshape(-1, predictions.shape[-1])
    real_values = real_values.reshape(-1, real_values.shape[-1])

    predictions = scaler.inverse_transform(predictions)
    real_values = scaler.inverse_transform(real_values)

    predictions = predictions[:sample_size]
    real_values = real_values[:sample_size]
    return predictions, real_values

def plot_predictions(predictions, real_values):
    

    start_date = pd.to_datetime('2024-09-26 00:00')
    time_index = pd.date_range(start=start_date, periods=len(predictions), freq='3h')

    plt.figure(figsize=(14, 7))
    plt.plot(time_index, predictions[:, 0], label='Predictions', color='blue')
    plt.plot(time_index, real_values[:, 0], label='Real Values', color='orange')
    plt.xlabel('Time')
    plt.ylabel('kWh')
    plt.title('Energy Usage Prediction vs Real Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    tick_positions = pd.date_range(start=start_date, end=time_index[-1], freq='6h')

    plt.gca().xaxis.set_major_locator(FixedLocator(mdates.date2num(tick_positions)))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()

    plt.show()
    
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
    data = data.resample('3h', on='time').sum().reset_index() # resample the data to hourly
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
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    scaled_data = input_scaler.fit_transform(features.values)
    output_scaler.fit(features[['energy_usage']])
    seq_length = 8 * 120 # 120 days of data
    prediction_length = 8 # 1 day of data

    dates = data.index

    X, y, sequence_dates = create_sequences(scaled_data, dates, seq_length, prediction_length)
    y = y[:, :].reshape(-1, prediction_length, 1)

    X_train, y_train, X_val, y_val, X_test, y_test, train_dates, val_dates, test_dates = split_data(X, y, sequence_dates)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    test_data = TensorDataset(X_test, y_test)

    batch_size = 16
    num_workers = 4
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    input_size = X_train.shape[2]
    hidden_size = 512
    num_layers = 2
    output_size = y_train.shape[2]
    dropout = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRU(input_size, hidden_size, num_layers, output_size, prediction_length, dropout)
    model.to(device)
    
    learning_rate = 0.0001
    num_epochs = 200
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3)
    criterion = nn.HuberLoss(delta=0.1)
    if args.test:
        model.load_state_dict(torch.load('checkpoints/lstm_165.pth', weights_only=True)['model_state_dict']) 
        test(model, criterion, test_loader, device)
    elif args.predict:
        model.load_state_dict(torch.load('lstm_65.pth', weights_only=True)['model_state_dict'])
        predictions, real_values = predict(model, test_loader, output_scaler, device)
        plot_predictions(predictions, real_values)
    else:
        train(model, optimizer, criterion, scheduler, num_epochs, train_loader, val_loader, device)

if __name__ == '__main__':
    main()
