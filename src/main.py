import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from model import LSTM

def init_dataloaders(train_dataset, test_dataset):
    batch_size = 64
    n_workers = 4

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    return train_loader, test_loader

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequence = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(sequence)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def split_data(sequences, targets):
    train_size = int(len(sequences) * 0.8)
    X_train, X_test = sequences[:train_size], sequences[train_size:]
    y_train, y_test = targets[:train_size], targets[train_size:]
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def rmse_loss(y_pred, y_true):
    mse_loss = nn.MSELoss()
    epsilon = 1e-8
    return torch.sqrt(mse_loss(y_pred, y_true) + epsilon)

def train(model, optimizer, num_epochs, train_loader, device):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(device), y_train.to(device)
            y_pred = model(X_train)
            loss = rmse_loss(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)

        if (epoch+1) % 5 == 0:
            torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                f'checkpoints/lstm_{epoch+1}.pth')
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')

def test(model, test_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)
            loss = rmse_loss(y_pred, y_test)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    print(f'Test Loss: {avg_loss}')

def predict_next_day(model, scaler, data, seq_length, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        sequence = data.values[-seq_length:]
        sequence = np.expand_dims(sequence, axis=0) # add batch dimension
        sequence = torch.tensor(sequence, dtype=torch.float32).to(device)
        for _ in range(seq_length):
            y_pred = model(sequence).to(device)
            predictions.append(y_pred.cpu().numpy())
            y_pred = y_pred.view(1, 1, -1)
            y_pred = y_pred.cpu().numpy()
            sequence = sequence.cpu().numpy()
            sequence = np.append(sequence, y_pred, axis=1) # append the prediction to the sequence
            sequence = torch.tensor(sequence, dtype=torch.float32).to(device)
    predictions = np.array(predictions).squeeze()
    predictions = scaler.inverse_transform(predictions)
    print(predictions)
    last_date = data.index[-1]
    date_range = pd.date_range(start=last_date, periods=seq_length+1, freq='15min')[1:]
    prediction_df = pd.DataFrame(predictions, columns=data.columns, index=date_range)

    plt.figure(figsize=(16, 8))
    for column in prediction_df.columns:
        plt.plot(prediction_df.index, prediction_df[column], label=column)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Predicted Energy and Gas Usage for the Next Day')
    plt.legend()
    plt.show()

def main():
    df = pd.read_csv('data/energy_gas_usage.csv', parse_dates=['time'])
    df.set_index('time', inplace=True) # replace the index column with the time column
    
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(data, columns=df.columns, index=df.index)

    seq_length = 96 # full day
    data = scaled_df.values
    sequences, targets = create_sequences(data, seq_length)

    X_train, X_test, y_train, y_test = split_data(sequences, targets)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    train_loader, test_loader = init_dataloaders(train_data, test_data)

    input_size = X_train.shape[2]
    hidden_size = 64
    num_layers = 1
    num_classes = y_train.shape[1]
    learning_rate = 0.001
    num_epochs = 2

    model = LSTM(input_size, hidden_size, num_layers, num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(model, optimizer, num_epochs, train_loader, device)
    test(model, test_loader, device)

    predict_next_day(model, scaler, scaled_df, seq_length, device)

if __name__ == '__main__':
    main()
