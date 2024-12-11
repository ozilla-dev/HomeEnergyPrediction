import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, prediction_length, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.prediction_length = prediction_length

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)
        
        x, _ = self.lstm(x, (h0, c0))
        x = self.ln(x)

        x = self.fc(x)
        x = x[:, -self.prediction_length:, :]
        return x
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, prediction_length, dropout):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.prediction_length = prediction_length

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)
        
        x, _ = self.gru(x, h0)
        x = self.ln(x)
        x = self.dropout(x)

        x = self.fc(x)
        x = x[:, -self.prediction_length:, :]
        return x