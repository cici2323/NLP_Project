from opacus.layers import DPLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F



class LSTM_model(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers:int=1):

        super(LSTM_model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self,x):

        x = self.embedding(x)
        x,_ = self.rnn(x)
        x = x[:, -1, :]
        output = self.out(x).squeeze()
        return output

class DPLSTM_model(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers:int=1):

        super(DPLSTM_model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = DPLSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self,x):

        x = self.embedding(x)
        x,_ = self.rnn(x)
        x = x[:, -1, :]
        output = self.out(x).squeeze()
        return output

    
class DNN_model(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 seq_len: int):

        super(DNN_model, self).__init__()
        
        # convolutional layer
        self.embedding_dim=embedding_dim
        self.seq_len=seq_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self,x):

        x = self.embedding(x)
        x = torch.sigmoid(x.sum(axis=1))
        x = torch.sigmoid(self.fc(x))
        output = self.out(x)
        return output

    
class CNN_model(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 filters: int,
                 output_dim: int):

        super(CNN_model, self).__init__()
        
        # convolutional layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, filters, 3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.out = nn.Linear(filters, output_dim)

    def forward(self,x):

        x = self.embedding(x)
        x = x.transpose(1,2)
        x = self.pool(F.relu(self.conv(x)))
        x = x.squeeze()
        
        output = self.out(x)
        return output
    
