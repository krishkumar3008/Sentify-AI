import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, output_dim=1, num_layers=1, bidirectional=True, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        # text: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text)) # [batch_size, seq_len, embedding_dim]
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        if self.lstm.bidirectional:
            # hidden has shape [num_layers * 2, batch_size, hidden_dim]
            # final forward hidden state: hidden[-2, :, :]
            # final backward hidden state: hidden[-1, :, :]
            hidden_last = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden_last = hidden[-1, :, :]
            
        out = self.fc(self.dropout(hidden_last))
        return self.sigmoid(out)
