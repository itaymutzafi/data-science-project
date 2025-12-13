"""LSTM Model implementation using PyTorch under the hood but exposing a scikit-learn like interface."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class LSTMRegressor:
    """
    Scikit-learn compatible wrapper for PyTorch LSTM.
    Handles data preparation (sequences), training loop, and prediction.
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 50, 
        num_layers: int = 1, 
        dropout: float = 0.0,
        seq_length: int = 10,
        learning_rate: float = 0.001,
        num_epochs: int = 50,
        batch_size: int = 32,
        device: str = "auto"
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        self.model = LSTMModel(input_size, hidden_size, num_layers, 1, dropout).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for i in range(len(data) - self.seq_length):
            x = data[i:(i + self.seq_length)]
            y = target[i + self.seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Trains the model."""
        self.model.train()
        
        # Prepare Data
        # Ensure X and y are aligned and numpy arrays
        X_vals = X.values.astype(np.float32)
        y_vals = y.values.astype(np.float32)
        
        # Create sequences
        # NOTE: This reduces dataset size by seq_length
        if len(X_vals) <= self.seq_length:
            print("Warning: Not enough data for sequence length.")
            return
            
        X_seq, y_seq = self._create_sequences(X_vals, y_vals)
        
        # To Tensor
        X_tensor = torch.from_numpy(X_seq).to(self.device)
        y_tensor = torch.from_numpy(y_seq).unsqueeze(1).to(self.device) # (N, 1)
        
        # DataLoader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False) # Shuffle False for TS? Usually shuffle allowed for training sets if i.i.d assumption holds locally, but strict TS might prefer False. Let's keep False for safety in expanding window.
        
        # Training Loop
        for epoch in range(self.num_epochs):
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            # if (epoch+1) % 10 == 0:
            #     print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.6f}')

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predicts returns."""
        self.model.eval()
        
        X_vals = X.values.astype(np.float32)
        
        # We need sequences to predict. 
        # If we are effectively "forecasting", for index i, we need X[i-seq_len : i].
        # The standard interface predict(X) usually implies mapping row i -> prediction i.
        # But for LSTM, row i prediction depends on past window.
        # We will assume X represents the AVAILABLE history at prediction time.
        # However, to match the "aligned" input of sklearn (where row i is the features for target i),
        # we have a discrepancy with sequence models.
        
        # SOLUTION: We will align so that the output index matches the input index WHERE PREDICTION IS POSSIBLE.
        # Indices 0 to seq_len-1 will be NaN or dropped.
        
        if len(X_vals) <= self.seq_length:
            return pd.Series(np.nan, index=X.index)

        # Create sequences from input X. 
        # IMPORTANT: When we did fit, y[i+seq_len] corresponded to X[i:i+seq]
        # So if we pass X_test, we effectively predict targets starting from X_test.index[seq_len]
        
        # But wait, usually X in sklearn containsfeatures at time T to predict T (or T+1 if lagged).
        # We've already lagged features in `features/sentiment_analysis.py`.
        # So X[t] contains Info[t-1].
        # An LSTM usually wants X[t-seq]...X[t] to predict y[t].
        
        # Let's simple create sequences on X.
        # The predictions will correspond to the end of each sequence.
        
        sequences = []
        # We can predict for indices from seq_length to len(X)
        for i in range(len(X_vals) - self.seq_length + 1):
             # If we want to predict for row K, we need sequence ending at K.
             # So X[K-seq_len+1 : K+1] ?
             # Let's match training:
             # Training: input X[i : i+seq], target y[i+seq]
             # So if we feed X[i : i+seq], we get prediction for time of row i+seq.
             
             # Wait, range(len - seq) in training puts target at i+seq.
             # Here we want to generate predictions for the provided X rows.
             # Only rows [seq_length:] can be predicted.
             pass
             
        # Let's construct strictly:
        # Preds will be shorter by seq_length
        # Returns indexes match X.index[seq_length:]
        
        X_seq_list = []
        valid_indices = []
        
        # To get prediction for time T (which corresponds to row T in X),
        # we need input sequence X[T-seq_length : T]. 
        # Note: X is exclusive of target, so X[T] is valid input for y[T].
        # But LSTM needs history. 
        
        for i in range(self.seq_length, len(X_vals)):
            seq = X_vals[i-self.seq_length : i] 
            # Note: This gives sequence of length `seq_length`
            # Ending right before i? No, 0 to seq_length -> length seq_length.
            # Let's verify:
            # i=10, seq_len=10. i-10=0. X[0:10] is rows 0..9.
            # Row 10 needs history.
            X_seq_list.append(seq)
            valid_indices.append(X.index[i]) # Prediction for this index
            
        if not X_seq_list:
            return pd.Series(dtype=float)
            
        X_tensor = torch.tensor(np.array(X_seq_list), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy().flatten()
            
        # Create Series with correct index
        # We pad the beginning with NaNs to match original length?
        # Or just return aligned series? Aligned is better for evaluation logic.
        
        return pd.Series(data=preds, index=valid_indices)
