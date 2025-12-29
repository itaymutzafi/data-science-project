"""LSTM Model implementation using PyTorch under the hood but exposing a scikit-learn like interface."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseModel

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class LSTMRegressor(BaseModel):
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
        self.training_losses: List[float] = []
        
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[LSTMModel] = None
        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self._train_tail: Optional[np.ndarray] = None
        self._train_tail_index: Optional[List[pd.Timestamp]] = None

        self._init_model()
        
    def _init_model(self) -> None:
        """Initialize model, loss, and optimizer."""
        self.model = LSTMModel(self.input_size, self.hidden_size, self.num_layers, 1, self.dropout).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.training_losses = []

    def reset_weights(self) -> None:
        """Reset model weights and optimizer state for a fresh fit."""
        self._init_model()
        self._train_tail = None
        self._train_tail_index = None

    def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for training."""
        xs, ys = [], []
        # Align X[i : i+seq] with y[i+seq-1].
        # Using history up to T to predict target at T.
        
        for i in range(len(data) - self.seq_length + 1):
            x = data[i : (i + self.seq_length)] # indices i to i+seq-1
            # Assuming 'target' is already shifted (row T contains Return T->T+1)
            # We want to predict Target[T] using X ending at T.
            # So y index is i + seq_length - 1
            y_val = target[i + self.seq_length - 1]
            xs.append(x)
            ys.append(y_val)
        return np.array(xs), np.array(ys)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMRegressor":
        """Train the model on the provided series."""
        if self.model is None:
            self._init_model()

        self.model.train()
        
        # Prepare Data
        # Ensure X and y are aligned and numpy arrays
        X_vals = X.values.astype(np.float32)
        y_vals = y.values.astype(np.float32)
        
        # Create sequences
        # NOTE: This reduces dataset size by seq_length
        if len(X_vals) <= self.seq_length:
            print("Warning: Not enough data for sequence length.")
            self._train_tail = X_vals
            self._train_tail_index = list(X.index)
            return self
            
        X_seq, y_seq = self._create_sequences(X_vals, y_vals)
        
        # To Tensor
        X_tensor = torch.from_numpy(X_seq).to(self.device)
        y_tensor = torch.from_numpy(y_seq).unsqueeze(1).to(self.device) # (N, 1)
        
        # DataLoader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False) # Shuffle=False for time-series integrity
        
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
            self.training_losses.append(loss.item())

        # Store tail of training data to help align future predictions
        self._train_tail = X_vals[-self.seq_length :]
        self._train_tail_index = list(X.index[-self.seq_length :])
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict returns aligned to ``X`` index, padding early rows with NaN when history is insufficient."""
        if self.model is None:
            self._init_model()

        if X.empty:
            return pd.Series(dtype=float)

        self.model.eval()
        
        X_vals = X.values.astype(np.float32)
        history_vals = self._train_tail if self._train_tail is not None else np.empty((0, self.input_size), dtype=np.float32)
        history_index = self._train_tail_index if self._train_tail_index is not None else []

        combined_values = np.vstack([history_vals, X_vals]) if len(history_vals) else X_vals
        combined_index: List = list(history_index) + list(X.index)

        if len(combined_values) <= self.seq_length:
            return pd.Series(np.nan, index=X.index)

        x_index_set = set(X.index)
        X_seq_list: List[np.ndarray] = []
        pred_indices: List = []

        for i in range(self.seq_length, len(combined_values)):
            seq = combined_values[i - self.seq_length : i]
            target_idx = combined_index[i]
            if target_idx in x_index_set:
                X_seq_list.append(seq)
                pred_indices.append(target_idx)
        
        if not X_seq_list:
            return pd.Series(np.nan, index=X.index)
            
        X_tensor = torch.tensor(np.array(X_seq_list), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy().flatten()
            
        pred_series = pd.Series(data=preds, index=pred_indices)
        return pred_series.reindex(X.index)
