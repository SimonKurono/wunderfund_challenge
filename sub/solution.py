import os
import sys
import numpy as np
import torch
import torch.nn as nn
import json

# Add project root folder to path for importing utils
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "competition_package"))

from utils import DataPoint


class MultiFeatureLSTM(nn.Module):
    """
    LSTM model for predicting all features simultaneously.
    Input: [batch, lookback, num_features]
    Output: [batch, num_features]
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # Output layer: maps hidden state to all features
        self.linear = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        # x: [batch, lookback, input_size]
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        # Take the output from the last timestep
        # lstm_out: [batch, lookback, hidden_size]
        # We want: [batch, hidden_size]
        last_hidden = lstm_out[:, -1, :]
        # Map to output features
        output = self.linear(last_hidden)  # [batch, input_size]
        return output


class PredictionModel:
    """
    Model that uses a trained LSTM to predict the next value
    based on a sliding window of previous states.
    """
    
    def __init__(self):
        # Model hyperparameters (from training)
        self.lookback = 7
        self.num_features = 32
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.2
        
        # Initialize model
        self.model = MultiFeatureLSTM(
            input_size=self.num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # Load model weights
        # Try multiple possible paths for model file
        possible_paths = [
            os.path.join(CURRENT_DIR, "model", "lstm_model.pt"),
            os.path.join(CURRENT_DIR, "lstm_model.pt"),
            "model/lstm_model.pt",
            "lstm_model.pt"
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path:
            # Load state dict
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()  # Set to evaluation mode
        else:
            # If model file doesn't exist, use untrained model (for testing)
            print(f"Warning: Model file not found. Tried: {possible_paths}")
            print("Using untrained model. Make sure to include model/lstm_model.pt in your submission.")
        
        # Track current sequence and history
        self.current_seq_ix = None
        self.sequence_history = []
        
    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict the next state given the current data point.
        
        Args:
            data_point: DataPoint object with current state and metadata
            
        Returns:
            numpy array of predictions if need_prediction is True, None otherwise
        """
        # Check if we've moved to a new sequence
        if self.current_seq_ix != data_point.seq_ix:
            # Reset state for new sequence
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []
        
        # Add current state to history
        self.sequence_history.append(data_point.state.copy())
        
        # If prediction is not needed, return None
        if not data_point.need_prediction:
            return None
        
        # Need at least 'lookback' steps to make a prediction
        if len(self.sequence_history) < self.lookback:
            # If we don't have enough history, use the current state as prediction
            # This should only happen in early steps of a sequence
            return data_point.state.copy()
        
        # Get the last 'lookback' states for prediction
        window = self.sequence_history[-self.lookback:]
        
        # Convert to tensor format: [batch=1, lookback, features]
        window_array = np.array(window, dtype=np.float32)
        window_tensor = torch.tensor(window_array, dtype=torch.float32).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(window_tensor)
            # Convert back to numpy array
            prediction_np = prediction.squeeze(0).cpu().numpy()
        
        return prediction_np


if __name__ == "__main__":
    # Test the model with the provided scorer
    from utils import ScorerStepByStep
    
    # Check existence of test file
    test_file = os.path.join(CURRENT_DIR, "competition_package", "datasets", "train.parquet")
    
    if not os.path.exists(test_file):
        print(f"Test file not found at {test_file}")
        print("Please ensure the dataset is available.")
    else:
        # Create and test our model
        model = PredictionModel()
        
        # Load data into scorer
        scorer = ScorerStepByStep(test_file)
        
        print("Testing LSTM model...")
        print(f"Feature dimensionality: {scorer.dim}")
        print(f"Number of rows in dataset: {len(scorer.dataset)}")
        
        # Evaluate our solution
        results = scorer.score(model)
        
        print("\nResults:")
        print(f"Mean R² across all features: {results['mean_r2']:.6f}")
        print("\nR² for first 5 features:")
        for i in range(min(5, len(scorer.features))):
            feature = scorer.features[i]
            print(f"  {feature}: {results[feature]:.6f}")
        
        print(f"\nTotal features: {len(scorer.features)}")
        
        print("\n" + "=" * 60)
        print("Model is ready for submission!")
        print("=" * 60)

