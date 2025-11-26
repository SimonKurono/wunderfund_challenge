import os
import sys
import json
from collections import deque

import numpy as np
import torch
import torch.nn as nn

# Add project root folder to path for importing utils
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "competition_package"))

from utils import DataPoint


class MultiFeatureLSTM(nn.Module):
    """
    LSTM model for predicting raw features from engineered inputs.
    Input: [batch, lookback, input_features]
    Output: [batch, output_features]
    """

    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.linear(last_hidden)


class RollingFeatureBuffer:
    """
    Maintains per-sequence history and builds rolling feature vectors on the fly.
    """

    SUPPORTED_STATS = ("mean", "std")

    def __init__(self, base_dim, lookback, windows, stats):
        self.base_dim = base_dim
        self.lookback = lookback
        self.windows = sorted(windows) if windows else []
        self.stats = [stat for stat in stats if stat in self.SUPPORTED_STATS]
        self.max_window = max(self.windows, default=1)
        max_history = max(self.lookback, self.max_window)
        self.raw_history: deque[np.ndarray] = deque(maxlen=max_history)
        self.feature_history: deque[np.ndarray] = deque(maxlen=self.lookback)
        self.feature_size = self._compute_feature_size()

    def _compute_feature_size(self) -> int:
        extra_multiplier = len(self.windows) * len(self.stats)
        return self.base_dim * (1 + extra_multiplier)

    def reset(self) -> None:
        self.raw_history.clear()
        self.feature_history.clear()

    def append(self, state: np.ndarray) -> np.ndarray:
        state_vec = np.asarray(state, dtype=np.float32)
        self.raw_history.append(state_vec)
        history_list = list(self.raw_history)
        parts = [state_vec]

        for window in self.windows:
            recent = history_list[-window:]
            stacked = np.stack(recent, axis=0)
            if "mean" in self.stats:
                parts.append(stacked.mean(axis=0))
            if "std" in self.stats:
                parts.append(stacked.std(axis=0, ddof=0))

        feature_vec = np.concatenate(parts).astype(np.float32)
        self.feature_history.append(feature_vec)
        return feature_vec

    def has_enough_history(self) -> bool:
        return len(self.feature_history) >= self.lookback

    def get_window(self) -> np.ndarray:
        window = list(self.feature_history)[-self.lookback :]
        return np.stack(window, axis=0).astype(np.float32)


class PredictionModel:
    """
    Model that uses a trained LSTM to predict the next value
    based on a sliding window of previous states.
    """
    
    def __init__(self):
        self.config = self._load_config()
        self.lookback = self.config.get("lookback", 7)
        self.hidden_size = self.config.get("hidden_size", 128)
        self.num_layers = self.config.get("num_layers", 2)
        self.dropout = self.config.get("dropout", 0.2)
        self.rolling_windows = self.config.get("rolling_windows", [5, 15, 30])
        self.feature_stats = self.config.get("feature_stats", ["mean", "std"])
        self.base_num_features = self.config.get("base_num_features", 32)
        self.feature_buffer = RollingFeatureBuffer(
            base_dim=self.base_num_features,
            lookback=self.lookback,
            windows=self.rolling_windows,
            stats=self.feature_stats,
        )
        self.input_feature_size = self.config.get("input_feature_size", self.feature_buffer.feature_size)
        self.output_feature_size = self.config.get("output_feature_size", self.base_num_features)

        self.model = MultiFeatureLSTM(
            input_size=self.input_feature_size,
            output_size=self.output_feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self._load_weights()

        self.current_seq_ix = None
        
    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        """
        Predict the next state given the current data point.
        
        Args:
            data_point: DataPoint object with current state and metadata
            
        Returns:
            numpy array of predictions if need_prediction is True, None otherwise
        """
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.feature_buffer.reset()
        if data_point.state.shape[0] != self.base_num_features:
            self.base_num_features = data_point.state.shape[0]
            self.feature_buffer = RollingFeatureBuffer(
                base_dim=self.base_num_features,
                lookback=self.lookback,
                windows=self.rolling_windows,
                stats=self.feature_stats,
            )

        self.feature_buffer.append(data_point.state.copy())
        
        # If prediction is not needed, return None
        if not data_point.need_prediction:
            return None
        
        if not self.feature_buffer.has_enough_history():
            return data_point.state.copy()

        window_tensor = torch.tensor(
            self.feature_buffer.get_window(), dtype=torch.float32
        ).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(window_tensor)
            prediction_np = prediction.squeeze(0).cpu().numpy()
        return prediction_np

    def _load_config(self) -> dict:
        possible_paths = [
            os.path.join(CURRENT_DIR, "model", "model_config.json"),
            os.path.join(CURRENT_DIR, "model_config.json"),
            "model/model_config.json",
            "model_config.json",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        return {}

    def _load_weights(self) -> None:
        requested_path = self.config.get("model_path", "model/lstm_with_features.pt")
        possible_paths = [
            os.path.join(CURRENT_DIR, requested_path),
            os.path.join(CURRENT_DIR, "model", "lstm_with_features.pt"),
            os.path.join(CURRENT_DIR, "lstm_with_features.pt"),
            requested_path,
        ]
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path:
            state = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(state)
            self.model.eval()
        else:
            print(f"Warning: Model file not found. Tried: {possible_paths}")
            print("Using randomly initialized weights; include model artifacts in submission.")


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

