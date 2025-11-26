import json
import os
from dataclasses import dataclass, asdict
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


@dataclass
class TrainingConfig:
    lookback: int = 16
    hidden_size: int = 256
    num_layers: int = 3
    dropout: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 25
    rolling_windows: Sequence[int] = (5, 15, 30)
    feature_stats: Sequence[str] = ("mean", "std")
    min_step: int = 0
    train_split: float = 0.8
    model_dir: str = os.path.join("sub", "model")
    model_path: str = os.path.join("sub", "model", "lstm_with_features.pt")
    config_path: str = os.path.join("sub", "model", "model_config.json")


class MultiFeatureLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.linear(last_hidden)


def add_rolling_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    windows: Sequence[int],
    stats: Sequence[str],
) -> pd.DataFrame:
    df = df.sort_values(["seq_ix", "step_in_seq"]).copy()
    grouped = df.groupby("seq_ix", group_keys=False)
    for window in windows:
        rolling = grouped[feature_cols].rolling(window=window, min_periods=1)
        if "mean" in stats:
            df_mean = rolling.mean().reset_index(level=0, drop=True)
            df_mean.columns = [f"{col}_rollmean_{window}" for col in feature_cols]
            df[df_mean.columns] = df_mean
        if "std" in stats:
            df_std = rolling.std(ddof=0).reset_index(level=0, drop=True)
            df_std = df_std.fillna(0.0)
            df_std.columns = [f"{col}_rollstd_{window}" for col in feature_cols]
            df[df_std.columns] = df_std
    return df


def create_sequence_dataset(
    df: pd.DataFrame,
    input_cols: Sequence[str],
    target_cols: Sequence[str],
    lookback: int,
    min_step: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, List[int]]:
    df = df.sort_values(["seq_ix", "step_in_seq"])
    X, y, seq_ids = [], [], []
    for seq_id, seq_df in df.groupby("seq_ix"):
        seq_df = seq_df[seq_df["step_in_seq"] >= min_step]
        if len(seq_df) <= lookback:
            continue
        inputs = seq_df[input_cols].to_numpy(dtype=np.float32)
        targets = seq_df[target_cols].to_numpy(dtype=np.float32)
        for i in range(len(inputs) - lookback):
            X.append(inputs[i : i + lookback])
            y.append(targets[i + lookback])
            seq_ids.append(seq_id)
    return (
        torch.tensor(np.stack(X), dtype=torch.float32),
        torch.tensor(np.stack(y), dtype=torch.float32),
        seq_ids,
    )


def train_model(config: TrainingConfig) -> None:
    df = pd.read_csv("competition_package/datasets/train.csv")
    base_feature_cols = [col for col in df.columns if col not in ("seq_ix", "step_in_seq", "need_prediction")]
    df = add_rolling_features(df, base_feature_cols, config.rolling_windows, config.feature_stats)

    engineered_cols = [col for col in df.columns if col not in ("seq_ix", "step_in_seq", "need_prediction")]

    unique_seq = sorted(df["seq_ix"].unique())
    split_idx = int(len(unique_seq) * config.train_split)
    train_ids = set(unique_seq[:split_idx])
    val_ids = set(unique_seq[split_idx:])

    train_df = df[df["seq_ix"].isin(train_ids)].copy()
    val_df = df[df["seq_ix"].isin(val_ids)].copy()

    X_train, y_train, _ = create_sequence_dataset(
        train_df, engineered_cols, base_feature_cols, config.lookback, config.min_step
    )
    X_val, y_val, _ = create_sequence_dataset(
        val_df, engineered_cols, base_feature_cols, config.lookback, config.min_step
    )

    model = MultiFeatureLSTM(
        input_size=len(engineered_cols),
        output_size=len(base_feature_cols),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    train_dataset = data.TensorDataset(X_train, y_train)
    val_dataset = data.TensorDataset(X_val, y_val)
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=config.batch_size)

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= max(1, len(train_loader))

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                val_loss += criterion(preds, batch_y).item()
            val_loss /= max(1, len(val_loader))
        print(f"Epoch {epoch+1}/{config.num_epochs}: train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}")

    os.makedirs(config.model_dir, exist_ok=True)
    torch.save(model.state_dict(), config.model_path)
    config_payload = asdict(config)
    config_payload.update(
        {
            "input_feature_size": len(engineered_cols),
            "output_feature_size": len(base_feature_cols),
            "base_num_features": len(base_feature_cols),
            "rolling_windows": list(config.rolling_windows),
            "feature_stats": list(config.feature_stats),
        }
    )
    with open(config.config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)
    print(f"Saved weights to {config.model_path}")
    print(f"Saved config to {config.config_path}")


if __name__ == "__main__":
    train_model(TrainingConfig())

