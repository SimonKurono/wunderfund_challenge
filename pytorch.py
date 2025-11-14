import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# --- Data prep ---
df = pd.read_csv('competition_package/datasets/train.csv')
seq_1 = df[df['seq_ix'] == 0]
seq_1_0 = seq_1[['0']].values.astype('float32')
train_size = int(len(seq_1_0) * 0.80)
test_size = len(seq_1_0) - train_size
train, test = seq_1_0[:train_size], seq_1_0[train_size:]

lookback = 1

def create_dataset(dataset, lookback):
    """
    dataset: numpy array of shape (T, 1)
    returns:
        X: (N, lookback, 1)
        y: (N, 1)
    """
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i : i + lookback]        # (lookback, 1)
        target  = dataset[i + lookback]            # single step (1,)
        X.append(feature)
        y.append(target)
    X = torch.tensor(np.array(X), dtype=torch.float32)  # (N, lookback, 1)
    y = torch.tensor(np.array(y), dtype=torch.float32)  # (N, 1)
    return X, y

X_train, y_train = create_dataset(train, lookback=lookback)
X_test,  y_test  = create_dataset(test,  lookback=lookback)

print("X_train shape:", X_train.shape)  # (N_train, lookback, 1)
print("y_train shape:", y_train.shape)  # (N_train, 1)
print("X_test shape:",  X_test.shape)
print("y_test shape:",  y_test.shape)

# --- Model ---

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        x, _ = self.lstm(x)        # (batch, seq_len, hidden)
        x = x[:, -1, :]            # last timestep -> (batch, hidden)
        x = self.linear(x)         # (batch, 1)
        return x

model = LSTMModel()

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loader = data.DataLoader(
    data.TensorDataset(X_train, y_train),
    batch_size=8,
    shuffle=True
)

# --- Training loop ---

n_epochs = 100

for epoch in range(1, n_epochs + 1):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)                 # (batch, 1)
        loss = loss_fn(y_pred, y_batch)         # y_batch: (batch, 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # log every 10 epochs
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train)
            train_rmse = torch.sqrt(loss_fn(y_pred_train, y_train)).item()

            y_pred_test = model(X_test)
            test_rmse = torch.sqrt(loss_fn(y_pred_test, y_test)).item()

        print(f'Epoch {epoch}: train RMSE {train_rmse:.4f}, test RMSE {test_rmse:.4f}')

# --- Build plots ---

with torch.no_grad():
    # convert original series to 1D for plotting
    seq_np = seq_1_0.astype('float32').squeeze()        # shape (T,)

    # training predictions
    y_pred_train = model(X_train).detach().cpu().numpy().squeeze()  # (N_train,)
    # test predictions
    y_pred_test  = model(X_test ).detach().cpu().numpy().squeeze()  # (N_test,)

    # arrays for plotting (same length as original series)
    train_plot = np.full_like(seq_np, np.nan, dtype=np.float32)
    test_plot  = np.full_like(seq_np, np.nan, dtype=np.float32)

    # place predictions in the correct time positions
    # X_train samples cover indices [0 .. train_size-1-lookback]
    # targets align to indices [lookback .. train_size-1]
    train_plot[lookback:train_size] = y_pred_train

    # test: indices [train_size+lookback .. len(seq_np)-1]
    test_start = train_size + lookback
    test_end   = test_start + len(y_pred_test)          # should equal len(seq_np)
    test_plot[test_start:test_end] = y_pred_test

# --- Plot ---

plt.figure(figsize=(10,4))
plt.plot(seq_np,     label='True',  alpha=0.7)
plt.plot(train_plot, label='Train pred', alpha=0.8)
plt.plot(test_plot,  label='Test pred',  alpha=0.8)
plt.legend()
plt.title("LSTM one-step-ahead predictions")
plt.show()
