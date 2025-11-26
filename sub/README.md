# Submission Artifacts

This folder contains everything required by the competition runtime:

- `solution.py`: Loads the LSTM model and performs online rolling-feature inference.
- `model/lstm_with_features.pt`: Serialized weights for the 3-layer, 256-hidden LSTM trained with engineered inputs.
- `model/model_config.json`: Hyperparameters plus feature-window definitions shared by training and inference.

Training (optional) can be reproduced by running:

```bash
cd ..  # repository root
./venv/bin/python training/train_lstm_with_features.py
```

Packaging for submission (run from inside `sub/`):

```bash
zip -r ../submission.zip .
```

