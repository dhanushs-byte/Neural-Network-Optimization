# Neural Network Tasks 1–4

## Files
- `task1_forward_pass.py`: NumPy forward pass for a 3-4-1 network.
- `task2_activation_loss.py`: Softmax, activation plots, MSE, and binary cross-entropy.
- `task3_gradient_descent.py`: Chain-rule gradients, 50 gradient-descent steps, and learning-rate plots.
- `task4_optimizers.py`: SGD, SGD momentum, Adam, Dropout, BatchNorm, early stopping, and StepLR.
- `requirements.txt`: Python dependencies.

## Install
```bash
python -m pip install -r requirements.txt
```

## Run all tasks
```bash
python task1_forward_pass.py
python task2_activation_loss.py
python task3_gradient_descent.py
python task4_optimizers.py
```

## Task 3 hand calculation
For `L = (w*x + b - t)^2`, `x=2`, `t=10`, `w=1`, `b=0`, the initial loss is 64.
- `dL/dw = 2*(2-10)*2 = -32`
- `dL/db = 2*(2-10) = -16`

## Hyperparameter table
| Hyperparameter | Too high | Too low |
|---|---|---|
| Learning rate | Unstable or diverging loss | Slow convergence |
| Batch size | More memory, fewer updates per epoch | Noisy gradients |
| Epochs | Overfitting or wasted time | Underfitting |
| Weight decay | Underfitting | Greater risk of overfitting |

## Concepts
SGD updates parameters from mini-batch gradients. Momentum smooths update direction. Adam adapts parameter-wise updates. AdamW decouples weight decay from the gradient update. Dropout reduces co-adaptation, BatchNorm normalizes activations, early stopping monitors validation loss, and StepLR reduces the learning rate on a schedule.

Charts are saved as PNG files in the current working directory when scripts are run.
