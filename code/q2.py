import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import itertools
from layers import Linear, ReLU, MSE

# Define the MLP class
class MLP:
    def __init__(self, d_x, d_h, lr):
        self.lin1 = Linear(d_x, d_h, lr=lr)
        self.act1 = ReLU()
        self.lin2 = Linear(d_h, d_h, lr=lr)
        self.act2 = ReLU()
        self.lin3 = Linear(d_h, 1, lr=lr)

    def forward(self, X):
        a1 = self.lin1.forward(X)
        h1 = self.act1.forward(a1)
        a2 = self.lin2.forward(h1)
        h2 = self.act2.forward(a2)
        yhat = self.lin3.forward(h2)
        return yhat

    def backward(self, g):
        g = self.lin3.backward(g)
        g = self.act2.backward(g)
        g = self.lin2.backward(g)
        g = self.act1.backward(g)
        g = self.lin1.backward(g)
        return g

    def train(self):
        self.lin1.train()
        self.act1.train()
        self.lin2.train()
        self.act2.train()
        self.lin3.train()

    def eval(self):
        self.lin1.eval()
        self.act1.eval()
        self.lin2.eval()
        self.act2.eval()
        self.lin3.eval()

# Load data
data = torch.load("Project1_data.pt", map_location="cpu", weights_only=True)

# Define hyperparameter ranges
learning_rates = [1e-1, 5e-2, 1e-2, 1e-3]
batch_sizes = [32, 64, 128]
epochs_list = [500, 1000, 2000]
layer_widths = [50, 100, 150, 200, 250]

# Logging setup
results = []
best_val_loss = float("inf")
best_model = None
best_hyperparams = None

# Loop over all hyperparameter combinations
for lr, batch_size, epochs, width in itertools.product(
    learning_rates, batch_sizes, epochs_list, layer_widths
    ):
    print(f"Training with lr={lr}, batch_size={batch_size}, epochs={epochs}, width={width}")

    # Initialize model and loss function
    mlp = MLP(2, width, lr=lr)
    mse = MSE()
    mse.train()

    # Create datasets and dataloaders
    train_dataset = TensorDataset(data["x_train"], data["y_train"].reshape(-1, 1))
    val_dataset = TensorDataset(data["x_val"], data["y_val"].reshape(-1, 1))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    train_loss_list = []
    val_loss_list = []
    train_step_count = []
    val_step_count = []
    step = 0
    
    for epoch in range(epochs):
        mlp.train()
        mse.train()
        for X, y in train_dataloader:
            yhat = mlp.forward(X)
            loss = mse.forward(yhat, y)
            g = mse.backward()
            g = mlp.backward(g)
            train_loss_list.append(loss.item())
            train_step_count.append(step)
            step += 1

        # Validation phase
        mlp.eval()
        mse.eval()
        val_loss = 0
        for X, y in val_dataloader:
            yhat = mlp.forward(X)
            val_loss += mse.forward(yhat, y).item()
        val_loss /= len(val_dataloader)
        val_loss_list.append(val_loss)
        val_step_count.append(step)

    # Log results
    results.append({
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "width": width,
        "train_loss": train_loss_list[-1],  # Final training loss
        "val_loss": val_loss_list[-1],      # Final validation loss
    })
    
    # Check if this is the best model
    if val_loss_list[-1] < best_val_loss:
        best_val_loss = val_loss_list[-1]
        best_model = mlp
        best_hyperparams = {
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "width": width,
        }
    
    # Plot training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(train_step_count, train_loss_list, label="Training Loss")
    plt.plot(val_step_count, val_loss_list, label="Validation Loss")
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Loss (lr={lr}, batch_size={batch_size}, epochs={epochs}, width={width})")
    plt.legend()
    plt.savefig(f"hpt/loss_lr{lr}_bs{batch_size}_ep{epochs}_w{width}.png")
    plt.close()

    # Plot test data colored by predicted y value
    mlp.eval()
    mse.eval()
    y_test = mlp.forward(data["x_test"]).squeeze()
    plt.figure(figsize=(8, 6))
    plt.scatter(data["x_test"][:, 0], data["x_test"][:, 1], c=y_test, cmap="viridis")
    plt.colorbar(label="Predicted Value")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(f"Test Data Predictions (lr={lr}, batch_size={batch_size}, epochs={epochs}, width={width})")
    plt.savefig(f"hpt/pred_lr{lr}_bs{batch_size}_ep{epochs}_w{width}.png")
    plt.close()# Save results to a file

with open("hyperparameter_tuning_results.txt", "w") as f:
    for result in results:
        f.write(f"lr={result['lr']}, batch_size={result['batch_size']}, epochs={result['epochs']}, width={result['width']}, "
                f"train_loss={result['train_loss']}, val_loss={result['val_loss']}\n")

# Save the best model and hyperparameters
print(f"Best hyperparameters: {best_hyperparams}")
print(f"Best validation loss: {best_val_loss}")

# Save the best model's predictions on the test set
best_model.eval()
mse.eval()
y_test_best = best_model.forward(data["x_test"]).squeeze()

# Save best model's predictions to a file
with open("best_model_predictions.txt", "w") as f:
    f.write("\n".join([str(x.item()) for x in y_test_best]))

# Plot test data colored by the best model's predictions
plt.figure()
plt.scatter(data["x_test"][:, 0], data["x_test"][:, 1], c=y_test_best, cmap="viridis")
plt.colorbar(label="Predicted Value")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"Test Data Predictions (Best Model: lr={best_hyperparams['lr']}, batch_size={best_hyperparams['batch_size']}, epochs={best_hyperparams['epochs']}, width={best_hyperparams['width']})")
plt.savefig("best_model_predictions.png")
plt.close()
