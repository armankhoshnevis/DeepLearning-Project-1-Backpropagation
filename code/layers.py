import torch

class BaseUnit:
    def __init__(self, lr):
        self.eval_mode = False
        self.lr = lr

    def eval(self):
        self.eval_mode = True

    def train(self):
        self.eval_mode = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

class Linear(BaseUnit):
    def __init__(self, d_in, d_out, lr=1e-3):
        super().__init__(lr)
        # Initialize weights (normal dist.: mean = 0, std = 0.05)
        self.W = torch.randn(d_in, d_out) * 0.05
        # Initialize bias
        self.b = torch.zeros(d_out)
        
        # Store input and output dimensions
        self.d_in = d_in
        self.d_out = d_out
        
        # Gradient placeholders
        self.h_W = None
        self.h_b = None
    
    def forward(self, X):
        n = X.shape[0]  # Batch size
        
        out = X @ self.W + self.b.unsqueeze(0)

        if not self.eval_mode:  # Training mode
            self.h_W = X  # Gradient w.r.t. weights
            self.h_b = torch.ones(n)  # Gradient w.r.t. bias

        return out
    
    def backward(self, grad):
        # grad is of shape n x d_out
        n = grad.shape[0]
        
        # Compute gradients
        grad_W = self.h_W.T @ grad  # X^T @ (dL/dh)
        grad_b = grad.sum(0)  # Sum over the batch dimension
        
        # Average the gradients over the batch dimension
        grad_W = grad_W / n
        grad_b = grad_b / n
        
        # Compute gradient for the previous layer (using W in the forward pass)
        grad_for_next = grad @ self.W.T  # (dL/dh) @ (dh/dX)
        
        # Update the parameters using the gradients
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
        
        return grad_for_next
    
class ReLU(BaseUnit):
    def __init__(self, lr=None):
        super().__init__(lr)
        self.sign = None  # Placeholder for sign of input

    def forward(self, X):
        if not self.eval_mode:
            # Store the sign of the input (binary mask)
            self.sign = (X > 0).float()  
        
        out = torch.clamp(X, min=0)  # Apply ReLU activation
        return out

    def backward(self, grad):
        grad_for_next = grad * self.sign
        return grad_for_next

class MSE(BaseUnit):
    def __init__(self, lr=None):
        super().__init__(lr)
        self.grad_return = None

    def forward(self, yhat, y):
        if not self.eval_mode:
            self.grad_return = yhat - y
        
        error = torch.mean((yhat - y) ** 2)
        return error

    def backward(self, grad=None):
        grad_for_next = 2 * self.grad_return / self.grad_return.shape[0]
        return grad_for_next
