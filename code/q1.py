import torch
import matplotlib.pyplot as plt
import os

# Function definitions
def theta(x1, x2):
    return torch.atan2(x2, x1)

def r(x1, x2):
    return torch.sqrt(x1**2 + x2**2)

def y(x1, x2):
    r_eval = r(x1, x2)
    theta_eval = theta(x1, x2)
    return r_eval**2 * (torch.sin(6 * theta_eval + 2 * r_eval)**2 + 1)

# Gradient definitions
def grad_theta_x1(x1, x2):
    return -x2 / (x1**2 + x2**2)

def grad_theta_x2(x1, x2):
    return x1 / (x1**2 + x2**2)

def grad_r_x1(x1, x2):
    r_eval = r(x1, x2)
    return x1 / r_eval

def grad_r_x2(x1, x2):
    r_eval = r(x1, x2)
    return x2 / r_eval

def grad_y_theta(reval, thetaeval):
    return 12 * reval**2 * torch.sin(6 * thetaeval + 2 * reval) * torch.cos(6 * thetaeval + 2 * reval)

def grad_y_r(reval, thetaeval):
    return (4 * reval**2 * torch.sin(6 * thetaeval + 2 * reval) * torch.cos(6 * thetaeval + 2 * reval) + 
            2 * reval * (torch.sin(6 * thetaeval + 2 * reval)**2 + 1))

def grad_y_x1(x1, x2):
    reval = r(x1, x2)
    thetaeval = theta(x1, x2)
    
    dy_dtheta = grad_y_theta(reval, thetaeval)
    dy_dr = grad_y_r(reval, thetaeval)
    
    dtheta_dx1 = grad_theta_x1(x1, x2)
    dr_dx1 = grad_theta_x2(x1, x2)
    
    return dy_dtheta * dtheta_dx1 + dy_dr * dr_dx1

def grad_y_x2(x1, x2):
    reval = r(x1, x2)
    thetaeval = theta(x1, x2)
    
    dy_dtheta = grad_y_theta(reval, thetaeval)
    dy_dr = grad_y_r(reval, thetaeval)
    
    dtheta_dx2 = grad_theta_x2(x1, x2)
    dr_dx2 = grad_theta_x2(x1, x2)
    
    return dy_dtheta * dtheta_dx2 + dy_dr * dr_dx2

# Main script
seeds = [1, 2, 3, 4, 5]  # seeds
num_steps = 2000  # maximum number of steps
tol = 1e-3  # error tolerance
lams_list = [1e-4, 1e-3, 1e-2, 1e-1, 1]  # step size list

# For plotting
x1 = torch.linspace(-5, 5, 100)
x2 = torch.linspace(-5, 5, 100)
X1, X2 = torch.meshgrid(x1, x2, indexing="ij")

Y = y(X1, X2)
os.makedirs("plots", exist_ok=True)

for lam_idx, lam in enumerate(lams_list):
    for seed in seeds:
        print("Running: lam =", lam, "seed =", seed)
        # Set the seed for reproducibility
        torch.manual_seed(seed)
        
        # Initialize a random starting point
        x1 = -5 + 10 * torch.rand(1)
        x2 = -5 + 10 * torch.rand(1)
        
        # Store the values for plotting
        y_vals = []
        x1_vals = [x1.item()]
        x2_vals = [x2.item()]
        
        for step in range(num_steps):
            # Evaluate the y value
            yeval = y(x1, x2)
            y_vals.append(yeval.item())
            
            # Calculate the gradients
            x1_grad = grad_y_x1(x1, x2)
            x2_grad = grad_y_x2(x1, x2)
            
            # Update the parameters
            x1 = x1 - lam * x1_grad
            x2 = x2 - lam * x2_grad
            
            # Store updated values for trajectory plotting
            x1_vals.append(x1.item())
            x2_vals.append(x2.item())
            
            # Check for convergence
            if y_vals[-1] < tol:
                break
        
        # Plot results
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # Contour plot with trajectory
        im = axs[0].contourf(X1.numpy(), X2.numpy(), Y.numpy(), cmap="Spectral", levels=100)
        axs[0].plot(x1_vals, x2_vals, linewidth=2, marker=".", color="black", markersize=2)
        axs[0].set_xlabel("X1")
        axs[0].set_ylabel("X2")
        fig.colorbar(im, ax=axs[0])
        
        # Loss over steps
        axs[1].plot(range(len(y_vals)), y_vals)
        axs[1].set_yscale("log")
        axs[1].set_xlabel("Step")
        axs[1].set_ylabel("y")
        axs[1].grid(True)
        
        fig.suptitle(f"Step size: {lam}, seed: {seed}")
        fig.tight_layout()
        fig.savefig(f"plots/q1_{lam}_{seed}.png", dpi=300)
        plt.clf()
        plt.close(fig)
