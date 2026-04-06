import torch
import numpy as np
import matplotlib.pyplot as plt
from ddpm import MLP, NoiseScheduler
import os
from tqdm.auto import tqdm

def plot_loss(exp_name, title, filename):
    loss = np.load(f"exps/{exp_name}/loss.npy")
    val_loss = np.load(f"exps/{exp_name}/val_loss.npy")
    epochs = len(val_loss)
    steps_per_epoch = len(loss) // epochs
    avg_train_loss = loss.reshape((epochs, steps_per_epoch)).mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(avg_train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def generate_samples(model, noise_scheduler, num_samples, device='cpu'):
    model.eval()
    sample = torch.randn(num_samples, 2).to(device)
    timesteps = list(range(len(noise_scheduler)))[::-1]
    for i, t in enumerate(timesteps):
        t_tensor = torch.full((num_samples,), t, dtype=torch.long).to(device)
        with torch.no_grad():
            residual = model(sample, t_tensor)
        sample = noise_scheduler.step(residual, t_tensor[0], sample)
    return sample.cpu().numpy()

def plot_samples_grid(exp_name, num_timesteps, beta_schedule, title, filename):
    model = MLP(hidden_size=128, hidden_layers=3, emb_size=128, time_emb="sinusoidal", input_emb="sinusoidal")
    model.load_state_dict(torch.load(f"exps/{exp_name}/model.pth"))
    
    noise_scheduler = NoiseScheduler(num_timesteps=num_timesteps, beta_schedule=beta_schedule)
    
    counts = [50, 200, 500, 1000]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=16)
    
    for ax, n in zip(axes, counts):
        samples = generate_samples(model, noise_scheduler, n)
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.6, label='Generated')
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='lower right')
        ax.set_title(f"n={n}")
        ax.set_aspect('equal')
        
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def visualize_reverse_diffusion(exp_name, num_timesteps, num_samples, filename):
    model = MLP(hidden_size=128, hidden_layers=3, emb_size=128, time_emb="sinusoidal", input_emb="sinusoidal")
    model.load_state_dict(torch.load(f"exps/{exp_name}/model.pth"))
    noise_scheduler = NoiseScheduler(num_timesteps=num_timesteps, beta_schedule="linear")
    
    model.eval()
    sample = torch.randn(num_samples, 2)
    timesteps = list(range(len(noise_scheduler)))[::-1]
    
    frames = []
    for i, t in enumerate(timesteps):
        t_tensor = torch.full((num_samples,), t, dtype=torch.long)
        with torch.no_grad():
            residual = model(sample, t_tensor)
        sample = noise_scheduler.step(residual, t_tensor[0], sample)
        
        if (t % 10 == 0):
            frames.append((t, sample.cpu().numpy()))
            
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    fig.suptitle(f"Reverse Diffusion Process visualization for T={num_timesteps}", fontsize=16)
    axes = axes.flatten()
    
    for idx, (t, frame) in enumerate(frames[:10]):
        ax = axes[idx]
        ax.scatter(frame[:, 0], frame[:, 1], alpha=0.6, color='blue', label='Generated')
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='lower right', fontsize='small')
        ax.set_title(f"t={t}")
        ax.set_aspect('equal')
        
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    # 5. Training with different timesteps
    print("Plotting losses...")
    plot_loss("base", "Loss vs Epoch (T=100, linear)", "results/loss_T100.png")
    plot_loss("T300", "Loss vs Epoch (T=300, linear)", "results/loss_T300.png")
    plot_loss("T10", "Loss vs Epoch (T=10, linear)", "results/loss_T10.png")
    
    # 7. Comparing linear and cosine noise schedules
    plot_loss("cosine", "Loss vs Epoch (T=100, cosine)", "results/loss_cosine.png")
    
    # 6. Sampling from the Trained Diffusion Models
    print("Plotting sampling grids...")
    plot_samples_grid("base", 100, "linear", "Sampling T=100, linear schedule", "results/sampling_T100.png")
    plot_samples_grid("T10", 10, "linear", "Sampling T=10, linear schedule", "results/sampling_T10.png")
    plot_samples_grid("T300", 300, "linear", "Sampling T=300, linear schedule", "results/sampling_T300.png")
    
    # 7b. Sampling with cosine
    plot_samples_grid("cosine", 100, "cosine", "Sampling T=100, cosine schedule", "results/sampling_cosine.png")
    
    # 8. Visualizing the Reverse Diffusion Process
    print("Plotting reverse diffusion...")
    visualize_reverse_diffusion("base", 100, 1000, "results/reverse_diffusion.png")
    print("Done!")
