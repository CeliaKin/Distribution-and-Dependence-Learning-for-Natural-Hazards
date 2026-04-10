# BASIC FLOW ON WINTER WINDSTORM DATA


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('winterwindstormwinddata.csv')
raw_losses = df['loss'].dropna().values
log_losses = np.log(raw_losses)


mu, std = log_losses.mean(), log_losses.std()
scaled_losses = (log_losses - mu) / std

x_t = torch.tensor(scaled_losses, dtype=torch.float32).view(-1, 1)

class MonotonicFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def log_prob(self, x):
        x.requires_grad_(True)
        z = self.net(x)

        dz_dx = torch.autograd.grad(z.sum(), x, create_graph=True)[0]

        log_prior = -0.5 * (z**2 + np.log(2 * np.pi))
        return log_prior + torch.log(dz_dx.abs() + 1e-6)

# Training
model = MonotonicFlow()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Training Normalizing Flow")
for i in range(2500):
    optimizer.zero_grad()
    loss = -model.log_prob(x_t).mean()
    loss.backward()
    optimizer.step()
    if i % 500 == 0:
        print(f"Iteration {i}: Loss {loss.item():.4f}")


x_range = torch.linspace(x_t.min()-1, x_t.max()+1, 500).view(-1, 1)


model.eval() 
log_p = model.log_prob(x_range)
pdf = torch.exp(log_p).detach().numpy()

plt.figure(figsize=(10, 6))

plt.hist(x_t.detach().numpy(), bins=50, density=True, alpha=0.3, 
         color='gray', label='Actual Data (Scaled Log)', edgecolor='white')

plt.plot(x_range.detach().numpy(), pdf, color='#0077b6', lw=2.5, label='Normalizing Flow PDF')

plt.gca().spines[['top', 'right']].set_visible(False)
plt.title("Normalizing Flow Density Estimation", fontsize=14, fontweight='bold')
plt.xlabel("Scaled Log Loss", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend(frameon=False)

plt.tight_layout()
plt.show()

