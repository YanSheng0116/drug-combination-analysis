import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# === Load data ===
X = np.load("data/processed/X.npy")
with open("data/processed/y.pkl", "rb") as f:
    y = pickle.load(f)

# === Preprocess: binary reward, sample subset ===
threshold = 20
rewards = (y > threshold).astype(np.float32).reshape(-1)

# For quick testing, sample a small subset
X_small = X
rewards_small = rewards

# Convert to tensor
X_tensor = torch.tensor(X_small, dtype=torch.float32)
reward_tensor = torch.tensor(rewards_small, dtype=torch.float32)

# === Define Actor and Critic networks ===
class Actor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output logits for 2 actions
        )

    def forward(self, x):
        return self.model(x)

class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output value
        )

    def forward(self, x):
        return self.model(x).squeeze(1)

# === Initialize models and optimizers ===
actor = Actor(X_tensor.shape[1])
critic = Critic(X_tensor.shape[1])
actor_optim = optim.Adam(actor.parameters(), lr=1e-3)
critic_optim = optim.Adam(critic.parameters(), lr=1e-3)

# === Training ===
actor_losses, critic_losses, avg_rewards, action_ratios = [], [], [], []

for epoch in range(10):
    logits = actor(X_tensor)
    probs = torch.softmax(logits, dim=1)
    dist = torch.distributions.Categorical(probs)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)

    selected_rewards = torch.where(
    actions == 1,
    reward_tensor * 1.0 + (1 - reward_tensor) * (-0.2),
    torch.zeros_like(reward_tensor)
)
    values = critic(X_tensor)
    advantage = selected_rewards - values.detach()

    # Loss functions
    actor_loss = -(log_probs * advantage).mean()
    critic_loss = nn.functional.mse_loss(values, selected_rewards)

    # Optimize
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    # Logging
    actor_losses.append(actor_loss.item())
    critic_losses.append(critic_loss.item())
    avg_rewards.append(selected_rewards.mean().item())
    action_ratios.append((actions == 1).float().mean().item())

    print(f"Epoch {epoch+1}: reward={avg_rewards[-1]:.4f}, actor_loss={actor_loss.item():.4f}, critic_loss={critic_loss.item():.4f}")


# === Plotting ===
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(actor_losses, marker='o')
plt.title("Actor Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")

plt.subplot(2, 2, 2)
plt.plot(critic_losses, marker='s', color='orange')
plt.title("Critic Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")

plt.subplot(2, 2, 3)
plt.plot(avg_rewards, marker='^', color='green')
plt.title("Average Reward (Action = 1)"); plt.xlabel("Epoch"); plt.ylabel("Avg Reward")

plt.subplot(2, 2, 4)
plt.plot(action_ratios, marker='d', color='purple')
plt.title("Action=1 Recommendation Ratio"); plt.xlabel("Epoch"); plt.ylabel("Ratio")

plt.tight_layout()
plt.show()
