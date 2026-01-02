import torch
import torch.optim as optim
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from collections import deque
from godot_env import GodotEnv
from ppo_model import ActorCritic

# --- Hyperparameters ---
MAX_EPISODES = 10000       
MAX_STEPS_PER_EP = 1000    
UPDATE_TIMESTEP = 4000     
LR = 0.0003                
GAMMA = 0.99               
EPS_CLIP = 0.2             
K_EPOCHS = 10              
MODEL_PATH = "quadruped_ppo1.pth"

class TrainingLogger:
    def __init__(self):
        self.episodes = []
        self.rewards = []
        self.avg_rewards = []
        self.steps = []
        self.losses = []
        
        # Setup the plot
        plt.ion() # Turn on interactive mode
        self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.tight_layout(pad=4.0)
        
    def update(self, ep, reward, step, loss=None):
        self.episodes.append(ep)
        self.rewards.append(reward)
        self.steps.append(step)
        
        # Calculate moving average (last 50 episodes)
        avg = np.mean(self.rewards[-50:])
        self.avg_rewards.append(avg)
        
        if loss is not None:
            self.losses.append(loss)
            
        self.plot()

    def plot(self):
        self.axs[0].clear()
        self.axs[0].set_title("Training Rewards")
        self.axs[0].plot(self.episodes, self.rewards, label="Raw Reward", alpha=0.3, color='blue')
        self.axs[0].plot(self.episodes, self.avg_rewards, label="Moving Avg (50)", color='red', linewidth=2)
        self.axs[0].set_ylabel("Total Reward")
        self.axs[0].legend()

        self.axs[1].clear()
        self.axs[1].set_title("Episode Length (Steps survived)")
        self.axs[1].plot(self.episodes, self.steps, color='green')
        self.axs[1].set_ylabel("Steps")
        self.axs[1].set_xlabel("Episode")
        
        plt.pause(0.01) # Brief pause to allow the UI to update

def train():
    env = GodotEnv()
    logger = TrainingLogger()
    
    policy = ActorCritic(env.state_dim, env.action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            policy.load_state_dict(checkpoint)
            print("Successfully loaded 38-dim policy.")
        except:
            print("Starting training from scratch.")

    states, actions, log_probs, rewards, d_buffer = [], [], [], [], []
    timestep = 0
    last_loss = 0.0
    
    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        assert len(state) == 38, f"Invalid state dim: {len(state)}"
        episode_reward = 0
        
        for t in range(MAX_STEPS_PER_EP):
            timestep += 1
            
            action, log_prob = policy.get_action(state)
            next_state, reward, done = env.step(action) 
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            d_buffer.append(done)
            
            state = next_state
            episode_reward += reward
            
            if timestep % UPDATE_TIMESTEP == 0:
                print(f"\n--- Updating Brain (Step {timestep}) ---")
                last_loss = update_policy(policy, optimizer, states, actions, log_probs, rewards, d_buffer)
                states, actions, log_probs, rewards, d_buffer = [], [], [], [], []
                torch.save(policy.state_dict(), MODEL_PATH)
            
            if done: break
        
        # Update our graphical dashboard
        logger.update(episode, episode_reward, t+1, last_loss)
        
        if episode % 10 == 0:
            print(f"Ep: {episode} | Steps: {t+1} | Reward: {episode_reward:.2f}")

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages, dtype=torch.float32)

def update_policy(policy, optimizer, states, actions, log_probs, rewards, dones):
    states_tensor = torch.FloatTensor(np.stack(states))
    actions_tensor = torch.FloatTensor(np.stack(actions))
    old_log_probs = torch.stack(log_probs).detach()

    lprobs, s_values, d_entropy = policy.evaluate(states_tensor, actions_tensor)

    # --- GAE ---
    with torch.no_grad():
        advantages = compute_gae(rewards, s_values.tolist(), dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        returns = advantages + s_values

    total_loss = 0.0

    for _ in range(K_EPOCHS):
        lprobs, s_values, d_entropy = policy.evaluate(states_tensor, actions_tensor)

        ratio = torch.exp(lprobs - old_log_probs)
        s1 = ratio * advantages
        s2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

        actor_loss = -torch.min(s1, s2).mean()
        value_loss = torch.nn.functional.mse_loss(s_values, returns)
        entropy = d_entropy.mean()

        loss = actor_loss + 0.2 * value_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / K_EPOCHS


if __name__ == "__main__":
    # Ensure plot window stays open at the end
    try:
        train()
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        plt.ioff()
        plt.show()