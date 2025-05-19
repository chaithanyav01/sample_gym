import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

def run(episodes, is_training=True, render=False):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode='human' if render else None)

    if is_training:
        q = torch.zeros((env.observation_space.n, env.action_space.n), device=device)  # Initialize Q-table on GPU
    else:
        with open('frozen_lake8x8.pkl', 'rb') as f:
            q = torch.tensor(pickle.load(f), device=device)  # Load Q-table to GPU

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1
    epsilon_decay_rate = 0.0001
    rng = torch.Generator(device=device)

    rewards_per_episode = torch.zeros(episodes, device=device)

    for i in tqdm(range(episodes)):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if is_training and torch.rand(1, generator=rng, device=device) < epsilon:
                action = env.action_space.sample()
            else:
                action = torch.argmax(q[state, :]).item()

            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                q[state, action] += learning_rate_a * (
                    reward + discount_factor_g * torch.max(q[new_state, :]) - q[state, action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = torch.zeros(episodes, device=device)
    for t in range(episodes):
        sum_rewards[t] = torch.sum(rewards_per_episode[max(0, t-100):(t+1)])

    # Move to CPU before plotting (Matplotlib does not support GPU tensors)
    plt.plot(sum_rewards.cpu().numpy())
    plt.savefig('/home/chaithanyav/MTech Project/gym env/Frozen Lake V1/frozen_lake8x8.png')

    if is_training:
        with open("/home/chaithanyav/MTech Project/gym env/Frozen Lake V1/frozen_lake8x8.pkl", "wb") as f:
            pickle.dump(q.cpu().numpy(), f)  # Save Q-table as a CPU NumPy array

    print(q.cpu().numpy())

if __name__ == '__main__':
    run(1000, is_training=True, render=True)
