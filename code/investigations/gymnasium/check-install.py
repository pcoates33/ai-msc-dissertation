# Simple test from the https://github.com/Farama-Foundation/Gymnasium website to verify install

import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()
terminated = False
truncated = False
total_reward = 0

while not terminated and not truncated:
    if observation[2] > 0:
        action = 1
    else:
        action = 0
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"action {action} : Obs {observation}, reward {reward}, terminated {terminated}, truncated {truncated}, info {info}")
    total_reward += reward

print(f"total reward = {total_reward}")
# input('enter something to close')
env.close()
