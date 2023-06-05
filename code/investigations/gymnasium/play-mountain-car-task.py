# Semi-gradient sarsa for estimating q

import gymnasium as gym
import numpy as np
import random


class Tiling:

    def __init__(self, x_min, x_max, y_min, y_max, num_tilings=8) -> None:
        self.num_tilings = num_tilings
        self.grid_size = (num_tilings+1) * (num_tilings+1)
        self.tiling_vector_size = self.grid_size * num_tilings

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        x_range = x_max - x_min
        y_range = y_max - y_min

        self.x_tile_size = x_range / num_tilings
        self.y_tile_size = y_range / num_tilings

        self.x_tile_offset = self.x_tile_size / num_tilings
        self.y_tile_offset = self.y_tile_size / num_tilings

    def to_vector(self, x, y):
            # Determine the tiles that the x and y fall in and return a suitable vector.
            # Use x as x and y as y access.
            # If num_tilings is 8, then create a 9x9 grid, put the bottom left at min of x and y.
            # Think of the grid as a vector of size 81, bottom left is item 0, to the right of that is item 1
            # and so on. Then the next row up starts with item 9. The vector will be all zeros appart from the
            # tile that contains the x and y. This will be set to 1.
            # Then move the grid left and down by 1/8 of the tile size and repeat 7 more times to give 9 vectors
            # each of length 81. Concatenate these together to give a binary representation of the state.
            # Returns the concatenated vector.

            # first get how far into the range x and y are:
            x_absolute = max(self.x_min, min(self.x_max, x)) - self.x_min
            y_absolute = max(self.y_min, min(self.y_max, y)) - self.y_min
            # then see what tile that drops them into
            x_tile = int(x_absolute // self.x_tile_size)
            y_tile = int(y_absolute // self.y_tile_size)
            x_remainder = x_absolute % self.x_tile_size
            y_remainder = y_absolute % self.y_tile_size
            state_vector = np.zeros(self.tiling_vector_size, dtype=np.int8)
            tiling_offset = 0
            for tile_num in range(0, self.num_tilings):
                tile_offset = tiling_offset + y_tile * (self.num_tilings+1) + x_tile 
                if x_remainder < 0:
                    tile_offset += 1
                    x_remainder = self.x_tile_size
                else:
                    x_remainder -= self.x_tile_offset
                if y_remainder < 0:
                    tile_offset += self.num_tilings + 1
                    y_remainder = self.y_tile_size
                else:
                    y_remainder -= self.y_tile_offset
                state_vector[tile_offset] = 1
                tiling_offset += self.grid_size

            return state_vector
        

class StateActionValue:

    def __init__(self, tilings=None) -> None:
        self.actions = [-1, 0, 1]
        # self.weights = {action: np.random.rand(tilings.tiling_vector_size) for action in self.actions}
        self.weights = {action: np.zeros(tilings.tiling_vector_size) for action in self.actions}
        self.tilings = tilings
        
    def all_values(self, state):
        state_vector = self.tilings.to_vector(*state) 
        return {action: np.sum(self.weights[action]*state_vector) for action in self.actions}
    
    def value(self, state, action):
        state_vector = self.tilings.to_vector(*state) 
        return np.sum(self.weights[action]*state_vector)
    
    def update(self, state, action, adjustment):
        w = self.weights[action]
        state_vector = self.tilings.to_vector(*state)
        w += state_vector * adjustment

    def display_weights(self):
        pass

class Policy:

    def __init__(self, q_hat, epsilon) -> None:
        self.q_hat = q_hat
        self.epsilon = epsilon

    def e_greedy(self, state):
        # Get the action to use next.
        if random.random() < self.epsilon:
            return np.random.choice(self.q_hat.actions)
        else:
            # Get the action that provides the maxiumum state action value
            all_values = self.q_hat.all_values(state)
            return max(all_values, key=all_values.get)


q_hat = StateActionValue(tilings=Tiling(x_min=-1.2, x_max=0.6, y_min=-0.07, y_max=0.07))
policy = Policy(q_hat, 0.001)
discount_rate = 0.75 
step_size = 0.1/8
print(f'step size = {step_size}')
#env = gym.make("MountainCarContinuous-v0", render_mode="human")
env = gym.make("MountainCarContinuous-v0")
successful_runs = 0

for episode in range(1, 11):
    state, info = env.reset()

    action = policy.e_greedy(state)
    for i in range(5000):
        next_state, reward, terminated, truncated, debug_info = env.step([action])
        if reward == 0:
            reward = -1

        next_action = policy.e_greedy(next_state)

        adjustment = step_size * (reward + discount_rate * q_hat.value(state, action) + q_hat.value(next_state, next_action))
        # if i % 100 == 0:
        #     print(f'state:{state}, action:{action}, value:{q_hat.value(state, action)}')
        q_hat.update(state, action, adjustment)
        # if i % 100 == 0:
        #     print(f'adjustment:{adjustment}, new value:{q_hat.value(state, action)}')
        if terminated:
            print(f'episode = {episode} :) terminated at attempt {i}')
            successful_runs += 1
            break
        if truncated:
            print(f'episode = {episode} truncated at attempt {i}')
            break

        state = next_state
        action = next_action

    q_hat.display_weights()

print(f'There were {successful_runs} successful runs')

# for i in range(0, 25):
#     new_state, reward, terminated, truncated, debug_info = env.step([-1])
#     position, velocity = new_state
#     print(f'position = {position}, velocity = {velocity}, reward = {reward}')
#     print(state_tiling(position, velocity))

# for i in range(0, 50):
#     new_state, reward, terminated, truncated, debug_info = env.step([1])
#     position, velocity = new_state
#     print(f'position = {position}, velocity = {velocity}, reward = {reward}')
#     print(state_tiling(position, velocity))

