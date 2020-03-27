from bit_flip_env import Env
from agent import Agent
import random
import numpy as np

n_bits = 11
lr = 1e-3
gamma = 0.98
MAX_EPISODE_NUM = 100000
memory_size = 1e+6
batch_size = 128
k_future = 4

if __name__ == "__main__":
    print(f"Number of bits:{n_bits}")
    env = Env(n_bits)
    agent = Agent(n_bits=n_bits, lr=lr, memory_size=memory_size, batch_size=batch_size, gamma=gamma)

    for episode_num in range(MAX_EPISODE_NUM):
        state, goal = env.reset()
        episode_reward = 0
        episode = []
        done = False
        while not done:
            action = agent.choose_action(state, goal)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, done, next_state, goal)
            episode.append((state.copy(), action, reward, done, next_state.copy()))
            loss = agent.learn()
            state = next_state
            episode_reward += reward

        # HER
        for i, transition in enumerate(episode):
            state, action, reward, done, next_state = transition
            future_transitions = random.choices(episode[i:], k=k_future)
            for future_transition in future_transitions:
                new_goal = future_transition[-1]
                if np.sum(next_state == new_goal) == n_bits:
                    reward = 0
                else:
                    reward = -1

                agent.store(state, action, reward, done, next_state, new_goal)

        agent.update_epsilon()

        if episode_num == 0:
            global_running_r = episode_reward
        else:
            global_running_r = 0.99 * global_running_r + 0.01 * episode_reward
        if episode_num % 1500 == 0:
            print(f"Ep:{episode_num}| "
                  f"Ep_r:{episode_reward:3.3f}| "
                  f"Ep_running_r:{global_running_r:3.3f}| "
                  f"Loss:{loss:3.3f}| "
                  f"Epsilon:{agent.epsilon:3.3f}| "
                  f"Mem_size:{len(agent.memory)}")
