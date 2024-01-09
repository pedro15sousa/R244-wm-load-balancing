import park
import numpy as np

env = park.make('load_balance')

episodes = 10
rewards = []

observations = np.array([])
rewards = np.array([])
actions = np.array([])
terminals = np.array([])

for i in range(episodes):
    obs = env.reset()
    a_rollout = []
    s_rollout = []
    r_rollout = []
    d_rollout = []
    done = False
    episode_reward = 0
    while not done:
        act = env.action_space.sample()
        s, r, done, _ = env.step(act)
    
        a_rollout += [act]
        s_rollout += [s]
        r_rollout += [r]
        d_rollout += [done]
        episode_reward += r
        if done:
            # print("s_rollout: ", s_rollout)
            observations = np.append(observations, np.array(s_rollout))
            rewards = np.append(rewards, np.array(r_rollout))
            actions = np.append(actions, np.array(a_rollout))
            terminals = np.append(terminals, np.array(d_rollout))


    np.append(rewards, -episode_reward)  # Store the negative of the reward

print(len(observations))
print(terminals)
print(actions)
print(rewards)
