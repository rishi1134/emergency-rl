from multiprocessing import Process
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
# from gymnasium.wrappers import AtariPreprocessing
from torch.nn.utils import clip_grad_norm_
from gymnasium.wrappers import FrameStackObservation as FrameStack
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import sys
import matplotlib.pyplot as plt
from env import SumoEnvironment
import multiprocessing as mp
# from sumo_rl.agents import QLAgent
# from sumo_rl.exploration import EpsilonGreedy

from custom_observation import CustomEmergencyObservationFunction
from custom_reward import emergency_reward_fn

class SharedActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions, is_discrete=True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fcX = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)
        self.softmax = nn.Softmax(dim=-1)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fcX(x))
        act_pro = self.softmax(self.fc2(x))
        value = self.fc3(x)
        return act_pro, value



    # SharedActorCritic(input_dim, self.hidden_dim, num_actions, is_discrete)
    
def worker(episodes, name="CartPole-v1", sync_freq=20, global_model=None, global_optimizer=None):
    env = SumoEnvironment(
            net_file="single-intersection/single-intersection.net.xml",
            route_file="single-intersection/single-intersection.rou.xml",
            # out_csv_name="out_csv",
            # use_gui=True,
            use_gui=False,
            num_seconds=200,
            min_green=5,
            max_green=10,
            observation_class = CustomEmergencyObservationFunction,
            reward_fn = emergency_reward_fn
        )
    obs = env.reset()
    # obs = np.array(obs)
    obs = torch.from_numpy(obs['t'])
    local_model = SharedActorCritic(12, 128, env.action_space.n, True)
    local_model.load_state_dict(global_model.state_dict())
    with torch.no_grad():
        actor, critic = local_model(obs)
    gamma, alpha = 0.99, 0.01
    optimizer = torch.optim.Adam(local_model.parameters(), lr=alpha)
    clip_grad_norm_(local_model.parameters(), max_norm=0.5)
    reward_per_episode = [0] * episodes
    freq = 1
    episode_rewards = []
    reward_per_episode = [0] * episodes
    wait_per_episode_lane_1 = [0] * episodes
    wait_per_episode_lane_2 = [0] * episodes
    for i in range(episodes):
        obs = env.reset()
        states = []
        actions = []
        rewards = []
        critics = []
        done = False
        timesteps = 0
        done = {"__all__": False}
        while not done["__all__"]:
            obs_tensor = torch.from_numpy(obs['t'])
            with torch.no_grad():
                actor, critic = local_model(obs_tensor)
            # print(actor.shape)
            action = torch.multinomial(actor, num_samples=1).item()
            next_obs, reward, done, info = env.step({'t': action})
            if len(info["emergency_waiting_time"]) > 0:
                for k,v in info["emergency_waiting_time"].items():
                    if k == "flow_emergency_ns_1":
                        wait_per_episode_lane_1[i] += v
                    elif k == "flow_emergency_ns_2":
                        wait_per_episode_lane_2[i] += v
            reward_per_episode[i] += reward['t']
            reward = reward['t']
            states.append(obs['t'])
            actions.append(action)
            rewards.append(reward)
            critics.append(critic.squeeze().mean())
            obs = next_obs
            # done = (timesteps >= max_timesteps)
            timesteps += 1
        freq += 1
        episode_reward = 0
        advantages = []
        critic_targets = []
        for t in reversed(range(len(rewards))):
            episode_reward = rewards[t] + gamma * episode_reward
            critic_targets.insert(0, episode_reward)
            advantage = episode_reward - critics[t].item()
            advantages.insert(0, advantage)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        critic_targets = torch.tensor(critic_targets, dtype=torch.float32)
        policy_losses = []
        critic_losses = []
        for t in range(len(states)):
            state_tensor = torch.tensor([states[t]], dtype=torch.float)
            actor, critic = local_model(state_tensor)
            log_prob = torch.log(actor[0, actions[t]])
            policy_losses.append(-log_prob * advantages[t])
            critic_losses.append(F.mse_loss(critic.squeeze(), critic_targets[t]))
        # policy_loss = -torch.stack(policy_gradients).sum()
        total_loss = torch.stack(policy_losses).sum() + torch.stack(critic_losses).sum()
        optimizer.zero_grad()
        total_loss.backward()
        # TODO: Check!
        if freq % sync_freq == 0:
            for g, l in zip(global_model.parameters(), local_model.parameters()):
                if g.grad is None:
                    g._grad = l.grad.clone()
                else:
                    g._grad += l.grad.clone()
            local_model.load_state_dict(global_model.state_dict())
            global_optimizer.step()
            global_optimizer.zero_grad()
        clip_grad_norm_(local_model.parameters(), max_norm=0.5)
        optimizer.step()
        episode_rewards.append(sum(rewards))
        print(f"Episode {i}, Total Reward: {sum(rewards)}")
        reward_per_episode[i] = sum(rewards)
        # np.save("wait_per_episode_lane_1.npy", wait_per_episode_lane_1)
        # np.save("wait_per_episode_lane_2.npy", wait_per_episode_lane_2)        
if __name__ == '__main__':
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)
    else:
        sys.exit("Please declare the environment variable 'SUMO_HOME'")
    env = SumoEnvironment(
        net_file="single-intersection/single-intersection.net.xml",
        route_file="single-intersection/single-intersection.rou.xml",
        out_csv_name="out_csv",
        # use_gui=True,
        use_gui=False,
        num_seconds=5000,
        min_green=5,
        max_green=10,
        observation_class = CustomEmergencyObservationFunction,
        reward_fn = emergency_reward_fn
    )
    name = "sumo"
    global_model = SharedActorCritic(12, 128, env.action_space.n, True)
    episodes = 1000
    global_model.share_memory()  
    global_optimizer = torch.optim.Adam(global_model.parameters(), lr=0.01)
    clip_grad_norm_(global_model.parameters(), max_norm=0.5)
    processes = []
    manager = mp.Manager()
    for i in range(4):
        p = Process(target=worker, args=(episodes, name, 1, global_model, global_optimizer))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()   
    torch.save(global_model.state_dict(), f"auto_{name}.pt")