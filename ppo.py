from collections import namedtuple
from itertools import count
import os
import time
import numpy as np
import env
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

# Parameters
gamma = 0.99
render = True
seed = 1
log_interval = 10
N = 5
M = 3
Width = 180
fv = 20
Dn, Cn = np.random.uniform(300, 500, N), [600 for _ in range(N)]
fn = np.random.uniform(0.5, 1.5, N)
distance = np.random.uniform(0, 900, N)
P = np.random.uniform(40000, 60000, N)
mu, eta = np.zeros(N), 0.5
sigma, alpha = 1e-2, 2
threshold = 50000
T = 90
battery = 100

env = env.Environment(N, M, Width, fv, Dn, Cn, fn, distance, P, mu, eta, sigma, alpha, T, threshold, battery)
num_state = 26
# num_action = 64
num_users = 5
num_action_per_user = 4
# num_action = np.array(env.action.get_action_space()).shape[1]

torch.manual_seed(seed)
# env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class Actor(nn.Module):
    def __init__(self, num_state, num_action_per_user, num_users):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100).float()
        self.action_head = nn.Linear(100, num_action_per_user * num_users).float()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_logits = self.action_head(x)
        # Reshape logits to (batch_size, num_users, num_action_per_user)
        action_logits = action_logits.view(-1, num_users, num_action_per_user)
        # Apply softmax to ensure the sum of probabilities for each user is 1
        action_prob = F.softmax(action_logits, dim=-1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, num_state):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.1
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 2000
    batch_size = 64

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor(num_state, num_action_per_user, num_users)
        self.critic_net = Critic(num_state)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter('./exp')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-4)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 1e-4)
        if not os.path.exists('./param'):
            os.makedirs('./param/net_param')
            os.makedirs('./param/img')

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
            # print(action_prob)
            # 创建 Categorical 分布对象
        c = Categorical(action_prob[0])  # action_prob[0] 选择批量中的第一个样本
        # 从每个用户的动作概率中采样动作
        actions = c.sample()  # 采样得到的动作，形状为 (num_users,)
        # 将动作张量展平为列表
        actions = actions.flatten().tolist()  # 将动作张量转换为列表
        # 从动作概率中提取每个动作的概率
        action_probs = [action_prob[0, user, action].item() for user, action in enumerate(actions)]
        # print(action_probs)
        return actions, action_probs

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):  # sourcery skip: use-fstring-for-concatenation
        torch.save(self.actor_net.state_dict(), './param/net_param/actor_net' + str(time.time())[:10], +'.pkl')
        torch.save(self.critic_net.state_dict(), './param/net_param/critic_net' + str(time.time())[:10], +'.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        states = torch.tensor([t.state.flatten() for t in self.buffer], dtype=torch.float)
        actions = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1,
                                                                                       num_users)
        rewards = [t.reward for t in self.buffer]
        old_action_log_probs = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1,
                                                                                                         num_users)

        # 计算 Gt
        R = 0
        Gt = []
        for r in rewards[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)

        # PPO 更新
        for _ in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    print(f'I_ep {i_ep} ，train {self.training_step} times')

                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(states[index])
                delta = Gt_index - V
                advantage = delta.detach()

                action_probs = self.actor_net(states[index])  # 形状: (batch_size, num_users, num_action_per_user)
                action_probs = action_probs.gather(2, actions[index].unsqueeze(-1)).squeeze(
                    -1)  # 使用 gather 操作，形状: (batch_size, num_users)
                ratio = action_probs / old_action_log_probs[index]
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # 更新 actor 网络
                action_loss = -torch.min(surr1, surr2).mean()
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # 更新 critic 网络
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:]  # 清除经验


def main():  # sourcery skip: for-index-underscore
    agent = PPO()
    rewards_history = []  # 用于记录每个回合的奖励
    rewards_history_2 = []
    for i_epoch in range(300):
        seed = 0
        state = env.reset(seed)
        # env.clear_memory()

        total_reward = 0  # 记录每个 episode 的总奖励

        # PPO
        while True:
            action, action_prob = agent.select_action(state)
            next_state, reward, done, steps = env.step(action, seed)
            seed += 1

            trans = Transition(state, action, action_prob, reward, next_state)
            
            agent.store_transition(trans)
            state = next_state

            # total_reward += round(reward,3)  # 累加奖励

            if done :
                if len(agent.buffer) >= agent.batch_size:
                    agent.update(i_epoch)
                agent.writer.add_scalar('liveTime/livestep', steps, global_step=i_epoch)

                # 打印每个 episode 的总奖励和步数
                print(f"Episode {i_epoch} finished. Total reward: {reward}, Total steps: {steps}")
                rewards_history.append(reward)

                break

        # Random
        env.reset(seed)
        while True:
            _, reward_random, done_random, steps = env.step(None)
            if done_random:
                rewards_history_2.append(reward_random)
                print(f"Episode {i_epoch} finished. Random reward: {reward_random}, Total steps: {steps}")
                
                break

    # if len(rewards_history) >= 200:
    #     last_200_rewards = rewards_history[-200:]
    #     avg_reward_last_200 = np.mean(last_200_rewards)
    #     print(f"Average reward for the last 200 episodes: {avg_reward_last_200:.2f}")
    # else:
    #     print("Not enough episodes to calculate average reward for the last 200 episodes.")

    episode = [i for i in range(len(rewards_history))]
    if render: env.render(episode, rewards_history, rewards_history_2)



if __name__ == '__main__':
    main()
    


