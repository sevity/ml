#!/usr/bin/env python
# coding: utf-8

# 기본적인 동작확인을 위해서 엄청 단순화해서 해보는 버전

# In[1]:


import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


device


# In[3]:


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# 최근 state전이 pair 만개를 저장해두고 그야말로 랜덤하게 128개 배치뽑아서 트레이닝
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# In[4]:


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(10, 128)  # 2 input nodes, 50 in middle layers
        self.fc2 = nn.Linear(128, 2)  # 50 middle layer, 1 output node
        self.rl1 = nn.ReLU()
        self.rl2 = nn.ReLU()
        self.sm = nn.Softmax()
    def forward(self, x):
        x = self.fc1(x)
        x = self.rl1(x)
        x = self.fc2(x)
        x = self.sm(x)
        return x

# In[6]:


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


# 입실론 그리디가 적용되어 있고 입실론은 시간이 갈수록 줄어드는 구조인듯하다(EPS_DECAY)
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) *         math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            r = policy_net(state).max(1)[1].view(1, 1)
            print('state:',state[0], 'action', r)
            return r
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)


episode_durations = []

# 요거는 에피소드가 진행됨에 따라 몇스텝이나 살아남았는지를 y축에 두고 누적그래프를 그려주는 코드이다.
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# In[7]:


# 아래코드가 학습에 있어서 핵심인데..
# Q(s_t, a)랑 max Q(s_t+1, a)를 둘다 구해서 로스구하고 역전파하는 방식인데..
# stationary를 위해서 왼쪽거는 policy network, 오른쪽꺼는 target network을 사용하고
# target network은 10번 에피소드에 한번씩만 policy network과 동기화 해주는 구조 (어려운 개념은 없다.)
# 아래에서 배치사이즈 만큼 한번에 step돌리는거 같은데 그 구체적인 방법에 대해서 자세히 보지는 않았다. 나중에는 한번쯤은 봐야할듯
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        # 요개 내부 파라미터를 -1에서 1사이로 매번 조정해주는 기능같은데.. 왜 쓰는지 모르겠다.
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# In[8]:


# 아래가 메인루프라고 할 수 있겠다.
# 이게 돌려보니까.. 입실론 그리디에서 랜덤으로 빠질때는 에피소드는 하나 소모되지만 실제 네트웍학습은 스킵되고
# 리플레이 메모리 사이즈가 배치사이즈보다 작을때도 스킵되는등 초반에는 허수가 있다.
# 실제로는 50은 택도없고 꽤나 큰수가 돼야 수렴되기 시작하는걸로 보인다.

# 그리고 reward구조를 보면 step을 돌렸을때 살아남기만 하면 1.0을 주고
# pole의 기울기가 너무 커지면 done이 True가 되면서 에피소드가 끝나는 구조이다.
# 마지막에 done이 True일때 reward가 0으로 반영되는지는 확인안해봤는데 나중에 확인해보자. 0이라면 터미널 패널티가 있는 셈일듯 
num_episodes = 5000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    a = 1.0 * random.randrange(1,3)
    # a = 2.0
    state = torch.tensor([a] * 10)
    state = state.unsqueeze(0).to(device)

    for t in count():
        # Select and perform an action
        action = select_action(state)
        # _, reward, done, _ = env.step(action.item())
        reward = 0.0
        if action.item()==0:
            if state[0][0]>=2: reward = 100000.0
            # else : reward = -10.0
        # else: 
            # if state[0][0]<2: reward = 1.0
            # else : reward = -100.0

        done = False
        if random.randrange(1, 10)==1:
            done = True
        reward = torch.tensor([reward], device=device)
        print('reward',reward)

        # Store the transition in memory
        memory.push(state, action, state, reward)


        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
plt.ioff()
plt.show()


# In[ ]:




