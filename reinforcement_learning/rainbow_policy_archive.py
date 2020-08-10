import copy
import os
import random
from collections import namedtuple, deque, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd 

from reinforcement_learning.model import RainbowDQN
from reinforcement_learning.policy import Policy

class RainbowPolicy(Policy):
    """Dueling Double DQN policy"""

    def __init__(self, state_size, action_size, parameters, evaluation_mode=False):
        # self.evaluation_mode = evaluation_mode

        self.state_size = state_size
        self.action_size = action_size
        self.num_atoms = parameters.num_atoms
        self.Vmin = parameters.Vmin
        self.Vmax = parameters.Vmax
        # self.double_dqn = True
        # self.hidsize = 1
        
        if not evaluation_mode:
            self.hidsize = parameters.hidden_size
            self.buffer_size = parameters.buffer_size
            self.batch_size = parameters.batch_size
            self.update_every = parameters.update_every
            self.learning_rate = parameters.learning_rate
            self.tau = parameters.tau
            self.gamma = parameters.gamma
            self.buffer_min_size = parameters.buffer_min_size
        
        self.USE_CUDA = torch.cuda.is_available()
        self.Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if self.USE_CUDA else autograd.Variable(*args, **kwargs)

        self.current_model = RainbowDQN(state_size, action_size, num_atoms=self.num_atoms, Vmin=self.Vmin, Vmax=self.Vmax).to(self.device)
        self.target_model = RainbowDQN(state_size, action_size, num_atoms=self.num_atoms, Vmin=self.Vmin, Vmax=self.Vmax).to(self.device)
                    
        self.optimizer = optim.Adam(self.current_model.parameters(), 0.001)
        
        self.replay_buffer = ReplayBuffer(10000)
        self.update_target(self.current_model, self.target_model)
        # if not evaluation_mode:
        #     self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
        #     self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        #     self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, self.device)

        #     self.t_step = 0
        #     self.loss = 0.0
        

    def act(self, state, eps=0.): 
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.current_model.eval()
        with torch.no_grad():
            action_values = self.current_model(state)
        self.current_model.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
                self._learn()

    def _learn(self):
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        if self.double_dqn:
            # Double DQN
            q_best_action = self.qnetwork_local(next_states).max(1)[1]
            q_targets_next = self.qnetwork_target(next_states).gather(1, q_best_action.unsqueeze(-1))
        else:
            # DQN
            q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(-1)

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Compute loss
        self.loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # Update target network
        # self._soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        
        ###start
        num_frames = 15000
        batch_size = 32
        gamma      = 0.99
        
        losses = []
        all_rewards = []
        episode_reward = 0
        
        for frame_idx in range(1, num_frames + 1):
            
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                state = env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
                
            if len(replay_buffer) > batch_size:
                loss = compute_td_loss(batch_size)
                losses.append(loss.data[0])
                
            if frame_idx % 1000 == 0:
                self.update_target(self.current_model, self.target_model)

    def projection_distribution(self, next_state, rewards, dones):
        batch_size  = next_state.size(0)
        
        delta_z = float(self.Vmax - self.Vmin) / (self.num_atoms - 1)
        support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
        
        next_dist   = self.target_model(next_state).data.cpu() * support
        next_action = next_dist.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
        next_dist   = next_dist.gather(1, next_action).squeeze(1)
            
        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones   = dones.unsqueeze(1).expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)
        
        Tz = rewards + (1 - dones) * 0.99 * support
        Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
        b  = (Tz - self.Vmin) / delta_z
        l  = b.floor().long()
        u  = b.ceil().long()
            
        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size).long()\
                        .unsqueeze(1).expand(batch_size, self.num_atoms)
    
        proj_dist = torch.zeros(next_dist.size())    
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
            
        return proj_dist
    
    def compute_td_loss(self, state, action, reward, next_state, done):
        # self.state_size, self.action_size, reward, next_state, done = self.replay_buffer.sample(batch_size) 
    
        state      = self.Variable(torch.FloatTensor(np.float32(state)))
        next_state = self.Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
        action     = self.Variable(torch.LongTensor(action))
        reward     = torch.FloatTensor(reward)
        done       = torch.FloatTensor(np.float32(done))
    
        proj_dist = self.projection_distribution(next_state, reward, done)
        
        dist = self.current_model(state)
        action = action.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1, self.num_atoms)
        dist = dist.gather(1, action).squeeze(1)
        dist.data.clamp_(0.01, 0.99)
        loss = -(self.Variable(proj_dist) * dist.log()).sum(1)
        loss  = loss.mean()
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        self.current_model.reset_noise()
        self.target_model.reset_noise()
        
        return loss

    # def _soft_update(self, local_model, target_model, tau):
    #     # Soft update model parameters.
    #     # θ_target = τ*θ_local + (1 - τ)*θ_target
    #     for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
    #         target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename):
        torch.save(self.qnetwork_local.state_dict(), filename + ".local")
        torch.save(self.qnetwork_target.state_dict(), filename + ".target")

    def load(self, filename):
        if os.path.exists(filename + ".local"):
            self.qnetwork_local.load_state_dict(torch.load(filename + ".local"))
        if os.path.exists(filename + ".target"):
            self.qnetwork_target.load_state_dict(torch.load(filename + ".target"))


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(np.expand_dims(state, 0), action, reward, np.expand_dims(next_state, 0), done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(self.__v_stack_impr([e.state for e in experiences if e is not None])) \
            .float().to(self.device)
        actions = torch.from_numpy(self.__v_stack_impr([e.action for e in experiences if e is not None])) \
            .long().to(self.device)
        rewards = torch.from_numpy(self.__v_stack_impr([e.reward for e in experiences if e is not None])) \
            .float().to(self.device)
        next_states = torch.from_numpy(self.__v_stack_impr([e.next_state for e in experiences if e is not None])) \
            .float().to(self.device)
        dones = torch.from_numpy(self.__v_stack_impr([e.done for e in experiences if e is not None]).astype(np.uint8)) \
            .float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def __v_stack_impr(self, states):
        sub_dim = len(states[0][0]) if isinstance(states[0], Iterable) else 1
        np_states = np.reshape(np.array(states), (len(states), sub_dim))
        return np_states
