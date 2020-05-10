import numpy as np
import random
from collections import namedtuple, deque

from models import Actor, Critic

import torch
import torch.optim as optim
import torch.nn.functional as F

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Deep Deterministic Policy Gradient agent
class Agent:
    
    def __init__(self, state_size, action_size, seed, actor_file=None, critic_file=None):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            actor_file: path of file containing trained weights of actor network
            critic_file: path of file containing trained weights of critic network
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        
        #actor network:
        self.actor_local = Actor(state_size,action_size,seed).to(device)
        self.actor_target = Actor(state_size,action_size,seed).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(),LR)
        
        #critic network
        self.critic_local = Critic(state_size,action_size,seed).to(device)
        self.critic_target = Critic(state_size,action_size,seed).to(device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(),LR)
        
        #load trained weights if needed
        if actor_file:
            weights = torch.load(actor_file)
            self.actor_local.load_state_dict(weights)
            self.actor_target.load_state_dict(weights)
            
        if critic_file:
            weights = torch.load(critic_file)
            self.critic_local.load_state_dict(weights)
            self.critic_target.load_state_dict(weights)
        
        #init replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step=0
            
    def act(self, state):
        """Returns actions for given state as per current Actor network.
        
        Params
        ======
            state (array_like): current state
            
        """
        state = torch.from_numpy(state).float().to(device)
        
        self.actor_local.eval()
        
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
            
        self.actor_local.train()
        
        return np.clip(action,-1,1)
    
    def step(self, state,action,reward,next_state,done):
        self.memory.add(state,action,reward,next_state,done)
        
        self.t_step = (self.t_step+1)%UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory)>BATCH_SIZE:
                self.learn(GAMMA)
                
    def learn(self,GAMMA):
        """Update value parameters using batch of experience tuples.
        Params
        ======
            gamma (float): discount factor
        """
        states,actions,rewards,next_states,dones = self.memory.sample()
        
        #update critic
        target_next_actions = self.actor_target(next_states)
        target_next_q = self.critic_target(next_states,target_next_actions)
        target_q = rewards+(GAMMA*target_next_q*(1-dones))
        
        local_q = self.critic_local(states,actions)
        
        critic_loss = F.mse_loss(local_q,target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        #update actor
        local_actions = self.actor_local(states)
        
        actor_loss = -self.critic_local(states,local_actions).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
        
    def soft_update(self, local_model, target_model, tau=TAU):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

            
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)            
        
        