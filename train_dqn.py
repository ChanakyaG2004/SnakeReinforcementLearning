import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym
import matplotlib.pyplot as plt
from snake_env import SnakeEnv  

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99  
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.memory = deque(maxlen=2000)
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_every = 10  

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)  
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        with torch.no_grad():
            next_q_values = torch.max(self.target_model(next_states), dim=1)[0]
            targets = rewards + (self.gamma * next_q_values * (1 - dones))

        current_q_values = self.model(states).gather(1, actions)

        loss = self.criterion(current_q_values, targets.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

def train_dqn(episodes):
    env = SnakeEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    plot_episodes = []
    plot_foods_eaten = []

    
    plt.ion()  
    fig, ax = plt.subplots()
    ax.set_xlim(0, episodes)
    ax.set_ylim(0, 100)  
    ax.set_xlabel('Episode')
    ax.set_ylabel('Number of Foods Eaten')
    ax.set_title('Number of Foods Eaten Over Episodes')
    line, = ax.plot([], [], lw=2)

    def update_plot():
        line.set_xdata(plot_episodes)
        line.set_ydata(plot_foods_eaten)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)  

    for e in range(episodes):
        state = env.reset()
        total_foods_eaten = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if reward > 0:  
                total_foods_eaten += 1

            agent.replay(agent.batch_size)
            env.render()

        plot_episodes.append(e + 1)
        plot_foods_eaten.append(env.score)  

        print(f"Episode: {e + 1}/{episodes}, Foods Eaten: {env.score}, Epsilon: {agent.epsilon}")

        if (e + 1) % agent.update_target_every == 0:
            agent.update_target_model()

        update_plot()

    plt.ioff()
    plt.show()
    env.close()

if __name__ == "__main__":
    episodes = 1000
    train_dqn(episodes)
